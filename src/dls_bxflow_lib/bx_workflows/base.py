import copy
import json
import logging
import sys

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.explain import explain2
from dls_utilpack.require import require

from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Remote execution.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Configurator.
from dls_bxflow_lib.bx_configurators.bx_configurators import (
    bx_configurators_get_default,
)

# Object managers we interact with.
from dls_bxflow_lib.bx_filestores.bx_filestores import bx_filestores_get_default
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs, bx_jobs_get_default

# Settings manager.
from dls_bxflow_lib.bx_settings.bx_settings import BxSettings

# Task-related imports.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks
from dls_bxflow_run.bx_tasks.constants import Keywords as BxTaskKeywords
from dls_bxflow_run.bx_variables.bx_variables import BxVariables

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class Base:
    """
    This workflow takes a filename and generates a workflow job for the beamline.
    """

    # ------------------------------------------------------------------
    def __init__(self, **kwargs):
        self.__bx_settings = BxSettings(callsign(self))

        # Make sure the constructor isn't being given unexpected kwargs.
        for keyword, value in kwargs.items():
            if keyword not in self.constructor_kwargs:
                raise RuntimeError(
                    f"keyword {keyword} is not a known workflow constructor argument"
                )

        # Change the constructor kwargs into object attributes and bx settings.
        for keyword, value in self.constructor_kwargs.items():
            # If defined as a dict, presume it is giving value, prompt, etc.
            if isinstance(value, dict):
                setting_specification = copy.deepcopy(value)
                value = setting_specification.get("value", "")
                # If no uuid given, just use the keyword.
                if setting_specification.get("uuid") in [None, ""]:
                    setting_specification["uuid"] = keyword
            # If defined as a scalar, then presume it is a string value.
            # Prompt for the setting will just be the keyword.
            else:
                setting_specification = {
                    "uuid": keyword,
                    "type": "dls_bxflow_lib.bx_settings.string",
                }

            # Let the runtime kwargs override the value in the settings.
            if keyword in kwargs:
                value = kwargs[keyword]
                logger.debug(
                    f"[KEYCON] keyword {keyword} gets value from runtime {value}"
                )

            # Resolve ${dotted-tokens} in the value from the bx_configurator's internal state.
            value = bx_configurators_get_default().resolve(value)
            logger.debug(f"[KEYCON] keyword {keyword} resolves to {value}")

            # Make this keyword an attribute of this object
            # so we can use self.keyword.
            setattr(self, keyword, value)

            # Put this keyword's value into the workflow's settings
            # so it gets saved in the database.
            setting_specification["value"] = value
            self.add_setting(setting_specification)

        # Build an empty job object.
        bxjob_specification = copy.deepcopy(bx_jobs_get_default().specification())
        bxjob_specification["label"] = callsign(self).split(".")[-1]
        self.bx_job = BxJobs().build_object(bxjob_specification)
        self.bx_variables = BxVariables()

        # Pin the job's directory when the workflow is first created.
        # TODO: Protect against pinning two jobs at the same timestamp in two processes or on two hosts.
        bx_filestores_get_default().pin_job_directory(self.bx_job)

        # Module from whence came this workflow.
        module_name = self.__class__.__module__
        module = sys.modules[module_name]
        module_file = module.__file__

        # Remember the filename::classname for the workflow.
        # This gets written into the database.
        self.__filename_classname = f"{module_file}::{self.__class__.__name__}"

        # Provide the current global configurator as a property.
        self.__bx_configurator = bx_configurators_get_default()

    # -----------------------------------------------------------------------------
    def _get_bx_configurator(self):
        return self.__bx_configurator

    def _del_bx_configurator(self):
        del self.__bx_configurator

    bx_configurator = property(
        fget=_get_bx_configurator,
        fdel=_del_bx_configurator,
        doc="The bx_configurator property.",
    )

    # ------------------------------------------------------------------
    def uuid(self):
        """
        Unique workflow id.
        """
        return self.bx_job.uuid()

    # ------------------------------------------------------------------
    def set_job_label(self, label):
        """
        Set the job's label sometime after the constructure has set it.
        This MUST be done before any tasks are created.
        """
        if len(self.bx_job.bx_tasks.list()) > 0:
            raise RuntimeError(
                "not allowed to set job label after any tasks are created"
            )

        # Set the label on the job object.
        self.bx_job.set_label(label)

        # Overpin the job's directory with the new label.
        bx_filestores_get_default().overpin_job_directory(self.bx_job)

    # ------------------------------------------------------------------
    def tasks(self, bx_task_label_pattern):
        return self.bx_job.bx_tasks.match_by_label(bx_task_label_pattern)

    # ------------------------------------------------------------------
    def assemble_remex_hints(self, bx_task_specification):
        """
        Assemble remex_hints from task and configuration.
        """

        thing_type = bx_task_specification.get("type", "no_type")
        label = bx_task_specification.get("label", "no_label")
        callsign = f"{thing_type} {label}"

        # Get the specification's remex hints.
        task_remex_hints = bx_task_specification.get(RemexKeywords.HINTS)

        # If a nonblank string, it should be looked up in the configurator.
        if task_remex_hints is None or task_remex_hints == "":
            task_remex_hints = {}
        elif isinstance(task_remex_hints, str):
            try:
                task_remex_hints = self.bx_configurator.require(
                    f"predefined_remex_hints.{task_remex_hints}"
                )
            except NotFound:
                raise RuntimeError(
                    f"{callsign} specification {RemexKeywords.HINTS}."
                    f" {task_remex_hints} could not be found in the configuration"
                )
        elif not isinstance(task_remex_hints, dict):
            raise RuntimeError(
                f"{callsign} specification {RemexKeywords.HINTS}" " is not a dictionary"
            )

        # Get the configurator task type default.
        # TODO: When assembling task remex hints, allow to select a type default from the predefined set.
        task_type_default_hints = {}
        try:
            if thing_type is not None:
                task_type_default_hints = self.bx_configurator.require(
                    f"bx_task_remex_hints.{thing_type}"
                )
        except NotFound:
            pass

        # Get the configurator overall default.
        overall_default_hints = {}
        try:
            overall_default_hints = self.bx_configurator.require(
                "bx_task_remex_hints.default"
            )
        except NotFound:
            pass

        remex_hints = copy.deepcopy(overall_default_hints)
        remex_hints.update(task_type_default_hints)
        remex_hints.update(task_remex_hints)

        # Check that the clusters are valid.
        remex_clusters = require(
            f"{callsign} assembled remex_hints",
            remex_hints,
            RemexKeywords.CLUSTER,
        )
        if not isinstance(remex_clusters, list):
            remex_clusters = [remex_clusters]
        for remex_cluster in remex_clusters:
            RemexClusters.validate(remex_cluster)

        # Lay the new values back into the task specification.
        bx_task_specification[RemexKeywords.HINTS] = copy.deepcopy(remex_hints)

    # ------------------------------------------------------------------
    def add_dummy_task(
        self,
        label,
        remex_hints=None,
        prepare_environment=None,
        delay=0,
        deliberate_error=None,
        infile_variable=None,
        outfile_variable=None,
        outfile=None,
    ):

        try:

            if prepare_environment is None:
                prepare_environment = []

            bx_task_specification = {
                "type": "dls_bxflow_run.bx_tasks.dummy",
                "label": label,
                RemexKeywords.HINTS: remex_hints,
                "type_specific_tbd": {
                    "delay": delay,
                    "deliberate_error": deliberate_error,
                    "infile_variable": infile_variable,
                    "outfile_variable": outfile_variable,
                    "outfile": outfile,
                },
                BxTaskKeywords.PREPARE_ENVIRONMENT: prepare_environment,
            }

            # Assemble remex_hints from task and configuration.
            self.assemble_remex_hints(bx_task_specification)

            # Build the task.
            bx_task = BxTasks().build_object(
                bx_task_specification,
            )

            # Add task to the job object we own and set up default dependency on previous task.
            self.add_task(bx_task)

        except Exception:
            raise RuntimeError(f"error adding dummy task for {label}")

        return bx_task

    # ------------------------------------------------------------------
    def add_task(
        self,
        bx_task,
    ):
        # Caller is providing a specification, not an object?
        if isinstance(bx_task, dict):
            bx_task_specification = bx_task

            # Clone the specification dict because we may update its values.
            bx_task_specification = copy.deepcopy(bx_task_specification)

            # Assemble remex_hints into the specification dict.
            # These will be overlay the specified hints on top of
            # what the bx_configurator is saying.
            self.assemble_remex_hints(bx_task_specification)

            # Build the task object from the specification.
            bx_task = BxTasks().build_object(bx_task_specification)

        # Extend the prepare_environment key to the tasks's specification.
        self.specify_prepare_environment(bx_task)

        # Add task to the job object we own.
        self.bx_job.bx_tasks.add(bx_task)
        bx_filestores_get_default().pin_task_directory(self.bx_job, bx_task)

        # This is not the first task?
        if len(self.bx_job.bx_tasks.list()) > 1:
            last_bx_task = self.bx_job.bx_tasks.list()[-2]

            # Automatically depend this task on the one before.
            bx_task.dependency_bx_gates.add(last_bx_task.success_bx_gate)

        # Return the bx_task object.
        return bx_task

    # ------------------------------------------------------------------
    def specify_prepare_environment(
        self,
        bx_task,
    ):
        """
        Extend the prepare_environment key to the tasks's specification.
        """

        # Get the configured environment for this task type.
        configured_environments = bx_configurators_get_default().require(
            "bx_task_environments"
        )
        configured_environment = configured_environments.get(bx_task.thing_type(), None)

        if configured_environment is None:
            configured_environment = configured_environments.get("default", [])

        # Get the already-specified environment for this task.
        specification = bx_task.specification()
        specified_environment = specification.get(
            BxTaskKeywords.PREPARE_ENVIRONMENT, []
        )

        # Extend the task's environment with the configured one.
        specified_environment.extend(configured_environment)
        bx_task.specification()[
            BxTaskKeywords.PREPARE_ENVIRONMENT
        ] = specified_environment

    # ------------------------------------------------------------------------------------
    def build(self):
        """
        Build a workflow by chaining tasks together.
        """

        raise RuntimeError(f"{callsign(self)} needs the build method to be overridden")

    # ------------------------------------------------------------------
    async def start(self):

        # Tell the bx_job that any bx_task failure block its further execution.
        for bx_task in self.bx_job.bx_tasks.list():
            self.bx_job.blocked_by_bx_gates.add(
                [
                    bx_task.failure_bx_gate,
                ]
            )

        # Tell the bx_job that the last bx_task success will also block its further execution.
        final_bx_task = self.bx_job.bx_tasks.list()[-1]
        self.bx_job.blocked_by_bx_gates.add(
            [
                final_bx_task.success_bx_gate,
            ]
        )

        # Make a workflow database entry for this workflow.
        await bx_datafaces_get_default().set_bx_workflows(
            [
                {
                    "uuid": self.uuid(),
                    "bx_job_uuid": self.bx_job.uuid(),
                    "filename_classname": self.__filename_classname,
                    "bx_settings_json": json.dumps(
                        self.__bx_settings.as_list_of_dicts(), indent=4
                    ),
                }
            ]
        )

        # Tell the job about the workflow database entry.
        self.bx_job.set_workflow_uuid(self.uuid())

        # Enable the job and let it run.
        await self.bx_job.enable()

    # ------------------------------------------------------------------
    async def wait(self, timeout=None, naptime=1.0):

        # Enable the job and let it run.
        await self.bx_job.wait(timeout=timeout, naptime=naptime)

    # ----------------------------------------------------------------------------------------
    def visualize(self):
        import dask
        from dask.delayed import Delayed

        bx_job = self.bx_job

        async def target_func():
            pass

        # ------------------------------------------------------------------
        # Turn the bx_gates into dask tasks.

        dsk = {}
        full_gate_labels = {}
        for bx_task in bx_job.bx_tasks.list():
            for bx_gate in bx_task.controlled_bx_gates.list():
                full_gate_label = "%s.%s" % (bx_task.label(), bx_gate.label())
                full_gate_labels[bx_gate.uuid()] = full_gate_label
                dask_task = (target_func, bx_task.label())
                dsk[full_gate_label] = dask_task

        for bx_task in bx_job.bx_tasks.list():
            dask_task = (target_func,)

            for bx_gate in bx_task.dependency_bx_gates.list():
                full_gate_label = full_gate_labels[bx_gate.uuid()]
                dask_task = dask_task + (full_gate_label,)

            dsk[bx_task.label()] = dask_task

        # -----------------------------------------------------------------
        blocked_by_bx_gate_uuids = []
        for bx_gate in bx_job.blocked_by_bx_gates.list():
            blocked_by_bx_gate_uuids.append(bx_gate.uuid())

        # for plotting dags using matplotlib:
        # https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/

        filebase = callsign(self).split(".")[-1]

        final_label = bx_job.bx_tasks.list()[-1].label()
        dsk_delayed = Delayed(final_label, dsk)
        png_filename = f"{filebase}.png"
        dask.visualize(
            dsk_delayed,
            filename=png_filename,
            engine="graphviz",
            rankdir="LR",
            optimize_graph=False,
            verbose=True,
        )
        html_filename = f"{filebase}.html"
        dask.visualize(
            dsk_delayed,
            filename=html_filename,
            engine="cytoscape",
            rankdir="LR",
            optimize_graph=False,
            verbose=True,
        )

        return [png_filename, html_filename]

    # ----------------------------------------------------------------------------------------
    def get_settings(self):
        return self.__bx_settings

    # ----------------------------------------------------------------------------------------
    def add_setting(self, specification):
        return self.__bx_settings.add(self.__bx_settings.build_object(specification))

    # ----------------------------------------------------------------------------------------
    def settings_load_from_json(self, json_string):
        """
        Build the interal settings list from a json string.
        Typically this json_string comes out of a string database field.
        """

        if json_string is None or json_string == "":
            return

        try:
            settings_dicts = json.loads(json_string)

            self.__bx_settings.load_from_dicts(settings_dicts)
        except Exception as exception:
            raise RuntimeError(explain2(exception, "loading settings from json string"))

    # ----------------------------------------------------------------------------------------
    def compose_html_inputs(self):
        return self.__bx_settings.compose_html_inputs()

    # ----------------------------------------------------------------------------------------
    def compose_settings_python_dict_assignment(self, assign_to):
        return self.__bx_settings.compose_python_dict_assignment(assign_to)

    # ----------------------------------------------------------------------------------------
    def compose_settings_python_variable_assignment(self):
        """
        Return a list of one or more lines that accomplishes a python assignment.
        """

        python_lines = []
        python_lines.append("# These values are supplied by the workflow settings.")
        python_lines.append("")

        python_lines.extend(self.__bx_settings.compose_python_variable_assignment())

        return "\n".join(python_lines)
