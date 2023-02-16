# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import DuplicateLabelException, NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

# Task types.
from dls_bxflow_run.bx_tasks.constants import Types as BxTaskTypes

logger = logging.getLogger(__name__)


class BxTasks(Things):
    """
    List of available bx_tasks.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name="bx_tasks"):
        Things.__init__(self, name)
        self.__labels = {}

    # ----------------------------------------------------------------------------------------
    def __str__(self):
        return "BxTasks [%s]" % (", ".join(list(self.__labels.keys())))

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification=None, predefined_uuid=None):
        """"""

        # If a string, parse for json, yaml or whatever.
        specification = self.parse_specification(specification)

        bx_task_class = self.lookup_class(specification["type"])

        try:
            bx_task_object = bx_task_class(
                specification, predefined_uuid=predefined_uuid
            )
        except Exception:
            raise RuntimeError(
                f"building bx_task object of class {bx_task_class.__name__}"
            )

        return bx_task_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == BxTaskTypes.SYMLINK:
            from dls_bxflow_run.bx_tasks.symlink import Symlink

            return Symlink

        if class_type == "dls_bxflow_run.bx_tasks.dawn1":
            from dls_bxflow_run.bx_tasks.dawn1 import Dawn1

            return Dawn1

        if class_type == "dls_bxflow_run.bx_tasks.dawn2":
            from dls_bxflow_run.bx_tasks.dawn2 import Dawn2

            return Dawn2

        if class_type == "dls_bxflow_run.bx_tasks.dummy":
            from dls_bxflow_run.bx_tasks.dummy import Dummy

            return Dummy

        if class_type == BxTaskTypes.FILENAME_CLASSNAME:
            from dls_bxflow_run.bx_tasks.filename_classname import FilenameClassname

            return FilenameClassname

        if class_type == BxTaskTypes.MODULE_CLASSNAME:
            from dls_bxflow_run.bx_tasks.module_classname import ModuleClassname

            return ModuleClassname

        if class_type == BxTaskTypes.JUPYTER:
            from dls_bxflow_run.bx_tasks.jupyter import Jupyter

            return Jupyter

        if class_type == "dls_bxflow_run.bx_tasks.pickled_class":
            from dls_bxflow_run.bx_tasks.pickled_class import PickledClass

            return PickledClass

        if class_type == BxTaskTypes.PTYPY_MPI:
            from dls_bxflow_run.bx_tasks.ptypy_mpi import PtypyMpi

            return PtypyMpi

        if class_type == BxTaskTypes.PTYREX_MPI:
            from dls_bxflow_run.bx_tasks.ptyrex_mpi import PtyrexMpi

            return PtyrexMpi

        if class_type == "dls_bxflow_run.bx_tasks.shell":
            from dls_bxflow_run.bx_tasks.shell import Shell

            return Shell

        raise NotFound("unable to get bx_task class for %s" % (class_type))

    # -----------------------------------------------------------------------------
    def add(self, things):
        """
        Override this method of Things.  First check for duplicate labels.
        """

        if not hasattr(things, "__iter__"):
            things = [things]
        for thing in things:
            if thing.label() in self.__labels:
                raise DuplicateLabelException(
                    f"unwilling to add a second task with label {thing.label()}"
                )

        # Let the base class add the things normally.
        Things.add(self, things)

        for thing in things:
            self.__labels[thing.label()] = thing

    # ------------------------------------------------------------------
    def match_by_label(self, bx_task_label_pattern):
        """
        Return all the tasks in the list that match the label.
        """
        matching_bx_tasks = BxTasks()
        for bx_task in self.list():
            should_add = bx_task_label_pattern == "*"
            if bx_task.label().startswith(bx_task_label_pattern):
                should_add = True
            if should_add:
                matching_bx_tasks.add(bx_task)
        return matching_bx_tasks

    # ----------------------------------------------------------------------------------------
    def depend_on(self, upstream_bx_tasks):
        """
        All the tasks in our list depend on all the tasks in the given upstream list.
        """

        for self_bx_task in self.list():
            self_bx_task.dependency_bx_gates.clear()

            if isinstance(upstream_bx_tasks, str):
                label = upstream_bx_tasks
                upstream_bx_tasks = self.match_by_label(label)

            if upstream_bx_tasks is not None:
                for upstream_bx_task in upstream_bx_tasks.list():
                    self_bx_task.dependency_bx_gates.add(
                        upstream_bx_task.success_bx_gate
                    )
