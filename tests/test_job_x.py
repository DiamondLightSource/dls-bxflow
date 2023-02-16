import copy
import logging
import os

# Utilities.
from dls_utilpack.describe import describe

from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Remote execution.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing bx_jobs.
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs, bx_jobs_get_default

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks
from dls_bxflow_run.bx_tasks.constants import Keywords as BxTaskKeywords

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

# Task test classes.
from .task_classes.task_z import TaskZ

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestJobX:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        JobXTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class Xclass:
    """
    This is a user-defined class which gets instantiated and called at runtime.
    """

    # TODO: Remove requirement that all pickled_class tasks need bx_task kwarg.
    def __init__(self, bx_task=None):
        pass

    async def run(self):
        logger.info("running Xclass")

        import nonexistent.thing  # noqa


# Pythonpath where the Xclass can be found by main_isolated.
xclass_pythonpath = "%s:%s" % (
    os.path.dirname(os.path.dirname(__file__)),
    os.environ.get("PYTHONPATH", ""),
)


# ----------------------------------------------------------------------------------------
class JobXTester(BaseContextTester):
    """
    Class to test a series of failing tasks.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

        ipynb_filename = f"{os.getcwd()}/tests/jupyter/jupyter_bad_cell.ipynb"
        bx_task_specifications = [
            # Task which has a cell execution error.
            {
                "type": "dls_bxflow_run.bx_tasks.jupyter",
                "label": "jupyter_bad_cell.ipynb",
                "type_specific_tbd": {"ipynb_filename": ipynb_filename},
            },
            # # Task which has a module import error.
            {
                "type": "dls_bxflow_run.bx_tasks.pickled_class",
                "label": "bad_import",
                BxTaskKeywords.PREPARE_ENVIRONMENT: [
                    f"export PYTHONPATH={xclass_pythonpath}"
                ],
                "type_specific_tbd": {
                    "class": Xclass,
                },
            },
            # Task which has a bad pickle error.
            {
                "type": "dls_bxflow_run.bx_tasks.pickled_class",
                "label": "bad_pickle",
                "type_specific_tbd": {
                    "class": "not a pickle",
                },
            },
            # Task which has a class instantiation error.
            # We can pickle TaskZ from here, but it won't be in the PYTHONPATH when in main_isolated.
            {
                "type": "dls_bxflow_run.bx_tasks.pickled_class",
                "label": "task_z",
                "type_specific_tbd": {
                    "class": TaskZ,
                },
            },
        ]

        task_expected_error_lines = [
            [None, "CellExecutionError"],
            [None, None, "ModuleNotFoundError: No module named 'nonexistent'"],
            [
                None,
                'RuntimeError: did not get a class decoding pickle "not a pickle"',
            ],
            [
                None,
                'RuntimeError: did not get a class decoding pickle {"py/type": "tests.task_classes.task_z.TaskZ"}',
            ],
        ]

        # Load the configuration.
        bx_configurator = self.get_bx_configurator()

        # Don't start some services.
        bx_configurator.remove("dls_servbase_dataface_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()

        # Make a context from the configuration.
        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:
            # Define the class to run.

            # Create a bx_job using the default specification.
            bx_job_specification = copy.deepcopy(bx_jobs_get_default().specification())
            bx_job_specification["label"] = "workflow x job"
            bx_job = BxJobs().build_object(bx_job_specification)

            # Loop through the defined tasks.
            last_failure_gate = None
            for bx_task_specification in bx_task_specifications:
                bx_task_specification[RemexKeywords.HINTS] = {
                    RemexKeywords.CLUSTER: RemexClusters.LOCAL
                }

                # Build a task object.
                bx_task = BxTasks().build_object(bx_task_specification)

                # Add a task to the job.
                bx_job.bx_tasks.add([bx_task])

                # Chain the tasks together by their failure gates:
                if last_failure_gate is not None:
                    bx_task.dependency_bx_gates.add(last_failure_gate)

                last_failure_gate = bx_task.failure_bx_gate

            logger.info(describe("task z", bx_job.bx_tasks.list()[-1].specification()))

            # Tell the bx_job what will block its further execution.
            bx_job.blocked_by_bx_gates.add([last_failure_gate])

            # Schedule the bx_job to run.
            await bx_job.enable()

            # Wait for bx_job to finish.
            await bx_job.wait(timeout=10.0)

            # Get the list of output files produced.
            self.capture_tasks_execution_outputs(bx_job)

            # Summarize the job's execution.
            job_summary_text = await self._compose_job_summary(bx_job.uuid())

            # Get the tasks' database records.
            bx_task_records = await bx_datafaces_get_default().get_bx_tasks(
                bx_job.uuid()
            )

        # -------------------------------------------------------------
        # Context is now closed, database and all other services are unavailable.

        logger.info(f"job summary\n{job_summary_text}")

        # Make sure all the residuals are there, with non-empty stderr.
        for bx_task in bx_job.bx_tasks.list():
            execution_outputs = self.tasks_execution_outputs[bx_task.uuid()]
            self.assert_execution_residuals(
                execution_outputs, expect_stderr_empty=False
            )

        # Make sure the expected error lines are present.
        for index, bx_task_record in enumerate(bx_task_records):
            bx_task_specification = bx_task_specifications[index]
            bx_task_label = bx_task_specification["label"]

            expected_error_lines = task_expected_error_lines[index]

            recorded_error_lines = bx_task_record["error_lines"].split("\n")
            assert len(recorded_error_lines) == len(expected_error_lines)

            # logger.info(
            #     describe(f"{bx_task_label} recorded_error_lines", recorded_error_lines)
            # )

            for j, line in enumerate(recorded_error_lines):
                if expected_error_lines[j] is not None:
                    assert (
                        expected_error_lines[j] in recorded_error_lines[j]
                    ), f"task {bx_task_label} error line {j}"
