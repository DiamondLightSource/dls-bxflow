import copy
import logging
import os

# Utilities.
from dls_utilpack.describe import describe

from dls_bxflow_api.bx_databases.constants import BxJobFieldnames
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Remote execution.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managers we interact with.
from dls_bxflow_lib.bx_filestores.bx_filestores import bx_filestores_get_default

# Object managing bx_jobs.
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs, bx_jobs_get_default

# Object managing bx_launchers.
from dls_bxflow_lib.bx_launchers.states import States as BxLauncherStates

# News events.
from dls_bxflow_lib.bx_news.constants import Topics as BxNewsTopics

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

# Task types.
from dls_bxflow_run.bx_tasks.constants import Types as BxTaskTypes
from dls_bxflow_run.bx_tasks.execution_summary import ExecutionSummary

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)

# This is what the notebook will set for ExecutionSummary.
RANDOM_STRING_PNG = "random_string.png"


# ----------------------------------------------------------------------------------------
class TestJobNotebook:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        JobNotebookTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class JobNotebookTester(BaseContextTester):
    """
    Class to test single-task jupyternotebook job creation and running directly, without a workflow involved.

    Of particular interest is this notebook creates execution summary for th ejob using the imported ExecutionSummary class.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

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
            # Start a news consumer.
            await bx_context.add_news_consumer(self._consume_bx_news)

            ipynb_filename = f"{os.getcwd()}/tests/jupyter/jupyter2.ipynb"

            # Define the class and its constructor arguments.
            aclass_bx_task = BxTasks().build_object(
                {
                    "type": BxTaskTypes.JUPYTER,
                    "label": "task-jupyter2",
                    RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
                    "type_specific_tbd": {
                        "ipynb_filename": ipynb_filename,
                    },
                }
            )

            # Create a bx_job using the default specification.
            bx_job_specification = copy.deepcopy(bx_jobs_get_default().specification())
            bx_job_specification["label"] = "job-jupyter2"
            bx_job = BxJobs().build_object(bx_job_specification)

            # Add a task to the job.
            bx_job.bx_tasks.add([aclass_bx_task])

            # Tell the bx_job what will block its further execution.
            bx_job.blocked_by_bx_gates.add(
                [
                    aclass_bx_task.failure_bx_gate,
                    aclass_bx_task.success_bx_gate,
                ]
            )

            # Schedule the bx_job to run.
            await bx_job.enable()

            # Wait for bx_job to finish.
            await bx_job.wait(timeout=10.0)

            # Wait for all the news.
            # await asyncio.sleep(2.0)

            self.capture_tasks_execution_outputs(bx_job)
            tasks_execution_outputs = []
            for bx_task in bx_job.bx_tasks.list():
                # Capture the files which were output by each task.
                tasks_execution_outputs.append(
                    bx_filestores_get_default().get_runtime_execution_outputs(bx_task)
                )

            # Summarize the job's execution.
            job_summary_text = await self._compose_job_summary(bx_job.uuid())

            # Make sure all the residuals are there.
            self.assert_tasks_execution_residuals()

            # Verify the output file contents from the execution summary written by the task.
            self._assert_execution_output(
                ExecutionSummary().filename,
                self.tasks_execution_outputs[aclass_bx_task.uuid()],
                expected_content=RANDOM_STRING_PNG,
            )

            # Verify that the bx_job record has the execution summary from the task.
            record = await bx_datafaces_get_default().get_bx_job(bx_job.uuid())
            assert record[BxJobFieldnames.EXECUTION_SUMMARY] == RANDOM_STRING_PNG

        # -------------------------------------------------------------
        # Context is now closed, database and all other services are unavailable.

        logger.info(f"job summary\n{job_summary_text}")

        # logger.info(describe("self.consumed_news", self.consumed_news))

        # Check we got all the news.
        count = len(self.consumed_news)
        expected_news_count = 8
        if count != expected_news_count:
            logger.info(describe("self.consumed_news", self.consumed_news))

        assert count == expected_news_count

        topic, headline, payload = self.consumed_news[0]
        assert topic == BxNewsTopics.BXJOB_WAS_ENABLED
        assert payload["bx_job"]["uuid"] == bx_job.uuid()

        topic, headline, payload = self.consumed_news[1]
        assert topic == BxNewsTopics.BXTASK_WAS_STARTED
        assert payload["bx_task"]["uuid"] == aclass_bx_task.uuid()

        topic, headline, payload = self.consumed_news[2]
        assert topic == BxNewsTopics.BXTASK_WAS_FINISHED
        assert payload["bx_task"]["uuid"] == aclass_bx_task.uuid()

        topic, headline, payload = self.consumed_news[3]
        assert topic == BxNewsTopics.BXGATE_WAS_OPENED
        assert payload["bx_gate"]["bx_task_uuid"] == aclass_bx_task.uuid()

        topic, headline, payload = self.consumed_news[4]
        assert topic == BxNewsTopics.BXJOB_SUCCEEDED
        assert payload["bx_job"]["uuid"] == bx_job.uuid()

        topic, headline, payload = self.consumed_news[5]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED
        assert payload["bx_launcher"]["state"] == BxLauncherStates.IDLE

        topic, headline, payload = self.consumed_news[6]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED
        assert payload["bx_launcher"]["state"] == BxLauncherStates.SHUTDOWN

        topic, headline, payload = self.consumed_news[7]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED
        assert payload["bx_launcher"]["state"] == BxLauncherStates.SHUTDOWN
