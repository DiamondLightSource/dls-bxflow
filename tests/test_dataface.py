import asyncio
import copy
import logging

# Object managing datafaces.
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
from dls_bxflow_run.bx_tasks.states import States as BxTaskStates

# Variables.
from dls_bxflow_run.bx_variables.bx_variables import BxVariables

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestDatafaceLaptop:
    def test_dataface_laptop(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        DatafaceTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class DatafaceTester(BaseContextTester):
    """
    Class to test the news.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()

        # Don't build a launcher or scheduler or catalog.
        bx_configurator.remove("bx_launcher_specifications")
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()
        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:

            # Start a new consumer (listener).
            await bx_context.add_news_consumer(self._consume_bx_news)

            # Manager of variables.
            variables = BxVariables()

            # Add a variable.
            variables.add("param1", "value1")

            await variables.register("12345678")

            # Query for the variable.
            records = await bx_datafaces_get_default().query(
                "SELECT * FROM bx_variables"
            )

            assert len(records) == 1

            # Create a bx_job using the default specification.
            bx_job_specification = copy.deepcopy(bx_jobs_get_default().specification())
            bx_job_specification["label"] = "job1"
            bx_job = BxJobs().build_object(bx_job_specification)

            # Set up a task.
            bx_task_label = "task1"
            bx_task = BxTasks().build_object(
                {
                    "type": "dls_bxflow_run.bx_tasks.dummy",
                    "label": bx_task_label,
                    RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
                }
            )

            # Add a task to the job.
            bx_job.bx_tasks.add([bx_task])

            # Schedule the bx_job to run.
            await bx_job.enable()

            # Since job won't run because of no scheduler, just wait for news to be sent.
            await asyncio.sleep(0.5)

            # Update the task to generate a news.
            await bx_datafaces_get_default().update_bx_task(
                {"uuid": bx_task.uuid(), "state": BxTaskStates.STARTED}
            )

            # Open a gate to generate a news.
            await bx_datafaces_get_default().open_bx_gate(bx_task.uuid(), "success")

            # --------------------------------------------------------
            # Fetch the job back from the database.
            bx_job = BxJobs().build_object(
                bx_job_specification, predefined_uuid=bx_job.uuid()
            )
            await bx_job.fetch()
            assert len(bx_job.bx_tasks.list()) == 1
            assert bx_job.bx_tasks.list()[0].label() == bx_task_label

            # Get the news from the database.
            records = await bx_datafaces_get_default().get_bx_news(bx_job.uuid())
            assert len(records) == 3

        # Make sure we got news notifications
        news_count = len(self.consumed_news)
        assert news_count == 3
