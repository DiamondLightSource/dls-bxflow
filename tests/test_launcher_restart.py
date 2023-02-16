import asyncio
import copy
import json
import logging
import os
import time

# Utilities.
from dls_utilpack.describe import describe

# Object managing datafaces.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Remote execution.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing bx_jobs.
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs, bx_jobs_get_default
from dls_bxflow_lib.bx_launchers.bx_launchers import BxLaunchers

# Object managing bx_launchers.
from dls_bxflow_lib.bx_launchers.states import States as BxLauncherStates

# News events.
from dls_bxflow_lib.bx_news.constants import Topics as BxNewsTopics

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks
from dls_bxflow_run.bx_tasks.states import States as BxTaskStates

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestLauncherRestartPopener:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"

        # The launcher configuration to replace in the configuration file for this test.
        desired_bx_launcher_specification = "bx_launcher_popener_specification"

        LauncherRestartTester(desired_bx_launcher_specification).main(
            constants, configuration_file, output_directory
        )


# ----------------------------------------------------------------------------------------
class TestLauncherRestartQsubber:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"

        # The launcher configuration to replace in the configuration file for this test.
        actual_bx_launcher_specification = "bx_launcher_qsubber_specification"

        LauncherRestartTester(actual_bx_launcher_specification).main(
            constants, configuration_file, output_directory
        )


# ----------------------------------------------------------------------------------------
class LauncherRestartTester(BaseContextTester):
    """
    Class to test the launcher being restarted.
    """

    def __init__(
        self,
        desired_bx_launcher_specification,
    ):
        BaseContextTester.__init__(self)

        self.__desired_bx_launcher_specification = desired_bx_launcher_specification

    # ------------------------------------------------------------------------------------
    async def _main_coroutine(self, constants, output_directory):
        """ """

        # Make the fake qsub findable in the path.
        os.environ["PATH"] = "%s:%s" % (os.path.dirname(__file__), os.environ["PATH"])

        # Supply environment variable for substitution in the bx configurator.
        os.environ["OUTPUT_DIRECTORY"] = output_directory

        # Load the configuration.
        bx_configurator = self.get_bx_configurator()

        # Don't start the services we don't need for this test.
        bx_configurator.remove("dls_servbase_dataface_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()

        # Keep only the one desired bx_launcher_specification.
        bx_launcher_specifications = context_configuration["bx_launcher_specifications"]
        bx_launcher_specifications.clear()
        bx_launcher_specifications.append(self.__desired_bx_launcher_specification)

        # Make a context from the configuration.
        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:
            # Build a client to our desired launcher.
            launcher = BxLaunchers().build_object(
                bx_configurator.require(self.__desired_bx_launcher_specification)
            )

            try:
                # Start a news consumer.
                await bx_context.add_news_consumer(self._consume_bx_news)

                # Define the datafile we will write.
                dummy_label = "dummy-1"
                dummy_outfile = f"{output_directory}/dummy_datafile.json"
                # Let the task delay long enough to survive the launcher restart.
                dummy_task_delay = 8.0
                bx_task_specification = {
                    "type": "dls_bxflow_run.bx_tasks.dummy",
                    "label": dummy_label,
                    "directory": f"{output_directory}/{self.__desired_bx_launcher_specification}",
                    RemexKeywords.HINTS: {
                        RemexKeywords.CLUSTER: [RemexClusters.LOCAL, RemexClusters.TEST]
                    },
                    "type_specific_tbd": {
                        "outfile": dummy_outfile,
                        "delay": dummy_task_delay,
                    },
                }

                # Define the class and its constructor arguments.
                bx_task = BxTasks().build_object(bx_task_specification)

                # Create a bx_job using the default specification.
                bx_job_specification = copy.deepcopy(
                    bx_jobs_get_default().specification()
                )
                bx_job_specification["label"] = "test_job"
                bx_job = BxJobs().build_object(bx_job_specification)
                bx_job.set_data_label("<nodata>")

                # Add a task to the job.
                bx_job.bx_tasks.add([bx_task])

                # Tell the bx_job what will block its further execution.
                bx_job.blocked_by_bx_gates.add(
                    [
                        bx_task.failure_bx_gate,
                        bx_task.success_bx_gate,
                    ]
                )

                # Schedule the bx_job to run.
                await bx_job.enable()
                logger.debug("[ADDORPH] job enabled")

                time0 = time.time()
                timeout = 5.0
                while True:
                    # Let the server harvest the running task.
                    # await launcher.harvest()

                    # Check the task status.
                    task_record = await bx_datafaces_get_default().get_bx_task(
                        bx_task.uuid()
                    )

                    # Task finished?
                    if task_record["state"] == BxTaskStates.STARTED:
                        break
                    if time.time() - time0 > timeout:
                        raise RuntimeError(
                            f"bx_task not started within {timeout} seconds"
                        )

                    await asyncio.sleep(1.0)

                logger.debug("[ADDORPH] task started")

            finally:
                await launcher.close_client_session()

        logger.debug(
            "[ADDORPH] first context has now exited ===============================================================\n\n\n"
        )

        # Restart all the services.
        async with bx_context:
            # Start a news consumer from the restarted services.
            await bx_context.add_news_consumer(self._consume_bx_news)

            logger.debug(
                "[ADDORPH] second context is now started ===============================================================\n\n\n"
            )

            # Wait for bx_job to finish.
            await bx_job.wait(timeout=10.0)

            # Capture the files which were output by each task.
            self.capture_tasks_execution_outputs(bx_job)

            # Summarize the job's execution.
            job_summary_text = await self._compose_job_summary(bx_job.uuid())

        logger.debug(
            "[ADDORPH] second context has now exited ===============================================================\n\n\n"
        )

        # -------------------------------------------------------------
        # Context is now closed, database and all other services are unavailable.

        logger.info(f"job summary\n{job_summary_text}")

        # Make sure all the residuals are there.
        self.assert_tasks_execution_residuals()

        # Verify all the dummy task's output file contents.
        with open(dummy_outfile, "r") as stream:
            dummy_output = json.load(stream)
            assert dummy_output["bx_task_label"] == dummy_label

        # Check we got all the news.
        count = len(self.consumed_news)
        if count != 8:
            logger.info(describe("self.consumed_news", self.consumed_news))
        assert count == 8
        topic, headline, payload = self.consumed_news[0]
        assert topic == BxNewsTopics.BXJOB_WAS_ENABLED
        assert payload["bx_job"]["uuid"] == bx_job.uuid()

        topic, headline, payload = self.consumed_news[1]
        assert topic == BxNewsTopics.BXTASK_WAS_STARTED
        assert payload["bx_task"]["uuid"] == bx_task.uuid()

        topic, headline, payload = self.consumed_news[2]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED
        assert payload["bx_launcher"]["state"] == BxLauncherStates.SHUTDOWN

        topic, headline, payload = self.consumed_news[3]
        assert topic == BxNewsTopics.BXTASK_WAS_FINISHED
        assert payload["bx_task"]["uuid"] == bx_task.uuid()

        topic, headline, payload = self.consumed_news[4]
        assert topic == BxNewsTopics.BXGATE_WAS_OPENED
        assert payload["bx_gate"]["bx_task_uuid"] == bx_task.uuid()

        topic, headline, payload = self.consumed_news[5]
        assert topic == BxNewsTopics.BXJOB_SUCCEEDED
        assert payload["bx_job"]["uuid"] == bx_job.uuid()

        topic, headline, payload = self.consumed_news[6]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED
        assert payload["bx_launcher"]["state"] == BxLauncherStates.IDLE

        topic, headline, payload = self.consumed_news[7]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED
        assert payload["bx_launcher"]["state"] == BxLauncherStates.SHUTDOWN
