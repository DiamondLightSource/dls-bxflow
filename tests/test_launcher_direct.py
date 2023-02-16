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

# Remex (remote execution) API.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing bx_jobs.
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs, bx_jobs_get_default

# Launcher.
from dls_bxflow_lib.bx_launchers.bx_launchers import BxLaunchers

# News events.
from dls_bxflow_lib.bx_news.constants import Topics as BxNewsTopics

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks
from dls_bxflow_run.bx_tasks.states import States as BxTaskStates

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)

bx_task_uuid = "00002-2222-3333-4444-5555555"


# ----------------------------------------------------------------------------------------
class TestLauncherDirectQsubber:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"

        # The launcher configuration to replace in the configuration file for this test.
        launcher_keyword = "bx_launcher_qsubber_specification"

        LauncherDirectTester(launcher_keyword).main(
            constants, configuration_file, output_directory
        )


# ----------------------------------------------------------------------------------------
class LauncherDirectTester(BaseContextTester):
    """
    Class to test the launcher being restarted.
    """

    def __init__(
        self,
        launcher_keyword,
    ):
        BaseContextTester.__init__(self)

        self.__launcher_keyword = launcher_keyword

    # ------------------------------------------------------------------------------------
    async def _main_coroutine(self, constants, output_directory):
        """ """

        # Make the fake qsub findable in the path.
        os.environ["PATH"] = "%s:%s" % (os.path.dirname(__file__), os.environ["PATH"])

        # Supply environment variable for substitution in the bx configurator.
        os.environ["OUTPUT_DIRECTORY"] = output_directory

        # Directory where the task should write its files.
        bx_task_directory = f"{output_directory}/task_directory"

        # Load the configuration.
        bx_configurator = self.get_bx_configurator()

        # Don't start the services we don't need for this test.
        bx_configurator.remove("bx_launcher_specifications")
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_gui_specification")
        bx_configurator.remove("dls_servbase_dataface_specification")

        context_configuration = await bx_configurator.load()

        # Make a context from the configuration.
        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:
            # Start a news consumer.
            await bx_context.add_news_consumer(self._consume_bx_news)

            # Get the desired launcher's direct specification.
            launcher_keyword = context_configuration[self.__launcher_keyword]

            # Build the actual launcher.
            actual_bx_launcher_specification = launcher_keyword["type_specific_tbd"][
                "actual_bx_launcher_specification"
            ]

            actual_bx_launcher_specification["uuid"] = "direct-01"
            launcher = BxLaunchers().build_object(actual_bx_launcher_specification)

            # Define the datafile we will write.
            dummy_label = "dummy-1"
            dummy_outfile = f"{output_directory}/dummy_datafile.json"
            dummy_task_delay = 4.0
            bx_task_specification = {
                "uuid": bx_task_uuid,
                "type": "dls_bxflow_run.bx_tasks.dummy",
                "label": dummy_label,
                "directory": bx_task_directory,
                "remex_hints": {
                    RemexKeywords.CLUSTER: RemexClusters.HAMILTON,
                },
                "type_specific_tbd": {
                    "outfile": dummy_outfile,
                    "delay": dummy_task_delay,
                },
            }

            # Define the class and its constructor arguments.
            bx_task = BxTasks().build_object(
                bx_task_specification, predefined_uuid=bx_task_uuid
            )

            # Create a bx_job using the default specification.
            bx_job_specification = copy.deepcopy(bx_jobs_get_default().specification())
            bx_job_specification["label"] = "test_job"
            bx_job = BxJobs().build_object(bx_job_specification)

            # Add a task to the job.
            bx_job.bx_tasks.add([bx_task])

            # Tell the bx_job what will block its further execution.
            bx_job.blocked_by_bx_gates.add(
                [
                    bx_task.failure_bx_gate,
                    bx_task.success_bx_gate,
                ]
            )

            await bx_datafaces_get_default().set_bx_tasks([bx_task_specification])

            # Submit the bx_task.
            await launcher.submit(
                bx_job.uuid(),
                bx_job_specification,
                bx_task.uuid(),
                bx_task_specification,
            )

            time0 = time.time()
            timeout = 5.0
            while True:
                # Let the server harvest the running task.
                await launcher.harvest()

                # Check the task status.
                task_record = await bx_datafaces_get_default().get_bx_task(
                    bx_task.uuid()
                )

                assert (
                    task_record is not None
                ), f"no task record for uuid {bx_task.uuid()}"

                # Task finished?
                if task_record["state"] == BxTaskStates.FINISHED:
                    break
                if time.time() - time0 > timeout:
                    raise RuntimeError(f"bx_task not finished within {timeout} seconds")
                await asyncio.sleep(1.0)

            # Capture the files which were output by each task.
            self.capture_tasks_execution_outputs(bx_job)

        # -------------------------------------------------------------
        # Context is now closed, database and all other services are unavailable.

        # Make sure all the residuals are there.
        self.assert_tasks_execution_residuals()

        # Verify all the dummy task's output file contents.
        with open(dummy_outfile, "r") as stream:
            dummy_output = json.load(stream)
            assert dummy_output["bx_task_label"] == dummy_label

        # Check we got all the news.
        count = len(self.consumed_news)
        if count != 2:
            logger.info(describe("self.consumed_news", self.consumed_news))
        assert count == 2
        topic, headline, payload = self.consumed_news[0]
        assert topic == BxNewsTopics.BXTASK_WAS_STARTED
        assert payload["bx_task"]["uuid"] == bx_task.uuid()

        topic, headline, payload = self.consumed_news[1]
        assert topic == BxNewsTopics.BXTASK_WAS_FINISHED
        assert payload["bx_task"]["uuid"] == bx_task.uuid()
