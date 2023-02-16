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
from dls_bxflow_run.bx_tasks.constants import Keywords as BxTaskKeywords

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestJobALaptop:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        JobATester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class Aclass:
    """
    This is a user-defined class which gets instantiated and called at runtime.
    """

    def __init__(self, outfile, algorithm=None, bx_task=None):
        self.outfile = outfile
        self.algorithm = algorithm

    async def run(self):
        logger.info("running Aclass")
        with open(self.outfile, "wt") as stream:
            stream.write(self.algorithm)


# Pythonpath where the Aclass can be found
aclass_pythonpath = "%s:%s" % (
    os.path.dirname(os.path.dirname(__file__)),
    os.environ.get("PYTHONPATH", ""),
)


# ----------------------------------------------------------------------------------------
class JobATester(BaseContextTester):
    """
    Class to test single-task job creation and running directly, without a workflow involved.
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

            # Define the datafile we will write and the algorithm.
            aclass_datafile = "aclass_datafile.txt"
            aclass_algorithm = "my_good_aclass_algorithm"

            # Define the class and its constructor arguments.
            aclass_bx_task = BxTasks().build_object(
                {
                    "type": "dls_bxflow_run.bx_tasks.pickled_class",
                    "label": "aclass",
                    BxTaskKeywords.PREPARE_ENVIRONMENT: [
                        f"export PYTHONPATH={aclass_pythonpath}"
                    ],
                    RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
                    "type_specific_tbd": {
                        "class": Aclass,
                        "constructor_args": [aclass_datafile],
                        "constructor_kwargs": {"algorithm": aclass_algorithm},
                    },
                }
            )

            # Create a bx_job using the default specification.
            bx_job_specification = copy.deepcopy(bx_jobs_get_default().specification())
            bx_job_specification["label"] = "job a"
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

            # Verify the output file contents from the task.
            self._assert_execution_output(
                "aclass_datafile.txt",
                self.tasks_execution_outputs[aclass_bx_task.uuid()],
                expected_content=aclass_algorithm,
            )

            # Delete the job and all related records and directories.
            await bx_datafaces_get_default().delete_bx_job(bx_job.uuid())

            assert not os.path.exists(bx_job.get_directory())
            for bx_task in bx_job.bx_tasks.list():
                assert not os.path.exists(bx_task.get_directory())
            records = await bx_datafaces_get_default().get_bx_jobs()
            assert len(records) == 0
            records = await bx_datafaces_get_default().get_bx_tasks(bx_job.uuid())
            assert len(records) == 0
            records = await bx_datafaces_get_default().get_controlled_bx_gates(
                bx_job.uuid()
            )
            assert len(records) == 0
            records = await bx_datafaces_get_default().get_dependency_bx_gates(
                bx_job.uuid()
            )
            assert len(records) == 0
            records = await bx_datafaces_get_default().get_blocked_by_bx_gates(
                bx_job.uuid()
            )
            assert len(records) == 0
            records = await bx_datafaces_get_default().get_bx_variables(bx_job.uuid())
            assert len(records) == 0
            records = await bx_datafaces_get_default().get_bx_news(
                bx_job_uuid=bx_job.uuid()
            )
            assert len(records) == 0

        # -------------------------------------------------------------
        # Context is now closed, database and all other services are unavailable.

        logger.info(f"job summary\n{job_summary_text}")

        # logger.info(describe("self.consumed_news", self.consumed_news))

        # Check we got all the news.
        count = len(self.consumed_news)
        expected_news_count = 9
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
        assert topic == BxNewsTopics.BXJOB_WAS_DELETED
        assert payload["bx_job"]["uuid"] == bx_job.uuid()

        topic, headline, payload = self.consumed_news[7]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED
        assert payload["bx_launcher"]["state"] == BxLauncherStates.SHUTDOWN

        topic, headline, payload = self.consumed_news[8]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED
        assert payload["bx_launcher"]["state"] == BxLauncherStates.SHUTDOWN
