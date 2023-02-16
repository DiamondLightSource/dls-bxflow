import asyncio
import copy
import logging
import os

import pytest

# Pairstream data package.
from dls_pairstream_lib.pairstream import Data as PairstreamData

# Utilities.
from dls_utilpack.describe import describe
from dls_utilpack.isodatetime import isodatetime

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

publisher_datafile_filename = "publisher_datafile.txt"
consumer_datafile_filename = "consumer_datafile.txt"


# ----------------------------------------------------------------------------------------
@pytest.mark.skipif("not config.getoption('isolated_dataface')")
class TestJobPubcon:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        JobPubconTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class PublisherClass:
    """
    This is a user-defined class which gets instantiated and called at runtime.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, bx_task=None, the_published_data=None):
        self.__bx_task = bx_task
        self.__the_published_data = the_published_data

    # ----------------------------------------------------------------------------------------
    async def run(self):
        # Bind the publisher port to random port and set the bx_variable.
        await self.__bx_task.prepare_pairstream_publisher()

        # Write our output data file.
        with open(publisher_datafile_filename, "w") as stream:
            stream.write(self.__the_published_data)

        # Publish to pairstream.
        meta = {"the_published_data": self.__the_published_data}
        data = PairstreamData(bytearray(0))
        self.__bx_task.publish_pairstream(meta, data)

        return 0


# ----------------------------------------------------------------------------------------
class ConsumerClass:
    """
    This is a user-defined class which gets instantiated and called at runtime.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, bx_task=None, publisher_label=None):
        self.__bx_task = bx_task
        self.__publisher_label = publisher_label

    # ----------------------------------------------------------------------------------------
    async def run(self):

        # Connect to the publisher.
        await self.__bx_task.prepare_pairstream_consumer(self.__publisher_label)

        # Consume from the publisher.
        meta = {}
        data = PairstreamData()
        self.__bx_task.consume_pairstream(meta, data)

        # Write the data we got to our output file.
        with open(consumer_datafile_filename, "w") as stream:
            stream.write(meta["the_published_data"])

        return 0


# Pythonpath where the PublisherClass and ConsumerClass can be found
pickled_class_pythonpath = "%s:%s" % (
    os.path.dirname(os.path.dirname(__file__)),
    os.environ.get("PYTHONPATH", ""),
)


# ----------------------------------------------------------------------------------------
class JobPubconTester(BaseContextTester):
    """
    Class to test the "a" workflow.
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

            # Data passed from publisher to consumer.
            the_published_data = isodatetime()

            # Define the class and its constructor arguments.
            publisher_bx_task = BxTasks().build_object(
                {
                    "type": "dls_bxflow_run.bx_tasks.pickled_class",
                    "label": "aclass",
                    BxTaskKeywords.PREPARE_ENVIRONMENT: [
                        f"export PYTHONPATH={pickled_class_pythonpath}"
                    ],
                    RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
                    "type_specific_tbd": {
                        "class": PublisherClass,
                        "constructor_kwargs": {
                            "the_published_data": the_published_data
                        },
                    },
                }
            )

            # Define the class and its constructor arguments.
            consumer_bx_task = BxTasks().build_object(
                {
                    "type": "dls_bxflow_run.bx_tasks.pickled_class",
                    "label": "bclass",
                    BxTaskKeywords.PREPARE_ENVIRONMENT: [
                        f"export PYTHONPATH={pickled_class_pythonpath}"
                    ],
                    RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
                    "type_specific_tbd": {
                        "class": ConsumerClass,
                        "constructor_kwargs": {
                            "publisher_label": publisher_bx_task.label()
                        },
                    },
                }
            )

            # Create a bx_job using the default specification.
            bx_job_specification = copy.deepcopy(bx_jobs_get_default().specification())
            bx_job_specification["label"] = "job a"
            bx_job = BxJobs().build_object(bx_job_specification)

            # Add both tasks to the job.
            bx_job.bx_tasks.add([publisher_bx_task])
            bx_job.bx_tasks.add([consumer_bx_task])

            # Tell the bx_job what will block its further execution.
            bx_job.blocked_by_bx_gates.add(
                [
                    publisher_bx_task.failure_bx_gate,
                    publisher_bx_task.success_bx_gate,
                    consumer_bx_task.failure_bx_gate,
                    consumer_bx_task.success_bx_gate,
                ]
            )

            # Schedule the bx_job to run.
            await bx_job.enable()

            # Wait for all the news.
            # await self.wait_for_job_blocked(bx_job)

            # Wait for bx_job to finish.
            await bx_job.wait(timeout=10.0)

            # Wait for all the news.
            await asyncio.sleep(1.0)

            self.capture_tasks_execution_outputs(bx_job)
            tasks_execution_outputs = []
            for bx_task in bx_job.bx_tasks.list():
                # Capture the files which were output by each task.
                tasks_execution_outputs.append(
                    bx_filestores_get_default().get_runtime_execution_outputs(bx_task)
                )

            # Summarize the job's execution.
            job_summary_text = await self._compose_job_summary(bx_job.uuid())

        # -------------------------------------------------------------
        # Context is now closed, database and all other services are unavailable.

        logger.info(f"job summary\n{job_summary_text}")

        # Make sure all the residuals are there.
        self.assert_tasks_execution_residuals()

        # Verify the output file contents from the task.
        self._assert_execution_output(
            publisher_datafile_filename,
            self.tasks_execution_outputs[publisher_bx_task.uuid()],
            expected_content=the_published_data,
        )

        # Verify the output file contents from the task.
        self._assert_execution_output(
            consumer_datafile_filename,
            self.tasks_execution_outputs[consumer_bx_task.uuid()],
            expected_content=the_published_data,
        )

        return

        # logger.info(describe("self.consumed_news", self.consumed_news))

        # Check we got all the news.
        count = len(self.consumed_news)
        if count != 6:
            logger.info(describe("self.consumed_news", self.consumed_news))
        assert count == 6
        topic, headline, payload = self.consumed_news[0]
        assert topic == BxNewsTopics.BXJOB_WAS_ENABLED
        assert payload["bx_job"]["uuid"] == bx_job.uuid()

        topic, headline, payload = self.consumed_news[1]
        assert topic == BxNewsTopics.BXTASK_WAS_STARTED
        assert payload["bx_task"]["uuid"] == publisher_bx_task.uuid()

        topic, headline, payload = self.consumed_news[2]
        assert topic == BxNewsTopics.BXTASK_WAS_FINISHED
        assert payload["bx_task"]["uuid"] == publisher_bx_task.uuid()

        topic, headline, payload = self.consumed_news[3]
        assert topic == BxNewsTopics.BXGATE_WAS_OPENED
        assert payload["bx_gate"]["bx_task_uuid"] == publisher_bx_task.uuid()

        topic, headline, payload = self.consumed_news[4]
        assert topic == BxNewsTopics.BXJOB_SUCCEEDED
        assert payload["bx_job"]["uuid"] == bx_job.uuid()

        topic, headline, payload = self.consumed_news[5]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED
        assert payload["bx_launcher"]["state"] == BxLauncherStates.UNRESPONSIVE
