import asyncio
import logging
import os

import pytest

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
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs

# Object managing bx_launchers.
from dls_bxflow_lib.bx_launchers.states import States as BxLauncherStates

# News events.
from dls_bxflow_lib.bx_news.constants import Topics as BxNewsTopics

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

# Object managing variables.
from dls_bxflow_run.bx_variables.bx_variables import BxVariables

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
@pytest.mark.skipif("not config.getoption('isolated_dataface')")
class TestJobABcDLaptop:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        JobABcDTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class Aclass:
    """
    This is a user-defined class which gets instantiated and called at runtime.
    """

    def __init__(self, outfile, algorithm=None):
        self.outfile = outfile
        self.algorithm = algorithm

    async def run(self):
        logger.info("running Aclass")
        with open(self.outfile, "wt") as stream:
            stream.write(self.algorithm)


# ----------------------------------------------------------------------------------------
class JobABcDTester(BaseContextTester):
    """
    Class to test diamond-shaped dag job creation and running directly, without a workflow involved.
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

            acquired_datafile = f"{output_directory}/acquired_datafile.txt"
            preprocessed1_datafile = f"{output_directory}/preprocessed1_datafile.txt"
            preprocessed2_datafile = f"{output_directory}/preprocessed2_datafile.txt"
            postprocessed_datafile = f"{output_directory}/postprocessed_datafile.txt"

            # Parameters used to pass information into and between tasks.
            # Parameters have persistence and history of change.
            # Can be anything that can be pickled, from scalars to dicts to code.
            variables = BxVariables()

            # Initial directory comes from current beamline and visit knowledge.
            # This can be discovered from environment or /etc configuration.
            # Files expected to arrive here during scan, or already exist for reprocess.
            variables.add("acquired_datafile", acquired_datafile)
            variables.add("preprocessed1_datafile", preprocessed1_datafile)
            variables.add("preprocessed2_datafile", preprocessed2_datafile)
            variables.add("postprocessed_datafile", postprocessed_datafile)

            # Set up series of tasks.  Some are common, some are beamline-written.
            # A task will run when all its dependencies are set.
            acquisition_bx_task = BxTasks().build_object(
                {
                    "type": "dls_bxflow_run.bx_tasks.dummy",
                    "label": "acquisition",
                    RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
                    "type_specific_tbd": {"outfile_variable": "acquired_datafile"},
                }
            )
            acquisition_bx_task.variables = variables

            # Add a preprocess task which depends on the acquisition task.
            preprocess1_bx_task = BxTasks().build_object(
                {
                    "type": "dls_bxflow_run.bx_tasks.dummy",
                    "label": "preprocess1",
                    RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
                    "type_specific_tbd": {"outfile_variable": "preprocessed1_datafile"},
                }
            )
            preprocess1_bx_task.variables = variables
            preprocess1_bx_task.dependency_bx_gates.add(
                acquisition_bx_task.success_bx_gate
            )

            # Add another preprocess task which also depends on the acquisition task.
            preprocess2_bx_task = BxTasks().build_object(
                {
                    "type": "dls_bxflow_run.bx_tasks.dummy",
                    "label": "preprocess2",
                    RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
                    "type_specific_tbd": {
                        "outfile_variable": "preprocessed2_datafile",
                        "delay": 0.0,
                    },
                }
            )
            preprocess2_bx_task.variables = variables
            preprocess2_bx_task.dependency_bx_gates.add(
                acquisition_bx_task.success_bx_gate
            )

            # --------------------------------------------------------------------------

            # Add another preprocess task which also depends on the acquisition task.
            postprocess_bx_task = BxTasks().build_object(
                {
                    "type": "dls_bxflow_run.bx_tasks.dummy",
                    "label": "postprocess",
                    RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
                    "type_specific_tbd": {"outfile_variable": "postprocessed_datafile"},
                }
            )
            postprocess_bx_task.variables = variables
            postprocess_bx_task.dependency_bx_gates.add(
                preprocess1_bx_task.success_bx_gate
            )
            postprocess_bx_task.dependency_bx_gates.add(
                preprocess2_bx_task.success_bx_gate
            )

            # Create a bx_job as collection of tasks.
            bx_job = BxJobs().build_object(
                {"type": "dls_bxflow_lib.bx_jobs.standard", "label": "bx_job1"}
            )
            bx_job.bx_tasks.add(
                [
                    acquisition_bx_task,
                    preprocess1_bx_task,
                    preprocess2_bx_task,
                    postprocess_bx_task,
                ]
            )
            bx_job.blocked_by_bx_gates.add(
                [
                    acquisition_bx_task.failure_bx_gate,
                    preprocess1_bx_task.failure_bx_gate,
                    preprocess2_bx_task.failure_bx_gate,
                    postprocess_bx_task.failure_bx_gate,
                    postprocess_bx_task.success_bx_gate,
                ]
            )

            # Register all the variables that have been added.
            await variables.register(bx_job.uuid())

            # Schedule the bx_job to run.
            await bx_job.enable()

            # Wait for bx_job to finish.
            # await self.wait_for_job_blocked()
            await bx_job.wait(timeout=20.0)

            # Wait for the final news to arrive.
            # TODO: Add method to stop news producer only after flush.
            await asyncio.sleep(0.5)

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

        # Check the out-of-cwd files it wrote.
        assert os.path.exists(acquired_datafile), f"{acquired_datafile} exists"
        assert os.path.exists(
            preprocessed1_datafile
        ), f"{preprocessed1_datafile} exists"
        assert os.path.exists(
            preprocessed2_datafile
        ), f"{preprocessed2_datafile} exists"
        assert os.path.exists(
            postprocessed_datafile
        ), f"{postprocessed_datafile} exists"

        # Check we got all the news.
        news_count = len(self.consumed_news)
        news_count_expected = 21
        if news_count != news_count_expected:
            logger.debug(describe("news", self.consumed_news))

        assert news_count == news_count_expected, "news count"

        topic, headline, payload = self.consumed_news[0]
        assert topic == BxNewsTopics.BXJOB_WAS_ENABLED, "news 0"
        assert payload["bx_job"]["uuid"] == bx_job.uuid()

        topic, headline, payload = self.consumed_news[1]
        assert topic == BxNewsTopics.BXTASK_WAS_STARTED, "news 1"
        assert payload["bx_task"]["uuid"] == acquisition_bx_task.uuid()

        topic, headline, payload = self.consumed_news[-5]
        assert topic == BxNewsTopics.BXJOB_SUCCEEDED, "news -5"
        assert payload["bx_job"]["uuid"] == bx_job.uuid()

        topic, headline, payload = self.consumed_news[-4]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED, "news -4"
        assert payload["bx_launcher"]["state"] == BxLauncherStates.IDLE

        topic, headline, payload = self.consumed_news[-3]
        assert topic == BxNewsTopics.BXJOB_WAS_DELETED, "news -3"
        assert payload["bx_job"]["uuid"] == bx_job.uuid()

        topic, headline, payload = self.consumed_news[-2]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED, "news -2"
        assert payload["bx_launcher"]["state"] == BxLauncherStates.SHUTDOWN

        topic, headline, payload = self.consumed_news[-1]
        assert topic == BxNewsTopics.BXLAUNCHER_WAS_UPDATED, "news -1"
        assert payload["bx_launcher"]["state"] == BxLauncherStates.SHUTDOWN
