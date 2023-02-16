import asyncio
import copy
import logging
import os
import uuid

import pytest

# Utilities.
from dls_utilpack.describe import describe
from dls_utilpack.isodatetime import isodatetime

# Database field names.
from dls_bxflow_api.bx_databases.constants import BxJobFieldnames

# Object managing catalogs.
from dls_bxflow_lib.bx_catalogs.bx_catalogs import bx_catalogs_get_default

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing bx_jobs.
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs, bx_jobs_get_default

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)

bx_job1_uuid = str(uuid.uuid4())
bx_job2_uuid = str(uuid.uuid4())
bx_job_label = "mighty workflow"
bx_task_uuid = str(uuid.uuid4())
bx_task_label = "great task"
attachment_path = "/a/b/c/d.log"


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Catalog access directly.
@pytest.mark.skipif("not config.getoption('ispyb')")
class TestCatalogDirect:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        CatalogDirectTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Catalog access through the server.
@pytest.mark.skipif("not config.getoption('ispyb')")
class TestCatalogIndirect:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        CatalogIndirectTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
class CatalogDirectTester(BaseContextTester):
    """
    Test the catalog api by direct call.
    """

    os.environ["ISPYB_CREDENTIALS"] = "tests/configurations/ispyb-local.cfg"

    async def _main_coroutine(self, constants, output_directory):
        """ """

        now = isodatetime()

        bx_job1_record = {
            BxJobFieldnames.UUID: bx_job1_uuid,
            BxJobFieldnames.LABEL: bx_job_label,
            BxJobFieldnames.DATA_LABEL: f"mydata/{now}",
            BxJobFieldnames.VISIT: "cy29757-3",
        }
        bx_job2_record = {
            BxJobFieldnames.UUID: bx_job2_uuid,
            BxJobFieldnames.LABEL: bx_job_label,
            BxJobFieldnames.DATA_LABEL: f"mydata/{now}",
            BxJobFieldnames.VISIT: "cy29757-3",
        }
        bx_configurator = self.get_bx_configurator()

        # Don't build a launcher or scheduler.
        bx_configurator.remove("bx_launcher_specifications")
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()

        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:
            # For short.
            bx_catalog = bx_catalogs_get_default()

            await bx_catalog.create_workflow_run(bx_job1_record)

            # Create second workflow on the same data label.
            await bx_catalog.create_workflow_run(bx_job2_record)

            # Change the message.
            message = "test_message"
            await bx_catalog.update_workflow_run_message(bx_job1_uuid, message)

            # Attach a file.
            await bx_catalog.attach_workflow_run_file(
                bx_job1_uuid, bx_task_uuid, attachment_path
            )

            # Read the workflow from the catalog database.
            workflow_run = await bx_catalog.query_workflow_run(bx_job1_uuid)
            logger.info(describe("workflow_run", workflow_run))
            assert workflow_run["message"] == "test_message"

            # Read the workflow's files from the catalog database.
            workflow_run_files = await bx_catalog.query_workflow_run_files(bx_job1_uuid)
            logger.info(describe("workflow_run_files", workflow_run_files))
            assert len(workflow_run_files) == 1

            # Read the second workflow's files from the catalog database.
            workflow_run_files = await bx_catalog.query_workflow_run_files(bx_job2_uuid)
            logger.info(describe("workflow_run_files", workflow_run_files))
            assert len(workflow_run_files) == 0


# ----------------------------------------------------------------------------------------
class CatalogIndirectTester(BaseContextTester):
    """
    Test catalog server which self-discovers work from news.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()
        context_configuration = await bx_configurator.load()

        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:
            # Define the task.
            bx_task = BxTasks().build_object(
                {
                    "type": "dls_bxflow_run.bx_tasks.dummy",
                    "label": "task1",
                    "type_specific_tbd": {"delay": 2.0},
                }
            )

            # Create a bx_job.
            bx_job_specification = copy.deepcopy(bx_jobs_get_default().specification())
            bx_job_specification["label"] = "job1"
            bx_job = BxJobs().build_object(bx_job_specification)

            # Add the task to the job.
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

            # Wait a second for the task to get scheduled and launched.
            await asyncio.sleep(1.0)

            # Get the workflow from the catalog database.
            workflow_run = await bx_catalogs_get_default().query_workflow_run(
                bx_job.uuid()
            )
            # logger.info(describe("in-process workflow_run", workflow_run))
            assert workflow_run["message"] == "bxflow task task1 started"

            # Wait for bx_job to finish.
            await bx_job.wait(timeout=5.0)

            # Give a chance for the news to propagate.
            await asyncio.sleep(0.2)

            # Read the workflow from the catalog database.
            workflow_run = await bx_catalogs_get_default().query_workflow_run(
                bx_job.uuid()
            )
            # logger.info(describe("final workflow_run", workflow_run))
            assert workflow_run["message"] == "bx_job finished sucessfully"

            # # Read the workflow's files from the catalog database.
            # workflow_run_files = (
            #     await bx_catalogs_get_default().query_workflow_run_files(bx_job.uuid())
            # )
            # logger.info(describe("workflow_run_files", workflow_run_files))
            # assert len(workflow_run_files) == 0
