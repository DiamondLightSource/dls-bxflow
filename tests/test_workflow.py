import logging
import os

from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

# The workflows we want to test.
from tests.workflows.a.workflow import A
from tests.workflows.c.workflow import C

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestWorkflowA:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Run the test in a coroutine.
        configuration_file = "tests/configurations/backend.yaml"
        WorkflowATester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class TestWorkflowC:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Run the test in a coroutine.
        configuration_file = "tests/configurations/backend.yaml"
        WorkflowCTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class WorkflowATester(BaseContextTester):
    """
    Class to test any type of configurator.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()

        # Don't start the services we don't need for this test.
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()

        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:

            workflow = A(outfile="/some/file")

            workflow.build()

            # Enter the workflow into the database.
            # Since there is no scheduler, the job's task's will not run.
            await workflow.start()

            bx_job_record = await bx_datafaces_get_default().get_bx_job(
                workflow.bx_job.uuid()
            )
            bx_workflow_records = await bx_datafaces_get_default().get_bx_workflows(
                [bx_job_record["uuid"]]
            )

            assert len(bx_workflow_records) == 1, "number of workflow records"

            assert bx_job_record["bx_workflow_uuid"] == workflow.uuid()

            got_filename_classname = bx_workflow_records[0]["filename_classname"]
            expect_filename_classname = (
                f"{os.path.dirname(__file__)}/workflows/a/workflow.py::A"
            )
            assert got_filename_classname == expect_filename_classname


# ----------------------------------------------------------------------------------------
class WorkflowCTester(BaseContextTester):
    """
    Class to test any type of configurator.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()

        # Don't start the services we don't need for this test.
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()

        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:

            workflow = C(notebook="c", X="new X")

            workflow.build()

            # Enter the workflow into the database.
            # Since there is no scheduler, the job's task's will not run.
            await workflow.start()

            bx_job_record = await bx_datafaces_get_default().get_bx_job(
                workflow.bx_job.uuid()
            )
            bx_workflow_records = await bx_datafaces_get_default().get_bx_workflows(
                [bx_job_record["uuid"]]
            )

            assert len(bx_workflow_records) == 1, "number of workflow records"

            assert bx_job_record["bx_workflow_uuid"] == workflow.uuid()

            got_filename_classname = bx_workflow_records[0]["filename_classname"]
            expect_filename_classname = (
                f"{os.path.dirname(__file__)}/workflows/c/workflow.py::C"
            )
            assert got_filename_classname == expect_filename_classname

            got_job_label = bx_job_record["label"]
            expect_job_label = "c"
            assert got_job_label == expect_job_label

            python_variable_assignment = (
                workflow.get_settings().compose_python_variable_assignment()
            )
            python_variable_assignment = "\n".join(python_variable_assignment)
            logger.debug(f"python_variable_assignment\n{python_variable_assignment}")
