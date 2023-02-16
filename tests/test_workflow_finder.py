import logging

import pytest

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Objects we may need to access
from dls_bxflow_lib.bx_workflows.workflow_finder import WorkflowFinder

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestWorkflowFinder:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Run the test in a coroutine.
        configuration_file = "tests/configurations/backend.yaml"
        WorkflowFinderTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class WorkflowFinderTester(BaseContextTester):
    """
    Class to test any type of configurator.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

        # self.notfound("A", "no configurator")

        bx_configurator = self.get_bx_configurator()

        # Don't start the services we don't need for this test.
        bx_configurator.remove("bx_news_specification")
        bx_configurator.remove("bx_dataface_specification")
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_launcher_specifications")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()

        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:

            # There is no A at all.
            self.notfound("A", ["A.workflow_filename_classname"])

            # There is a D configured, but there is no python for it.
            self.notfound(
                "D", ["could not find python file", "could not get python module"]
            )

            # There is a B configured.
            self.found("B", ["B configurated", "got class B"])

            # There is no C configured, but there is a python file.
            self.found("tests/workflows/c/workflow.py::C", ["got class C"])

            # Get G from python file with a class Workflow.
            self.found(
                "tests/workflows/g/workflow.py", ["got class Workflow", "from file"]
            )

            # Get G from python module with a class Workflow.
            self.found(
                "tests.workflows.g.workflow", ["got class Workflow", "from module"]
            )

    # ----------------------------------------------------------------------------------------
    def found(self, workflow_name, expecteds):

        workflow_finder = WorkflowFinder()

        workflow_finder.find_class_object(workflow_name)

        text = workflow_finder.compose_messages_as_text(workflow_name)

        logger.debug(f"expected found result for {workflow_name}:\n{text}")

        for expected in expecteds:
            assert (
                expected in text
            ), f'{workflow_name} did not get expected "{expected}"'

    # ----------------------------------------------------------------------------------------
    def notfound(self, workflow_name, expecteds):

        workflow_finder = WorkflowFinder()

        with pytest.raises(NotFound) as exception_info:
            workflow_finder.find_class_object(workflow_name)

        text = str(exception_info.value)

        logger.debug(f"expected not-found result for {workflow_name}:\n{text}")

        for expected in expecteds:
            assert (
                expected in text
            ), f'{workflow_name} did not get expected "{expected}"'
