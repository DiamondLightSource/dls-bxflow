import asyncio
import logging
import multiprocessing
import os

import pytest

# Object managers we interact with.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Results composers.
from dls_bxflow_lib.bx_composers.bx_composers import BxComposers

# Configurator.
from dls_bxflow_lib.bx_configurators.bx_configurators import (
    BxConfigurators,
    bx_configurators_set_default,
)

# Filestore manager.
from dls_bxflow_lib.bx_filestores.bx_filestores import bx_filestores_get_default

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class BaseContextTester:
    """
    This is a base class for tests which use BxContext.
    """

    def __init__(self):
        # For a capturing the news events as they arrive.
        self.consumed_news = []
        self.tasks_execution_outputs = {}
        self.residuals = ["stdout.txt", "stderr.txt", "main.log"]

        # Temporary directory (created later) for the live of this main program.
        self.__temporary_directory = None

    def main(self, constants, configuration_file, output_directory):
        """
        This is the main program which calls the test using asyncio.
        """

        # Save these for when the configuration is loaded.
        self.__configuration_file = configuration_file
        self.__output_directory = output_directory

        multiprocessing.current_process().name = "main"

        # self.__blocked_event = asyncio.Event()

        failure_message = None
        try:
            # Run main test in asyncio event loop.
            asyncio.run(self._main_coroutine(constants, output_directory))

        except Exception as exception:
            logger.exception(
                "unexpected exception in the test method", exc_info=exception
            )
            failure_message = str(exception)

        if failure_message is not None:
            pytest.fail(failure_message)

    # ----------------------------------------------------------------------------------------
    def get_bx_configurator(self):

        bx_configurator = BxConfigurators().build_object(
            {
                "type": "dls_bxflow_lib.bx_configurators.yaml",
                "type_specific_tbd": {"filename": self.__configuration_file},
            }
        )

        # For convenience, always do these replacement.
        bx_configurator.substitute({"output_directory": self.__output_directory})
        bx_configurator.substitute({"temporary_directory": self.__output_directory})

        # Add various things from the environment into the configurator.
        bx_configurator.substitute(
            {
                "CWD": os.getcwd(),
                "PYTHONPATH": os.environ.get("PYTHONPATH", "PYTHONPATH"),
            }
        )

        # Set the global value of our configurator which might be used in other modules.
        bx_configurators_set_default(bx_configurator)

        return bx_configurator

    # # -------------------------------------------------------------------------------
    # async def wait_for_job_blocked(self, bx_job):
    #     """ """
    #     await self.__blocked_event.wait()

    # ---------------------------------------------------------------------------------
    async def _consume_bx_news(self, topic, headline, details):
        """Receive incoming news.
        Keep a list of what newsfeed messages were received for use in asserts.
        """

        self.consumed_news.append((topic, headline, details))

    # ---------------------------------------------------------------------------------
    async def _compose_job_summary(
        self,
        bx_job_uuid,
    ):

        # Make a text composer.
        bx_composer = BxComposers().build_object(
            {"type": "dls_bxflow_lib.bx_composers.text"}
        )

        # Get the news and compose it.
        bx_news_records = await bx_datafaces_get_default().get_bx_news(
            bx_job_uuid=bx_job_uuid
        )
        news_text = bx_composer.compose_bx_news(bx_news_records)

        # Get the job record.
        bx_job_record = await bx_datafaces_get_default().get_bx_job(bx_job_uuid)

        # Get the job's tasks/gates records.
        bx_jobs_bx_tasks_bx_gates_records = (
            await bx_datafaces_get_default().get_bx_jobs_bx_tasks_bx_gates(
                bx_job_uuid=bx_job_uuid
            )
        )
        details_text = bx_composer.compose_bx_job_details(
            bx_job_record, bx_jobs_bx_tasks_bx_gates_records
        )

        bx_variable_records = await bx_datafaces_get_default().get_bx_variables(
            bx_job_uuid
        )
        variables_text = bx_composer.compose_bx_variables(bx_variable_records)

        return f"{news_text}\n\n{details_text}\n\n{variables_text}"

    # ----------------------------------------------------------------------------------------
    def capture_tasks_execution_outputs(self, bx_job):
        """
        Capture the files which were output by each task.
        """

        filestore = bx_filestores_get_default()
        for bx_task in bx_job.bx_tasks.list():
            bx_task_uuid = bx_task.uuid()
            execution_outputs = filestore.get_runtime_execution_outputs(
                bx_task.get_directory()
            )
            self.tasks_execution_outputs[bx_task_uuid] = execution_outputs

    # ----------------------------------------------------------------------------------------
    def assert_tasks_execution_residuals(self):
        """
        Make sure all the residuals are there.
        """

        for execution_outputs in self.tasks_execution_outputs.values():
            self.assert_execution_residuals(execution_outputs)

    # ----------------------------------------------------------------------------------------
    def assert_execution_residuals(self, execution_outputs, expect_stderr_empty=True):
        """
        Make sure all the residuals are there.
        """

        for residual in self.residuals:
            self._assert_execution_output(
                residual, execution_outputs, expect_stderr_empty=expect_stderr_empty
            )

    # ----------------------------------------------------------------------------------------
    def _assert_execution_output(
        self,
        basename,
        execution_outputs,
        expected_content=None,
        expect_stderr_empty=True,
    ):

        execution_output = None
        for execution_output in execution_outputs:

            if execution_output.basename == basename:
                if expected_content is not None:
                    with open(execution_output.filename, "r") as stream:
                        actual_content = stream.read()
                        assert (
                            actual_content == expected_content
                        ), f"expected content of {execution_output.filename}\n  "
                        f"expected {expected_content}\n  actual {actual_content}"

                elif basename == "stderr.txt":
                    if expect_stderr_empty:
                        assert (
                            execution_output.bytes == 0
                        ), f"{execution_output.filename} expects zero length"
                    else:
                        assert (
                            execution_output.bytes != 0
                        ), f"{execution_output.filename} expects non-zero length"
                return

        assert False, f"did not find residual file {basename}"
