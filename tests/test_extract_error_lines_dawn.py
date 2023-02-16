import copy
import logging
import os

# Remex (remote execution) API.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Filestore manager.
from dls_bxflow_lib.bx_filestores.bx_filestores import bx_filestores_get_default

# BxJobs manager.
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs, bx_jobs_get_default

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks
from dls_bxflow_run.bx_tasks.constants import ExtractionErrorLinesMessages

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)

bx_job_uuid = "bx_job_uuid-0001"


# ----------------------------------------------------------------------------------------
class TestExtractErrorLinesDawn:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/filestore.yaml"
        ErrorLinesDawnTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class ErrorLinesDawnTester(BaseContextTester):
    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()
        specification = await bx_configurator.load()
        bx_context = BxContexts().build_object(specification)

        async with bx_context:
            await self.no_file()
            await self.bad_file()
            await self.empty_file()
            await self.java_file()

    # ------------------------------------------------------------------------
    async def build_task(self, bx_task_uuid):
        """
        Build a dawm task object.
        """

        # Build a job object, needed to locate the output files.
        bx_job_specification = copy.deepcopy(bx_jobs_get_default().specification())
        bx_job_specification["label"] = "my/good/job"
        bx_job = BxJobs().build_object(
            bx_job_specification, predefined_uuid=bx_job_uuid
        )
        bx_filestores_get_default().pin_job_directory(bx_job)

        bx_task_specification = {
            "type": "dls_bxflow_run.bx_tasks.dawn1",
            "label": "my_dummy_task",
            "remex_hints": {
                RemexKeywords.CLUSTER: RemexClusters.LOCAL,
            },
            "bx_job_uuid": bx_job.uuid(),
        }

        # Build the task.
        bx_task = BxTasks().build_object(
            bx_task_specification, predefined_uuid=bx_task_uuid
        )
        bx_filestores_get_default().pin_task_directory(bx_job, bx_task)

        # Runtime directory where the isolated task wrote its files.
        runtime_directory = bx_task.get_directory()

        # Name of of the stderr filename.
        # TODO: Centralize naming convention of stderr.txt filename.
        stderr_filename = f"{runtime_directory}/dawn_stdout.txt"

        return bx_job, bx_task, stderr_filename

    # ----------------------------------------------------------------------------------------
    async def no_file(self):
        """ """

        bx_job, bx_task, stderr_filename = await self.build_task("bx_task_uuid_no_file")

        error_lines = bx_task.extract_error_lines()

        assert len(error_lines) == 1
        assert ExtractionErrorLinesMessages.DOES_NOT_EXIST in error_lines[0]

    # ----------------------------------------------------------------------------------------
    async def bad_file(self):
        """ """

        bx_job, bx_task, stderr_filename = await self.build_task(
            "bx_task_uuid_bad_file"
        )

        # Make a directory at the filename path, which is a problem.
        os.makedirs(stderr_filename)

        # Extract the error lines from the file.
        error_lines = bx_task.extract_error_lines()

        assert len(error_lines) == 1
        assert ExtractionErrorLinesMessages.PROBLEM_READING in error_lines[0]

    # ----------------------------------------------------------------------------------------
    async def empty_file(self):
        """ """

        bx_job, bx_task, stderr_filename = await self.build_task(
            "bx_task_uuid_empty_file"
        )

        os.makedirs(os.path.dirname(stderr_filename))
        with open(stderr_filename, "w") as stream:
            stream.write("")

        # Extract the error lines from the file.
        error_lines = bx_task.extract_error_lines()

        assert len(error_lines) == 1
        assert ExtractionErrorLinesMessages.EXISTS_BUT_IS_EMPTY in error_lines[0]

    # ----------------------------------------------------------------------------------------
    async def java_file(self):
        """
        File appears formatted by java.
        """

        bx_job, bx_task, stderr_filename = await self.build_task(
            "bx_task_uuid_formatted_file"
        )

        message_part = "message part only"
        os.makedirs(os.path.dirname(stderr_filename))
        with open(stderr_filename, "w") as stream:
            stream.write(f"Started IEventService {message_part}\n")
            stream.write(f"08:14:47.091 ERROR {message_part}\n")

        # Extract the error lines from the file.
        error_lines = bx_task.extract_error_lines()

        assert len(error_lines) == 1
        assert message_part == error_lines[0]
