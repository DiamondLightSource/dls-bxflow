import asyncio
import copy
import logging
import os
import time

# Object managing datafaces.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Remote execution.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing launchers.
from dls_bxflow_lib.bx_filestores.bx_filestores import bx_filestores_get_default

# BxJobs manager.
from dls_bxflow_lib.bx_jobs.bx_jobs import bx_jobs_get_default
from dls_bxflow_lib.bx_launchers.base import EXIT_CODES

# BxLaunchers manager.
from dls_bxflow_lib.bx_launchers.bx_launchers import BxLaunchers

# BxTasks manager.
from dls_bxflow_run.bx_tasks.constants import Keywords as BxTaskKeywords
from dls_bxflow_run.bx_tasks.states import States as BxTaskStates

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


bx_job_uuid = "00001-2222-3333-4444-5555555"

bx_job_capsule = (
    "my good job",
    bx_job_uuid,
)

bx_task_uuid = "00002-2222-3333-4444-5555555"
bx_task_label = "my launcher task (part 1)"


# ----------------------------------------------------------------------------------------
class TestLauncherPopenerGoodLaptop:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"
        # The launcher configuration to replace in the configuration file for this test.
        desired_bx_launcher_specification = "bx_launcher_popener_specification"

        # Good test.
        bx_task_type_specific_tbd = {"delay": 0}
        expected_exit_code = 0
        LauncherTester(
            desired_bx_launcher_specification,
            bx_task_type_specific_tbd,
            expected_exit_code,
            None,
        ).main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class TestLauncherPopenerBadTaskLaptop:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"
        # The launcher configuration to replace in the configuration file for this test.
        desired_bx_launcher_specification = "bx_launcher_popener_specification"

        # Bad test.
        bx_task_type_specific_tbd = {"deliberate_error": "deliberate_error"}
        expected_exit_code = 1
        expected_error_lines = ["Dummy run method", "deliberate_error"]
        LauncherTester(
            desired_bx_launcher_specification,
            bx_task_type_specific_tbd,
            expected_exit_code,
            expected_error_lines,
        ).main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class TestLauncherPopenerBadPrepLaptop:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"
        # The launcher configuration to replace in the configuration file for this test.
        desired_bx_launcher_specification = "bx_launcher_popener_specification"

        # Bad test.
        bx_task_type_specific_tbd = {}
        bx_task_prepare_environment = ["bad_things_happen"]
        expected_exit_code = EXIT_CODES.PREPARE_ENVIRONMENT
        expected_error_lines = ["bad_things_happen", "prepare_environment"]
        LauncherTester(
            desired_bx_launcher_specification,
            bx_task_type_specific_tbd,
            expected_exit_code,
            expected_error_lines,
            bx_task_prepare_environment=bx_task_prepare_environment,
        ).main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class TestLauncherQsubberGoodLaptop:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"
        # The launcher configuration to replace in the configuration file for this test.
        desired_bx_launcher_specification = "bx_launcher_qsubber_specification"

        # Good test.
        bx_task_type_specific_tbd = {"delay": 0}
        expected_exit_code = 0
        LauncherTester(
            desired_bx_launcher_specification,
            bx_task_type_specific_tbd,
            expected_exit_code,
            None,
        ).main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class LauncherTester(BaseContextTester):
    """
    Class to test the launcher running jobs directly, without a workflow involved.
    """

    def __init__(
        self,
        desired_bx_launcher_specification,
        bx_task_type_specific_tbd,
        expected_exit_code,
        expected_error_lines,
        bx_task_prepare_environment=[],
    ):
        BaseContextTester.__init__(self)

        self.__desired_bx_launcher_specification = desired_bx_launcher_specification
        self.__bx_task_type_specific_tbd = bx_task_type_specific_tbd
        self.__expected_exit_code = expected_exit_code
        self.__expected_error_lines = expected_error_lines
        self.__bx_task_prepare_environment = bx_task_prepare_environment

    async def _main_coroutine(self, constants, output_directory):
        """ """

        # Make the fake qsub findable in the path.
        os.environ["PATH"] = "%s:%s" % (os.path.dirname(__file__), os.environ["PATH"])

        # Supply environment variable for substitution in the bx configurator.
        os.environ["OUTPUT_DIRECTORY"] = output_directory

        # Directory where the task should write its files.
        bx_task_directory = f"{output_directory}/task_directory"

        bx_configurator = self.get_bx_configurator()

        # Don't start the services we don't need for this test.
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("dls_servbase_dataface_specification")
        bx_configurator.remove("bx_gui_specification")

        # Load the configuration file and resolve the substitutions.
        context_configuration = await bx_configurator.load()

        # Keep only the one desired bx_launcher_specification.
        bx_launcher_specifications = context_configuration["bx_launcher_specifications"]
        bx_launcher_specifications.clear()
        bx_launcher_specifications.append(self.__desired_bx_launcher_specification)

        bx_context = BxContexts().build_object(context_configuration)

        # Files discovered after the task has run.
        execution_outputs = []

        async with bx_context:
            # Build a client to our desired launcher.
            launcher = BxLaunchers().build_object(
                bx_configurator.require(self.__desired_bx_launcher_specification)
            )

            try:
                bx_job_specification = copy.deepcopy(
                    bx_jobs_get_default().specification()
                )
                bx_job_specification["label"] = "my/good/job"
                bx_job_specification["directory"] = "my_good_job"

                # This file will be written by dummy outside the filestore.
                dummy_outfile = "%s/dummy_datafile.json" % (output_directory)
                bx_task_specification = {
                    "uuid": bx_task_uuid,
                    "type": "dls_bxflow_run.bx_tasks.dummy",
                    "label": bx_task_label,
                    "directory": bx_task_directory,
                    "bx_job_uuid": bx_job_uuid,
                    # TODO: Add test to make sure BxTaskKeywords.NEEDS_DATAFACE does the right thing.
                    BxTaskKeywords.NEEDS_DATAFACE: True,
                    # The following keywords are honored:
                    # cluster
                    # redhat_release
                    # m_mem_free
                    # h_rt
                    # q
                    # pe
                    # gpu
                    # gpu_arch
                    # h
                    RemexKeywords.HINTS: {
                        RemexKeywords.CLUSTER: [
                            RemexClusters.LOCAL,
                            RemexClusters.TEST,
                        ],
                        RemexKeywords.REDHAT_RELEASE: "7",
                        RemexKeywords.MEMORY_LIMIT: "123G",
                        RemexKeywords.TIME_LIMIT: "8:00:11",
                        RemexKeywords.QUEUE: "high.q",
                        RemexKeywords.PARALLEL_ENVIRONMENT: "smp 1",
                        RemexKeywords.GPU: "2",
                        RemexKeywords.GPU_ARCH: "Volta",
                        RemexKeywords.HOST: "!(a,b)",
                    },
                    "type_specific_tbd": {"outfile": dummy_outfile, "delay": 0},
                    BxTaskKeywords.PREPARE_ENVIRONMENT: self.__bx_task_prepare_environment,
                }

                # Override the task behavior as specified for this test.
                bx_task_specification["type_specific_tbd"].update(
                    self.__bx_task_type_specific_tbd
                )

                # bx_task = BxTasks().build_object(bx_task_specification)
                await bx_datafaces_get_default().set_bx_tasks([bx_task_specification])

                # Submit the bx_task.
                await launcher.submit(
                    bx_job_uuid,
                    bx_job_specification,
                    bx_task_uuid,
                    bx_task_specification,
                )

                time0 = time.time()
                timeout = 5.0
                while True:
                    # Let the server harvest the running task.
                    await launcher.harvest()

                    # Check the task status.
                    task_record = await bx_datafaces_get_default().get_bx_task(
                        bx_task_uuid
                    )

                    # Task finished?
                    if task_record["state"] == BxTaskStates.FINISHED:
                        break
                    if time.time() - time0 > timeout:
                        raise RuntimeError(
                            f"bx_task not finished within {timeout} seconds"
                        )
                    await asyncio.sleep(1.0)

                if self.__expected_exit_code == 0:
                    execution_outputs = (
                        bx_filestores_get_default().get_runtime_execution_outputs(
                            bx_task_directory
                        )
                    )

            finally:
                await launcher.close_client_session()

        # For convenience of debugging, say what files got written
        if (
            task_record["exit_code"] != self.__expected_exit_code
            or task_record["exit_code"] != 0
        ):
            logger.info(f"{len(execution_outputs)} residual files were produced")
            for execution_output in execution_outputs:
                logger.info(f"  {execution_output.filename}")

        if task_record["exit_code"] != self.__expected_exit_code:
            raise AssertionError(
                "task had exit_code %d but expected %d"
                % (task_record["exit_code"], self.__expected_exit_code)
            )

        if self.__expected_exit_code == 0:
            assert len(task_record["error_lines"]) == 0

            # Verify all the dummy output file contents.
            # TODO: Make launcher test actually execute the script by qsub.
            # with open(dummy_outfile, "r") as stream:
            #     dummy_output = json.load(stream)
            #     assert dummy_output["bx_task_label"] == bx_task_label

            # Verify the existence of the residual files.
            self.assert_execution_residuals(execution_outputs)

        else:

            error_lines = task_record["error_lines"].split("\n")
            assert len(error_lines) == len(self.__expected_error_lines)

            for index, line in enumerate(self.__expected_error_lines):
                assert line in error_lines[index], f"error line {index}"
