import logging
import os

import pytest

# Utilities.
from dls_utilpack.describe import describe
from dls_utilpack.module import module_use

# Remote execution.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

# Task constants.
from dls_bxflow_run.bx_tasks.constants import Types as BxTaskTypes

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------


@pytest.mark.skipif("not config.getoption('gpu')")
class TestPtyrexMpiGood:
    def test(self, constants, logging_setup, output_directory):
        """
        Tests that a ptyrex_mpi task can be run.
        """

        ptyrex_configfile = f"{os.path.dirname(__file__)}/configurations/ptyrex_configfiles/region_p4_p6.json"
        ptyrex_configfile_updated_fields = {}
        ptyrex_configfile_substitutions = {}

        # The data filename is checked by the task, so it must exist.
        data_filename = f"{output_directory}/some_data.hdf5"
        with open(data_filename, "w") as stream:
            stream.write("")

        task_specification = {
            "type": BxTaskTypes.PTYREX_MPI,
            "label": "my_ptyrex_mpi",
            "remex_hints": {
                RemexKeywords.CLUSTER: RemexClusters.LOCAL,
                RemexKeywords.PARALLEL_ENVIRONMENT: "openmpi 1",
            },
            "type_specific_tbd": {
                "data_filename": data_filename,
                "ptyrex_configfile": ptyrex_configfile,
                "ptyrex_configfile_updated_fields": ptyrex_configfile_updated_fields,
                "ptyrex_configfile_substitutions": ptyrex_configfile_substitutions,
                # The pytrex_mpi will run in a script in which these modules are loaded.
                "load_modules": {
                    "directories": [output_directory],
                    "modules": ["my_ptyrex_module"],
                },
            },
        }

        configuration_file = "tests/configurations/filestore.yaml"
        PtyrexMpiTester(task_specification).main(
            constants,
            configuration_file,
            output_directory,
        )


# ----------------------------------------------------------------------------------------
class PtyrexMpiTester(BaseContextTester):
    """
    Test the ptyrex_mpi task.
    """

    def __init__(
        self,
        task_specification,
        expected_script_error=None,
    ):
        self.__task_specification = task_specification
        self.__expected_script_error = expected_script_error

    async def _main_coroutine(self, constants, output_directory):
        """ """

        # Write a modulefile that will be run as part of the pytrex_mpi task.
        # This module will set the PATH to include the stub_commands directory,
        # in which there will be a dummy mpirun command.
        with open(f"{output_directory}/my_ptyrex_module", "w") as stream:
            stream.write("#%Module1.0\n")
            stream.write(
                f"setenv  PATH my_good_path1:my_good_path2:{os.path.dirname(__file__)}/stub_commands:{os.environ['PATH']}\n"
            )

        # Make sure there is at least a stderr.txt for extract_error_lines
        # to look in if there is no ptyrex_mpi_stderr.txt.
        with open(f"{output_directory}/stderr.txt", "w") as stream:
            stream.write("")

        os.environ.update(module_use(output_directory))

        # --------------------------------------------------------------------

        os.makedirs(f"{output_directory}/.bxflow")
        original_directory = os.getcwd()

        bx_configurator = self.get_bx_configurator()
        specification = await bx_configurator.load()
        # Remex hints for task type.
        specification["bx_task_remex_hints"] = {BxTaskTypes.PTYREX_MPI: {}}

        bx_context = BxContexts().build_object(specification)

        async with bx_context:
            try:
                os.chdir(output_directory)

                # Define the class and its constructor arguments.
                bx_task = BxTasks().build_object(self.__task_specification)
                bx_task.set_directory(output_directory)

                return_code = await bx_task.run()

                if self.__expected_script_error is None:
                    assert return_code == 0
                    error_lines = bx_task.extract_error_lines()
                    assert isinstance(error_lines, list)
                    assert len(error_lines) == 0

                    # TODO: In test_ptyrex_mpi, check the yaml file to make sure the visit.directory is replaced.
                    assert os.path.exists(f"{output_directory}/ptyrex_configfile.yaml")
                else:
                    # assert return_code != 0
                    error_lines = bx_task.extract_error_lines()
                    logger.debug(describe("error_lines", error_lines))
                    assert isinstance(error_lines, list)
                    assert len(error_lines) == 2
                    assert self.__expected_script_error in error_lines[0]
                    assert error_lines[1].startswith("for more")

            finally:
                os.chdir(original_directory)
