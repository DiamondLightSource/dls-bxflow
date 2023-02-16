import logging
import os

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
class TestPtypyMpiGood:
    def test(self, constants, logging_setup, output_directory):
        """
        Tests that a ptypy_mpi task can be run.
        """

        # Make the fake ptypy_mpi_recipe findable in the path.
        ptypy_configfile = f"{os.path.dirname(__file__)}/configurations/ptypy_configfiles/i14_unknown_probe_dm.yaml"
        ptypy_configfile_updated_fields = {}

        task_specification = {
            "type": BxTaskTypes.PTYPY_MPI,
            "label": "my_ptypy_mpi",
            "remex_hints": {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
            "type_specific_tbd": {
                "data_filename": f"{output_directory}/i14-123456.nxs",
                "ptypy_configfile": ptypy_configfile,
                "ptypy_configfile_updated_fields": ptypy_configfile_updated_fields,
                "ptypy_configfile_substitutions": {},
                "ptypy_version": "my_ptypy_module",
                "propagate": "3.5",
            },
        }

        configuration_file = "tests/configurations/filestore.yaml"
        PtypyMpiTester(task_specification).main(
            constants,
            configuration_file,
            output_directory,
        )


# ----------------------------------------------------------------------------------------
class TestPtypyMpiBad:
    def test(self, constants, logging_setup, output_directory):
        """
        Tests that a ptypy_mpi task can be and error handled.
        """

        # Make the fake ptypy_mpi_recipe findable in the path.
        ptypy_configfile = f"{os.path.dirname(__file__)}/configurations/ptypy_configfiles/i14_unknown_probe_dm.yaml"
        ptypy_configfile_updated_fields = {}

        # Test that the ptypy_mpi_recipe script fails with exit code, and parse error out of file.
        # TODO: TestPtypyMPI should cover more error cases such as ptypy_mpi_recipe exits with code 0, even though logs errors.
        task_specification = {
            "type": BxTaskTypes.PTYPY_MPI,
            "label": "my_ptypy_mpi",
            "remex_hints": {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
            "type_specific_tbd": {
                "data_filename": f"{output_directory}/i14-123456.nxs",
                "ptypy_configfile": ptypy_configfile,
                "ptypy_configfile_updated_fields": ptypy_configfile_updated_fields,
                "ptypy_configfile_substitutions": {},
                "ptypy_version": "my_ptypy_module",
                "propagate": "bad_propagate",
            },
        }

        configuration_file = "tests/configurations/filestore.yaml"
        PtypyMpiTester(
            task_specification,
            expected_script_error="FileNotFoundError",
        ).main(
            constants,
            configuration_file,
            output_directory,
        )


# ----------------------------------------------------------------------------------------
class PtypyMpiTester(BaseContextTester):
    """
    Test the ptypy_mpi task.
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

        with open(f"{output_directory}/my_ptypy_module", "w") as stream:
            stream.write("#%Module1.0\n")
            stream.write(
                f"setenv  PATH my_good_path1:my_good_path2:{os.path.dirname(__file__)}/stub_commands:{os.environ['PATH']}\n"
            )

        os.environ.update(module_use(output_directory))

        # --------------------------------------------------------------------

        os.makedirs(f"{output_directory}/.bxflow")
        original_directory = os.getcwd()

        bx_configurator = self.get_bx_configurator()
        specification = await bx_configurator.load()
        # Remex hints for task type.
        specification["bx_task_remex_hints"] = {BxTaskTypes.PTYPY_MPI: {}}

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

                    # TODO: In test_ptypy_mpi, check the yaml file to make sure the visit.directory is replaced.
                    assert os.path.exists(f"{output_directory}/ptypy_configfile.yaml")
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
