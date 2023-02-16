import logging
import os

import pytest

# Remote execution.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestShellGood:
    def test(self, constants, logging_setup, output_directory):
        """
        Tests that a shell task can be run.
        """

        command = ["/bin/date"]

        task_specification = {
            "type": "dls_bxflow_run.bx_tasks.shell",
            "label": "my_shell",
            "remex_hints": {
                RemexKeywords.CLUSTER: RemexClusters.LOCAL,
                RemexKeywords.PARALLEL_ENVIRONMENT: "openmpi 1",
            },
            "type_specific_tbd": {"command": command},
        }

        configuration_file = "tests/configurations/filestore.yaml"
        GoodShellTester(task_specification).main(
            constants,
            configuration_file,
            output_directory,
        )


# ----------------------------------------------------------------------------------------
class TestShellException:
    def test(self, constants, logging_setup, output_directory):
        """
        Tests that a shell task can be run.
        """

        command = ["oops-not-a-command"]

        task_specification = {
            "type": "dls_bxflow_run.bx_tasks.shell",
            "label": "my_shell",
            "remex_hints": {
                RemexKeywords.CLUSTER: RemexClusters.LOCAL,
                RemexKeywords.PARALLEL_ENVIRONMENT: "openmpi 1",
            },
            "type_specific_tbd": {"command": command},
        }

        configuration_file = "tests/configurations/filestore.yaml"
        ExceptionShellTester(task_specification, RuntimeError).main(
            constants,
            configuration_file,
            output_directory,
        )


# ----------------------------------------------------------------------------------------
class TestShellBad:
    def test(self, constants, logging_setup, output_directory):
        """
        Tests that a shell task can be run.
        """

        bad_rc_sh = f"{output_directory}/bad_rc.sh"
        with open(bad_rc_sh, "wt") as stream:
            stream.write("#!/bin/bash\n")
            stream.write(">&2 echo something went wrong here\n")
            stream.write("exit 1\n")

        os.chmod(bad_rc_sh, 0o777)

        command = [bad_rc_sh]

        task_specification = {
            "type": "dls_bxflow_run.bx_tasks.shell",
            "label": "my_shell",
            "remex_hints": {
                RemexKeywords.CLUSTER: RemexClusters.LOCAL,
                RemexKeywords.PARALLEL_ENVIRONMENT: "openmpi 1",
            },
            "type_specific_tbd": {"command": command},
        }

        configuration_file = "tests/configurations/filestore.yaml"
        BadShellTester(
            task_specification,
            expected_error_lines=["something went wrong", "for more"],
        ).main(
            constants,
            configuration_file,
            output_directory,
        )


# ----------------------------------------------------------------------------------------
class GoodShellTester(BaseContextTester):
    """
    Test the shell task.
    """

    def __init__(
        self,
        task_specification,
    ):
        self.__task_specification = task_specification

    async def _main_coroutine(self, constants, output_directory):
        """ """

        os.makedirs(f"{output_directory}/.bxflow")
        original_directory = os.getcwd()

        bx_configurator = self.get_bx_configurator()
        specification = await bx_configurator.load()
        bx_context = BxContexts().build_object(specification)

        async with bx_context:
            try:
                os.chdir(output_directory)

                # Define the class and its constructor arguments.
                bx_task = BxTasks().build_object(self.__task_specification)
                bx_task.set_directory(output_directory)

                return_code = await bx_task.run()

                assert return_code == 0

            finally:
                os.chdir(original_directory)


# ----------------------------------------------------------------------------------------
class ExceptionShellTester(BaseContextTester):
    """
    Test the shell task.
    """

    def __init__(
        self,
        task_specification,
        expected_exception,
    ):
        self.__task_specification = task_specification
        self.__expected_exception = expected_exception

    async def _main_coroutine(self, constants, output_directory):
        """ """

        os.makedirs(f"{output_directory}/.bxflow")
        original_directory = os.getcwd()

        bx_configurator = self.get_bx_configurator()
        specification = await bx_configurator.load()
        bx_context = BxContexts().build_object(specification)

        async with bx_context:
            try:
                os.chdir(output_directory)

                # Define the class and its constructor arguments.
                bx_task = BxTasks().build_object(self.__task_specification)
                bx_task.set_directory(output_directory)

                with pytest.raises(self.__expected_exception):
                    await bx_task.run()

            finally:
                os.chdir(original_directory)


# ----------------------------------------------------------------------------------------
class BadShellTester(BaseContextTester):
    """
    Test the shell task.
    """

    def __init__(
        self,
        task_specification,
        expected_error_lines,
    ):
        self.__task_specification = task_specification
        self.__expected_error_lines = expected_error_lines

    async def _main_coroutine(self, constants, output_directory):
        """ """

        os.makedirs(f"{output_directory}/.bxflow")
        original_directory = os.getcwd()

        bx_configurator = self.get_bx_configurator()
        specification = await bx_configurator.load()
        bx_context = BxContexts().build_object(specification)

        async with bx_context:
            try:
                os.chdir(output_directory)

                # Define the class and its constructor arguments.
                bx_task = BxTasks().build_object(self.__task_specification)
                bx_task.set_directory(output_directory)

                return_code = await bx_task.run()

                assert return_code != 0
                error_lines = bx_task.extract_error_lines()

                assert len(error_lines) == len(self.__expected_error_lines)
                for index, expected_error_line in enumerate(
                    self.__expected_error_lines
                ):
                    if expected_error_line is not None:
                        assert expected_error_line in error_lines[index]

            finally:
                os.chdir(original_directory)
