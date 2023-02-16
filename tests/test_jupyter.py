import logging
import os

import pytest

# Utilities.
from dls_utilpack.sanitize import sanitize

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
class TestJupyterDirectGood:
    def test(self, constants, logging_setup, output_directory):
        """
        Tests that a jupyter task can be run.
        """

        ipynb_filename = f"{os.getcwd()}/tests/jupyter/jupyter1.ipynb"

        task_specification = {
            "type": "dls_bxflow_run.bx_tasks.jupyter",
            "label": "my_jupyter",
            RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
            "type_specific_tbd": {"ipynb_filename": ipynb_filename},
        }

        configuration_file = "tests/configurations/filestore.yaml"
        JupyterTester(task_specification, ipynb_filename).main(
            constants,
            configuration_file,
            output_directory,
        )


# ----------------------------------------------------------------------------------------
class TestJupyterDirectBad:
    def test(self, constants, logging_setup, output_directory):
        """
        Tests that a jupyter task can be run.
        """

        ipynb_filename = f"{os.getcwd()}/tests/jupyter/jupyter_bad_cell.ipynb"

        task_specification = {
            "type": "dls_bxflow_run.bx_tasks.jupyter",
            "label": "my_jupyter",
            RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
            "type_specific_tbd": {"ipynb_filename": ipynb_filename},
        }

        configuration_file = "tests/configurations/filestore.yaml"
        JupyterTester(
            task_specification,
            ipynb_filename,
            expected_error="name 'something_bad' is not defined",
        ).main(
            constants,
            configuration_file,
            output_directory,
        )


# ----------------------------------------------------------------------------------------
class TestJupyterShellGood:
    def test(self, constants, logging_setup, output_directory):
        """
        Tests that a jupyter task can be run.
        """

        ipynb_filename = f"{os.getcwd()}/tests/jupyter/jupyter1.ipynb"

        task_specification = {
            "type": "dls_bxflow_run.bx_tasks.jupyter",
            "label": "my_jupyter",
            RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
            "type_specific_tbd": {
                "ipynb_filename": ipynb_filename,
                "command": f"{os.path.dirname(__file__)}/jupyter/execute.sh",
            },
        }

        configuration_file = "tests/configurations/filestore.yaml"
        JupyterTester(task_specification, ipynb_filename).main(
            constants,
            configuration_file,
            output_directory,
        )


# ----------------------------------------------------------------------------------------
class TestJupyterShellBad:
    def test(self, constants, logging_setup, output_directory):
        """
        Tests that a jupyter task can be run.
        """

        ipynb_filename = f"{os.getcwd()}/tests/jupyter/jupyter_bad_cell.ipynb"

        task_specification = {
            "type": "dls_bxflow_run.bx_tasks.jupyter",
            "label": "my_jupyter",
            RemexKeywords.HINTS: {RemexKeywords.CLUSTER: RemexClusters.LOCAL},
            "type_specific_tbd": {
                "ipynb_filename": ipynb_filename,
                "command": f"{os.path.dirname(__file__)}/jupyter/execute.sh",
            },
        }

        configuration_file = "tests/configurations/filestore.yaml"
        JupyterTester(
            task_specification,
            ipynb_filename,
            expected_error="error executing jupyter notebook",
        ).main(
            constants,
            configuration_file,
            output_directory,
        )


# ----------------------------------------------------------------------------------------
class JupyterTester(BaseContextTester):
    """
    Test the jupyter task.
    """

    def __init__(
        self,
        task_specification,
        ipynb_filename,
        expected_error=None,
    ):
        self.__task_specification = task_specification
        self.__ipynb_filename = ipynb_filename
        self.__expected_error = expected_error

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

                if self.__expected_error is None:
                    return_code = await bx_task.run()
                    assert return_code == 0
                else:
                    with pytest.raises(Exception) as raised_exception:
                        return_code = await bx_task.run()
                    assert self.__expected_error in str(raised_exception.value)

                # Where was supposedly saved the executed notebook.
                sanitized_label = sanitize(bx_task.label())
                cwd_ipynb_filename = f"{output_directory}/{sanitized_label}.ipynb"

                html_filename = os.path.splitext(cwd_ipynb_filename)[0]
                html_filename = f"{html_filename}.html"

                assert os.path.exists(cwd_ipynb_filename)
                assert os.path.exists(html_filename)

            finally:
                os.chdir(original_directory)
