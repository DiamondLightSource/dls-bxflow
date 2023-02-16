import asyncio
import json
import logging
import multiprocessing
import os

import pytest

# Remex (remote execution) API.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

logger = logging.getLogger(__name__)


class TestDawn2:

    # ----------------------------------------------------------------------------------------
    def test(
        self,
        constants,
        logging_setup,
        output_directory,
    ):
        """ """

        multiprocessing.current_process().name = "main"

        failure_message = None
        try:
            # Run main in asyncio event loop.
            asyncio.run(self._main_coroutine(constants, output_directory))

        except Exception as exception:
            logger.exception(
                "unexpected exception in the test method", exc_info=exception
            )
            failure_message = str(exception)

        if failure_message is not None:
            pytest.fail(failure_message)

    # ----------------------------------------------------------------------------------------
    async def _main_coroutine(self, constants, output_directory):
        """ """

        template = {
            "runDirectory": "some/run/directory",
            "name": "TestRun",
            "filePath": "path_to_file",
            "dataDimensions": [-1, -2],
            "processingPath": "some/dawn/pipeline.nxs",
            "outputFilePath": "path_to_output",
            "deleteProcessingFile": False,
            "datasetPath": "/entry1/detector",
            "numberOfCores": 1,
            "xmx": 1024,
        }

        tname = "XrayPipeline"

        # Make up a template for the DAWN pipeline.
        # These templates are typically produced, for example,
        # by Tim Snow's DLS_BLI22_AutoProcessingGUI tool.
        # They are often saved in visit_directory/xml/templates.
        template_filename = f"{output_directory}/{tname}.json"
        with open(template_filename, "w") as stream:
            json.dump(template, stream)

        # Point to the mocked dawn script in the tests directory.
        dawn_executable = f"{os.path.dirname(__file__)}/stub_commands/dawn"

        # Make the fake dawn executable findable in the path.
        os.environ["PATH"] = "%s:%s" % (os.path.dirname(__file__), os.environ["PATH"])

        original_directory = os.getcwd()

        try:
            os.chdir(output_directory)
            os.makedirs(".bxflow")

            # Define the class and its constructor arguments.
            data_filename = f"{output_directory}/data_file.nxs"
            bx_task = BxTasks().build_object(
                {
                    "type": "dls_bxflow_run.bx_tasks.dawn2",
                    "label": "my_dawn",
                    "remex_hints": {
                        RemexKeywords.CLUSTER: RemexClusters.LOCAL,
                    },
                    "type_specific_tbd": {
                        "template_filename": template_filename,
                        "data_filename": data_filename,
                        "dawn_executable": dawn_executable,
                    },
                }
            )

            # Run the task.
            await bx_task.run()

            # Check that the task wrote a DAWN json file
            # with the correct input and output filenames.
            with open(f"{tname}.json", "r") as stream:
                template = json.load(stream)
                assert template["filePath"] == data_filename

                # The dawn2 task will choose the name for the output file
                # based on the name of the template it was given.
                assert template["outputFilePath"] == f"{output_directory}/{tname}.nxs"

            # Check the artefacts list is as expected.
            with open(f"{output_directory}/.bxflow/artefacts.txt") as stream:
                for line in stream.readlines():
                    assert line.strip().endswith(f"{tname}.nxs")
        finally:
            os.chdir(original_directory)

        # TODO: Enhance DAWN task testing by checking some ouptut was produced.
