import json
import logging
import os
import shutil

import h5py

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Base class for a dawn task.
from dls_bxflow_run.bx_tasks.dawn_base import DawnBase

# States.
from dls_bxflow_run.bx_tasks.states import States

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_run.bx_tasks.dawn1"


class Dawn1(DawnBase):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        DawnBase.__init__(
            self, thing_type, specification, predefined_uuid=predefined_uuid
        )

        self.state(States.PREPARED)

    # ----------------------------------------------------------------------------------------
    async def run(self):
        """
        Dummy will read infile_variable and write outfile_variable if given.
        """

        type_specific_tbd = self.specification().get("type_specific_tbd", {})

        dawn_pipeline = require(
            f"{callsign(self)} configuration",
            type_specific_tbd,
            "dawn_pipeline",
        )
        data_filename = require(
            f"{callsign(self)} configuration",
            type_specific_tbd,
            "data_filename",
        )
        dataset_path = require(
            f"{callsign(self)} configuration",
            type_specific_tbd,
            "dataset_path",
        )

        self.assert_dataset(dawn_pipeline, "/entry/process/0/data")
        self.assert_dataset(data_filename, dataset_path)

        # Make a copy of the pipeline.
        dawn_pipeline_copy = f"{self.get_directory()}/{os.path.basename(dawn_pipeline)}"
        shutil.copyfile(dawn_pipeline, dawn_pipeline_copy)

        json_filename = self.create_dawn_json_file(
            dawn_pipeline_copy, data_filename, dataset_path
        )

        logger.info(f"[DAWNJP] created json_filename {json_filename}")

        exit_code = self.launch_and_wait(json_filename)

        return exit_code

    # ----------------------------------------------------------------------------------------
    def assert_dataset(self, data_filename, dataset_path):
        dirname = os.path.dirname(data_filename)
        if not os.path.exists(dirname):
            raise AssertionError(f"cannot find directory {dirname}")
        if not os.path.isdir(dirname):
            raise AssertionError(f"cannot open directory {dirname}")
        if not os.path.exists(data_filename):
            raise AssertionError(f"cannot find file {data_filename}")

        h5file = None
        try:
            try:
                h5file = h5py.File(data_filename, "r")
            except Exception as exception:
                raise AssertionError(f"cannot open {data_filename}: {str(exception)}")
            if dataset_path not in h5file:
                raise AssertionError(
                    f"{data_filename} does not contain dataset {dataset_path}"
                )
        finally:
            if h5file is not None:
                h5file.close()

    # ----------------------------------------------------------------------------------------
    def create_dawn_json_file(self, dawn_pipeline, data_filename, dataset_path):
        """Create DAWN json config file.

        {
            "runDirectory": "/dls/i22/data/2022/cm31149-1",
            "name": "TestRun",
            "filePath": "path_to_file",
            "dataDimensions": [
                -1,
                -2
            ],
            "processingPath": "/dls/i22/data/2022/cm31149-1/xml/templates/gisaxs_Transmission_Pipeline.nxs",
            "outputFilePath": "path_to_output",
            "deleteProcessingFile": false,
            "datasetPath": "/entry1/detector",
            "numberOfCores": 1,
            "xmx": 1024
        }
        """

        output_filename = f"{self.get_directory()}/{self.label()}.nxs"
        contents = {
            "runDirectory": self.get_directory(),
            "name": self.label(),
            "filePath": data_filename,
            "dataDimensions": [-1, -2],
            "processingPath": dawn_pipeline,
            "outputFilePath": output_filename,
            "deleteProcessingFile": False,
            "datasetPath": dataset_path,
            "numberOfCores": 1,
            "xmx": 1024,
        }

        # The output filename will be a candidate for a catalog attachment.
        self.propose_artefact(output_filename)

        json_filename = f"{self.get_directory()}/{self.label()}.json"

        logger.info(f"[DAWNJP] creating json_filename {json_filename}")

        with open(json_filename, "wt") as stream:
            json.dump(contents, stream, indent=4)

        return json_filename
