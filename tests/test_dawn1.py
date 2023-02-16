import copy
import json
import logging
import os

import h5py
import pytest

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

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestDawn1:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/filestore.yaml"
        Dawn1Tester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class Dawn1Tester(BaseContextTester):

    # ----------------------------------------------------------------------------------------
    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()
        specification = await bx_configurator.load()
        bx_context = BxContexts().build_object(specification)

        async with bx_context:
            # Build a job object, needed to locate the output files.
            bx_job_specification = copy.deepcopy(bx_jobs_get_default().specification())
            bx_job_specification["label"] = "my/good/job"
            bx_job = BxJobs().build_object(
                bx_job_specification,
            )
            bx_filestores_get_default().pin_job_directory(bx_job)

            dawn_pipeline = f"{output_directory}/MyPhysicsPipeline.nxs"
            data_filename = f"{output_directory}/visit/i22-4996.nxs"
            dataset_path = "/entry1/instrument/Rapid2D/data"

            original_directory = os.getcwd()

            # Make the fake dawn findable in the path.
            os.environ["PATH"] = "%s:%s" % (
                os.path.dirname(__file__),
                os.environ["PATH"],
            )

            try:
                dawn_executable = f"{os.path.dirname(__file__)}/stub_commands/dawn"
                os.chdir(output_directory)
                os.makedirs(".bxflow")

                # Define the class and its constructor arguments.
                bx_task = BxTasks().build_object(
                    {
                        "type": "dls_bxflow_run.bx_tasks.dawn1",
                        "label": "my_dawn",
                        "remex_hints": {
                            RemexKeywords.CLUSTER: RemexClusters.LOCAL,
                        },
                        "type_specific_tbd": {
                            "dawn_pipeline": dawn_pipeline,
                            "data_filename": data_filename,
                            "dataset_path": dataset_path,
                            "dawn_executable": dawn_executable,
                        },
                    }
                )

                bx_task.bx_job_uuid(bx_job.uuid())

                bx_filestores_get_default().pin_task_directory(bx_job, bx_task)

                os.makedirs(bx_task.get_directory())

                # --------------------------------------------------------------
                # Expect: cannot find file.
                with pytest.raises(AssertionError) as asserted_exception:
                    await bx_task.run()
                logger.debug(asserted_exception.value)

                # --------------------------------------------------------------
                # Expect: cannot open hdf5 file.
                with open(dawn_pipeline, "w") as stream:
                    pass

                with pytest.raises(AssertionError) as asserted_exception:
                    await bx_task.run()
                logger.debug(asserted_exception.value)

                # --------------------------------------------------------------
                # Expect: cannot find dataset.
                with h5py.File(dawn_pipeline, "w") as h5file:
                    pass

                with pytest.raises(AssertionError) as asserted_exception:
                    await bx_task.run()
                logger.debug(asserted_exception.value)

                # --------------------------------------------------------------
                # Expect: cannot find directory.
                with h5py.File(dawn_pipeline, "w") as h5file:
                    h5file.create_dataset("/entry/process/0/data", (1))

                with pytest.raises(AssertionError) as asserted_exception:
                    await bx_task.run()
                logger.debug(asserted_exception.value)

                # --------------------------------------------------------------
                # Expect: cannot find fild.
                os.makedirs(os.path.dirname(data_filename))

                with pytest.raises(AssertionError) as asserted_exception:
                    await bx_task.run()
                logger.debug(asserted_exception.value)

                # --------------------------------------------------------------
                # Now we can finally run.
                with h5py.File(data_filename, "w") as h5file:
                    h5file.create_dataset(dataset_path, (1))

                await bx_task.run()

                return

                with open(f"{output_directory}/dawn_configuration.json") as stream:
                    dawn_configuration = json.load(stream)

                assert dawn_configuration["runDirectory"] == output_directory
                assert dawn_configuration["runDirectory"] == output_directory

                # Check the artefacts list is as expected.
                with open(f"{output_directory}/.bxflow/artefacts.txt") as stream:
                    for line in stream.readlines():
                        assert line.strip().endswith("dawn_output_data.nxs")

            finally:
                os.chdir(original_directory)
