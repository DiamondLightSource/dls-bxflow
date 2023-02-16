import logging

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.isodatetime import isodatetime_filename

# Exceptions.
from dls_bxflow_api.exceptions import NotSet

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Filestore manager.
from dls_bxflow_lib.bx_filestores.bx_filestores import bx_filestores_get_default

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestFilestoreExplicit:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/filestore.yaml"
        FilestoreTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class TestFilestoreDynamic:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/filestore.yaml"

        bx_filestore_specification = "bx_filestore_specification_dynamic"
        FilestoreTester(bx_filestore_specification=bx_filestore_specification).main(
            constants, configuration_file, output_directory
        )


# ----------------------------------------------------------------------------------------
class TestFilestoreScandir:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/filestore.yaml"

        bx_filestore_specification = "bx_filestore_specification_scandir"
        FilestoreTester(bx_filestore_specification=bx_filestore_specification).main(
            constants, configuration_file, output_directory
        )


# ----------------------------------------------------------------------------------------
class Capsule:
    def __init__(self, label, uuid):
        self.__label = label
        self.__uuid = uuid
        self.__directory = None

    # -----------------------------------------------------------------------------
    def set_directory(self, directory):
        self.__directory = directory

    def get_directory(self):
        if self.__directory is None:
            raise NotSet(f"{callsign(self)} directory has not been set")
        return self.__directory

    def label(self):
        return self.__label

    def uuid(self):
        return self.__uuid


# ----------------------------------------------------------------------------------------
class FilestoreTester(BaseContextTester):
    def __init__(self, bx_filestore_specification=None):
        self.__bx_filestore_specification = bx_filestore_specification

    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()
        specification = await bx_configurator.load()

        if self.__bx_filestore_specification is not None:
            specification["bx_filestore_specification"] = specification[
                self.__bx_filestore_specification
            ]

        bx_context = BxContexts().build_object(specification)

        async with bx_context:
            # Get a filestore object based on the context configuration.
            filestore = bx_filestores_get_default()

            # Our filestore needs to be explicitly assigned a directory?
            if hasattr(filestore, "set_directory"):
                filestore.set_directory(output_directory)

            assert filestore.get_beamline() == "b29"
            assert filestore.get_visit() == "cm00001-1"

            # --------------------------------------------------------------------
            bx_job = Capsule("my job", "000001-2222-3333-4444-55555555")
            bx_task = Capsule("my task/part1", "000002-2222-3333-4444-55555555")

            filestore.pin_job_directory(bx_job)
            filestore.pin_task_directory(bx_job, bx_task)
            runtime_directory = bx_task.get_directory()

            logger.info("runtime_directory %s" % (runtime_directory))

            # Directory should have the date in it.
            ymd = isodatetime_filename().split(".")[0]

            assert "my-job" in runtime_directory
            assert ymd in runtime_directory
            assert "2222" not in runtime_directory
            assert "my-task-part1" in runtime_directory
