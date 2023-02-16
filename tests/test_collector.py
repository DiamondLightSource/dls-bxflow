import asyncio
import logging
import os
import time

import pytest

# Object managing datafaces.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Object managing collectors.
from dls_bxflow_lib.bx_collectors.bx_collectors import bx_collectors_get_default

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts
from dls_bxflow_lib.bx_jobs.states import States as BxJobStates

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestCollectorManualDirect:
    def test(self, constants, logging_setup, output_directory):
        """
        Test manual fire not using the http server.
        """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"
        # The collector configuration to replace in the configuration file for this test.
        actual_bx_collector_specification = "bx_collector_specification_manual"

        CollectorTesterDirect(
            actual_bx_collector_specification=actual_bx_collector_specification,
        ).main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class TestCollectorManualServer:
    def test(self, constants, logging_setup, output_directory):
        """
        Test manual fire via the http server.
        """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"
        # The collector configuration to replace in the configuration file for this test.
        actual_bx_collector_specification = "bx_collector_specification_manual"

        CollectorTesterServer(
            actual_bx_collector_specification=actual_bx_collector_specification,
        ).main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
@pytest.mark.skipif("not config.getoption('activemq')")
class TestCollectorGdascanServer:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"
        # The collector configuration to replace in the configuration file for this test.
        actual_bx_collector_specification = "bx_collector_specification_gdascan"

        os.environ[
            "BXFLOW_GDA_PARSER"
        ] = f"{os.path.dirname(__file__)}/gda_parser.py::GdaParser"

        CollectorTesterServer(
            actual_bx_collector_specification=actual_bx_collector_specification,
        ).main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class TestCollectorBad1:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"
        # The collector configuration to replace in the configuration file for this test.
        actual_bx_collector_specification = "bx_collector_specification_manual"

        bad_constructor_kwargs = {"outfile": "workflow_a_outfile.txt", "some": "thing"}

        CollectorTesterServer(
            actual_bx_collector_specification=actual_bx_collector_specification,
            workflow_constructor_kwargs=bad_constructor_kwargs,
            expect_trigger_exception="not a known workflow constructor argument",
        ).main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class CollectorTesterDirect(BaseContextTester):
    """
    Class to test the collector.
    """

    def __init__(self, actual_bx_collector_specification=None, direct=False):
        BaseContextTester.__init__(self)

        self.__actual_bx_collector_specification = actual_bx_collector_specification
        self.__direct = direct

    async def _main_coroutine(self, constants, output_directory):
        """ """

        # Make it so the collector can find the workflow class.
        os.environ["PYTHONPATH"] = "%s:%s" % (
            os.path.dirname(__file__),
            os.environ.get("PYTHONPATH", ""),
        )
        os.environ["OUTPUT_DIRECTORY"] = output_directory

        bx_configurator = self.get_bx_configurator()

        # Don't build high level things.
        bx_configurator.remove("bx_launcher_specifications")
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()

        actual_bx_collector_specification = context_configuration[
            self.__actual_bx_collector_specification
        ]

        # Replace the collector actual with the one we want for this test.
        context_configuration[
            "bx_collector_specification"
        ] = actual_bx_collector_specification

        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:
            workflow_filename_classname = (
                f"{os.path.dirname(__file__)}/workflows/a/workflow.py::A"
            )

            workflow_constructor_kwargs = {"outfile": None}

            # Trigger the workflow.
            await bx_collectors_get_default().fire(
                {
                    "workflow_filename_classname": workflow_filename_classname,
                    "workflow_constructor_kwargs": workflow_constructor_kwargs,
                }
            )

            # Get all jobs.
            records = await bx_datafaces_get_default().get_bx_jobs()

            # Make sure we have one job in ready state.
            assert len(records) == 1
            assert records[0]["state"] == BxJobStates.READY


# ----------------------------------------------------------------------------------------
class CollectorTesterServer(BaseContextTester):
    """
    Class to test the collector.
    """

    def __init__(
        self,
        actual_bx_collector_specification=None,
        workflow_constructor_kwargs=None,
        expect_trigger_exception=None,
    ):
        BaseContextTester.__init__(self)

        self.__actual_bx_collector_specification = actual_bx_collector_specification
        self.__workflow_constructor_kwargs = workflow_constructor_kwargs
        self.__expect_trigger_exception = expect_trigger_exception

    async def _main_coroutine(self, constants, output_directory):
        """ """

        # Make it so the collector can find the workflow class.
        os.environ["PYTHONPATH"] = "%s:%s" % (
            os.path.dirname(__file__),
            os.environ.get("PYTHONPATH", ""),
        )
        os.environ["OUTPUT_DIRECTORY"] = output_directory

        # The file the task will write.
        outfile = f"{output_directory}/workflow_a_outfile.txt"

        # Tester may provide badly-constructed workflow constructor.
        if self.__workflow_constructor_kwargs is None:
            self.__workflow_constructor_kwargs = {"outfile": outfile}

        bx_configurator = self.get_bx_configurator()

        # Don't build high level things.
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()

        actual_bx_collector_specification = context_configuration[
            self.__actual_bx_collector_specification
        ]

        bx_collector_specification = context_configuration["bx_collector_specification"]
        bx_collector_specification["type_specific_tbd"][
            "actual_bx_collector_specification"
        ] = actual_bx_collector_specification

        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:
            workflow_filename_classname = (
                f"{os.path.dirname(__file__)}/workflows/a/workflow.py::A"
            )
            workflow_constructor_kwargs = self.__workflow_constructor_kwargs

            trigger_exception = None
            try:
                # Trigger the workflow.
                await bx_collectors_get_default().fire(
                    {
                        "workflow_filename_classname": workflow_filename_classname,
                        "workflow_constructor_kwargs": workflow_constructor_kwargs,
                    }
                )
            except Exception as exception:
                trigger_exception = exception

            if self.__expect_trigger_exception is not None:
                assert trigger_exception is not None
                assert self.__expect_trigger_exception in str(trigger_exception)
                return
            elif trigger_exception is not None:
                raise trigger_exception

            time0 = time.time()
            timeout = 5.0
            while True:
                # Get all jobs.
                records = await bx_datafaces_get_default().get_bx_jobs()

                if len(records) > 0:
                    break

                if time.time() - time0 > timeout:
                    raise RuntimeError(f"job not registered within {timeout} seconds")
                await asyncio.sleep(1.0)

            time0 = time.time()
            timeout = 5.0
            while True:
                # Get all jobs.
                records = await bx_datafaces_get_default().get_bx_jobs()

                # Make sure we have one job.
                assert len(records) == 1

                logger.debug("[DMOTF] job state is %s" % (records[0]["state"]))

                # Stop waiting when job succeeds or fails.
                if records[0]["state"] in [BxJobStates.SUCCEEDED, BxJobStates.FAILED]:
                    break

                if time.time() - time0 > timeout:
                    raise RuntimeError(f"job not finished within {timeout} seconds")
                await asyncio.sleep(1.0)

        # Make sure the task wrote its output file.
        assert os.path.exists(outfile)
