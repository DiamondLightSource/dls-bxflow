import asyncio
import logging
import os
import time

# Object managing datafaces.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestCollectorScraper:
    def test(self, constants, logging_setup, output_directory):
        """
        Test scraper collector's ability to automatically discover files.
        """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"
        # The collector configuration to replace in the configuration file for this test.
        actual_collspec = "bx_collector_specification_scraper"

        CollectorScraperTester(
            actual_collspec=actual_collspec,
        ).main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class CollectorScraperTester(BaseContextTester):
    """
    Class to test the collector.
    """

    def __init__(
        self,
        actual_collspec=None,
    ):
        BaseContextTester.__init__(self)

        self.__actual_collspec = actual_collspec

    async def _main_coroutine(self, constants, output_directory):
        """ """

        # Make it so the collector can find the workflow class.
        os.environ["PYTHONPATH"] = "%s:%s" % (
            os.path.dirname(__file__),
            os.environ.get("PYTHONPATH", ""),
        )
        os.environ["OUTPUT_DIRECTORY"] = output_directory

        # Write some scrape-able files.
        scan_directory = f"{output_directory}/scan_directory"
        os.makedirs(scan_directory)

        bx_configurator = self.get_bx_configurator()

        # Don't build high level things.
        bx_configurator.remove("bx_launcher_specifications")
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()

        # Refer to the desired actual collector specification.
        actual_collspec = context_configuration[self.__actual_collspec]

        actual_collspec_tbd = actual_collspec["type_specific_tbd"]

        actual_collspec_tbd["scrape_glob"] = f"{scan_directory}/*.nxs"
        workflow = f"{os.path.dirname(__file__)}/workflows/b/workflow.py::B"
        actual_collspec_tbd["workflow_filename_classname"] = workflow
        actual_collspec_tbd["workflow_constructor_kwargs"] = {}

        server_collspec = context_configuration["bx_collector_specification"]
        server_collspec["type_specific_tbd"][
            "actual_bx_collector_specification"
        ] = actual_collspec

        bx_context = BxContexts().build_object(context_configuration)

        job_count = 2

        # Start the servers.
        async with bx_context:
            # Wait long enough for the collector to activate and start ticking.
            await asyncio.sleep(2.0)

            # Get all jobs before we create any of the scrape-able files.
            records = await bx_datafaces_get_default().get_bx_jobs()

            if len(records) != 0:
                raise RuntimeError(f"found {len(records)} jobs but expected 0")

            # Create a few scrape-able files.
            # These will be picked up by the scraper's tick coroutine.
            for i in range(10000, 10000 + job_count):
                filename = f"{scan_directory}/b29-{i}.nxs"
                with open(filename, "w") as stream:
                    stream.write("")

            # Wait for all the jobs to appear.
            time0 = time.time()
            timeout = 5.0
            while True:

                # Get all jobs.
                records = await bx_datafaces_get_default().get_bx_jobs()

                if len(records) >= job_count:
                    break

                if time.time() - time0 > timeout:
                    raise RuntimeError(f"job not registered within {timeout} seconds")
                await asyncio.sleep(1.0)

            # Wait a couple more seconds to make sure there are no extra jobs appearing.
            await asyncio.sleep(2.0)
            # Get all jobs.
            records = await bx_datafaces_get_default().get_bx_jobs()

            if len(records) != job_count:
                raise RuntimeError(
                    f"found {len(records)} jobs but expected {job_count}"
                )

        logger.debug("------------ restarting collector --------------------")
        # Start the servers again.
        # This covers the case where collector starts by finding existing entries in the database.
        async with bx_context:
            await asyncio.sleep(2.0)
            # Get all jobs after servers start up and run briefly.
            records = await bx_datafaces_get_default().get_bx_jobs()

            if len(records) != job_count:
                raise RuntimeError(
                    f"found {len(records)} jobs but expected {job_count}"
                )
