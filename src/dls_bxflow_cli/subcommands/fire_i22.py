import asyncio

# Use standard logging in this module.
import logging
import os

# Base class for cli subcommands.
from dls_bxflow_cli.subcommands.base import Base

# Collector.
from dls_bxflow_lib.bx_collectors.bx_collectors import bx_collectors_get_default

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

logger = logging.getLogger()


# --------------------------------------------------------------
class FireI22(Base):
    """
    Fire a gdascan collector.
    """

    def __init__(self, args):
        super().__init__(args)

    # ----------------------------------------------------------------------------------------
    def run(self):
        """ """

        # Run in asyncio event loop.
        asyncio.run(self.__run_coro())

    # ----------------------------------------------------------
    async def __run_coro(self):
        """"""

        # Load the configuration per the rules of the cli.
        bx_configurator = self.get_bx_configurator()
        context_configuration = await bx_configurator.load()

        # Make a context from the configuration.
        bx_context = BxContexts().build_object(context_configuration)

        # Open the context (servers and clients).
        async with bx_context:

            bx_collector = bx_collectors_get_default()

            data_filename = self._args.data_filename
            visit_directory = os.path.dirname(data_filename)

            # This is an I22-specific gda message.
            message = {
                "status": "FINISHED",
                "filePath": data_filename,
                "visitDirectory": visit_directory,
                "swmrStatus": "ACTIVE",
                "scanNumber": 0,
                "scanDimensions": [45, 61],
                "scannables": [],
                "detectors": ["BL22I-ML-SCAN-01"],
                "percentageComplete": 100.0,
                "processingRequest": {},
            }

            # Fire the message to the collector.
            await bx_collector.fire(message)

    # ----------------------------------------------------------
    def add_arguments(parser):

        parser.add_argument(
            help="data file to process",
            type=str,
            metavar="filename",
            dest="data_filename",
        )

        return parser
