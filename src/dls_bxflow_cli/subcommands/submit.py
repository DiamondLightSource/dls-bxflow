import argparse
import asyncio
import json

# Use standard logging in this module.
import logging
import sys

# Utilities.
from dls_utilpack.require import require

# Base class for cli subcommands.
from dls_bxflow_cli.subcommands.base import Base
from dls_bxflow_lib.bx_guis.bx_guis import bx_guis_get_default

# BxGui protocolj things (must agree with javascript).
from dls_bxflow_lib.bx_guis.constants import Commands, Keywords
from dls_bxflow_lib.bx_guis.context import Context as BxGuiContext

# Settings manager.
from dls_bxflow_lib.bx_settings.bx_settings import BxSettings

logger = logging.getLogger()


# --------------------------------------------------------------
class Submit(Base):
    """
    Start one or more services and keep them running until ^C.
    """

    def __init__(self, args, mainiac):
        super().__init__(args)

        self.__mainiac = mainiac

    # ----------------------------------------------------------------------------------------
    def run(self):
        """ """

        # Run in asyncio event loop.
        asyncio.run(self.__run_coro())

    # ----------------------------------------------------------
    async def __run_coro(self):
        """"""

        # Load the configuration.
        bx_configurator = self.get_bx_configurator()
        await bx_configurator.load()

        specification = bx_configurator.require("bx_gui_specification")
        bx_gui_context = BxGuiContext(specification)

        async with bx_gui_context:
            bx_gui = bx_guis_get_default()

            request = {
                Keywords.COMMAND: Commands.GET_WORKFLOW_CONSTRUCTOR_KWARGS,
                "workflow_filename_classname": self._args.workflow_filename_classname,
            }

            logger.debug(
                "constructor kwargs request\n%s" % (json.dumps(request, indent=4))
            )

            bx_gui = bx_guis_get_default()
            response = await bx_gui.client_protocolj(request)

            logger.debug(
                "constructor kwargs response\n%s" % (json.dumps(response, indent=4))
            )

            # Make parser args out of constructor kwargs.
            settings = BxSettings("workflow settings")
            settings.load_from_dicts(
                require("server response payload", response, Keywords.PAYLOAD)
            )
            parser = argparse.ArgumentParser()
            settings.add_to_argument_parser(parser)

            # Reparse the command line.
            args, remainder = parser.parse_known_args(sys.argv)
            logger.debug(f"sys.argv for reparser is {str(sys.argv)}")
            logger.debug(f"args from reparser is {str(args)}")

            # Make a dict out of the args.
            args_dict = vars(args)

            # Request to run the workflow.
            request = {
                Keywords.COMMAND: Commands.START_WORKFLOW_NOCOOKIE,
                "workflow_filename_classname": self._args.workflow_filename_classname,
                Keywords.PAYLOAD: args_dict,
            }

            logger.debug("submission request\n%s" % (json.dumps(request, indent=4)))

            response = await bx_gui.client_protocolj(request)

            logger.debug("submission response\n%s" % (json.dumps(response, indent=4)))

            # logger.info("job submitted, waiting for it to finish...")

            # logger.info(
            #     "    ...you can hit ^C to stop waiting but workflow will still continue)"
            # )

    # ----------------------------------------------------------
    def add_arguments(parser):

        parser.add_argument(
            help="Workflow.",
            type=str,
            metavar="workflow filename::classname",
            dest="workflow_filename_classname",
        )

        parser.add_argument(
            "remainder",
            nargs=argparse.REMAINDER,
            help="Remaining args passed to the workflow.",
            metavar="args",
        )

        return parser
