import asyncio

# Use standard logging in this module.
import logging

# Utilities.
from dls_utilpack.require import require

# Base class for cli subcommands.
from dls_bxflow_cli.subcommands.base import Base

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts
from dls_bxflow_lib.bx_guis.bx_guis import bx_guis_get_default

logger = logging.getLogger()

# Specifications of services we can start, and their short names for parse args.
services = {
    "bx_news_specification": "news",
    "dls_servbase_dataface_specification": "dls_servbase_dataface",
    "bx_dataface_specification": "dataface",
    "bx_catalog_specification": "catalog",
    "bx_launcher_specifications": "launchers",
    "bx_scheduler_specification": "scheduler",
    "bx_collector_specification": "collector",
    "bx_gui_specification": "gui",
}


# --------------------------------------------------------------
class StartServices(Base):
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
        bx_configurator = self.get_bx_configurator(args_dict=vars(self._args))

        # Let the configurator know about any mpqueue logging.
        # bx_configurator.set_logging_mpqueue(self.__mainiac.mpqueue)

        context_configuration = await bx_configurator.load()

        if len(self._args.service_names) == 0:
            self._args.service_names = ["all"]

        if "all" in self._args.service_names:
            selected_service_names = []
            for _, service_name in services.items():
                selected_service_names.append(service_name)
        else:
            selected_service_names = self._args.service_names

        # Change all start_as to None, except the one we are starting.
        for keyword, specification in context_configuration.items():
            if keyword in services:
                service_name = services[keyword]
                if service_name in selected_service_names:
                    # Special case where specification is a list of other specifications (e.g. launchers).
                    if isinstance(specification, list):
                        for sub_keyword in specification:
                            sub_specification = require(
                                "service specification",
                                context_configuration,
                                sub_keyword,
                            )
                            sub_specification["context"] = {"start_as": "process"}
                    else:
                        specification["context"] = {"start_as": "process"}

        # Make a context from the configuration.
        bx_context = BxContexts().build_object(context_configuration)

        # Open the context (servers and clients).
        async with bx_context:
            if "gui" in selected_service_names:
                logger.info(
                    f"starting web gui, please browse to {bx_guis_get_default().client_url()}/index.html"
                )

            try:
                # Stay up until all processes are dead.
                # TODO: Use asyncio wait or sentinel for all started processes to be dead.
                while True:
                    await asyncio.sleep(1.0)
                    if not await bx_context.is_any_process_alive():
                        logger.info("all processes have shutdown")
                        break
            except KeyboardInterrupt:
                pass

    # ----------------------------------------------------------
    def add_arguments(parser):

        services_list = list(services.values())

        parser.add_argument(
            help='"all" or any combination of {%s}' % (" ".join(services_list)),
            nargs="*",
            type=str,
            metavar="service name(s)",
            dest="service_names",
            default=[],
        )

        parser.add_argument(
            "--visit",
            help="Visit.",
            type=str,
            metavar="visit name",
            default="VISIT",
            dest="visit",
        )

        # TODO: Centralize definition of configuration_keyword symbol.
        configuration_keyword = "configuration"
        parser.add_argument(
            "--configuration",
            "-c",
            help="Configuration file.",
            type=str,
            metavar="yaml filename",
            default=None,
            dest=configuration_keyword,
        )

        return parser
