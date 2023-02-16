import asyncio
import importlib

# Use standard logging in this module.
import logging

# Object managers we interact with.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Base class for cli subcommands.
from dls_bxflow_cli.subcommands.base import Base

# Results composers.
from dls_bxflow_lib.bx_composers.bx_composers import BxComposers

# Configurator.
from dls_bxflow_lib.bx_configurators.bx_configurators import BxConfigurators

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

logger = logging.getLogger()


# --------------------------------------------------------------
class ExecuteWorkflow(Base):
    """
    Check a previously recorded file.
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

        # Load the configuration.
        bx_configurator = self.get_bx_configurator(
            self._args.configuration_file, self._args.output_directory
        )
        context_configuration = await bx_configurator.load()

        # Make a context from the configuration.
        bx_context = BxContexts().build_object(context_configuration)

        # Open the context (servers).
        async with bx_context:
            workload_module = importlib.import_module(self._args.workload_module)
            workload_class = getattr(workload_module, self._args.workload_class)
            workload_object = workload_class()

            argv = ["--a", "1"]
            logger.info("position A")
            workload_object.parse_argv(argv)
            logger.info("position B")

            variables, bx_job = workload_object.build()

            # Register all the variables that have been added.
            await variables.register(bx_job.uuid())

            # Schedule the bx_job to run.
            await bx_job.enable()

            logger.info("job submitted... waiting for it to finish")

            try:
                # Wait for bx_job to finish.
                await bx_job.wait(timeout=None, naptime=1.0)
            except KeyboardInterrupt:
                pass

            # Summarize the job's execution.
            job_summary_text = await self._compose_job_summary(bx_job.uuid())

        # -------------------------------------------------------------
        # Context is now closed, database and all other services are unavailable.

        logger.info(f"job summary\n{job_summary_text}")

    # ----------------------------------------------------------
    def add_arguments(parser):

        parser.add_argument(
            help="configuration file",
            type=str,
            metavar="yaml filename",
            dest="configuration_file",
        )

        parser.add_argument(
            help="workload module",
            type=str,
            metavar="python module (with dots)",
            dest="workload_module",
        )

        parser.add_argument(
            help="workload class",
            type=str,
            metavar="python class name (usually CamelCase)",
            dest="workload_class",
        )

        parser.add_argument(
            help="output directory",
            type=str,
            metavar="filename",
            dest="output_directory",
        )

        return parser

    # ----------------------------------------------------------------------------------------
    async def get_bx_configurator(self, configuration_file, output_directory):

        bx_configurator = BxConfigurators().build_object(
            {
                "type": "dls_bxflow_lib.bx_configurators.yaml",
                "type_specific_tbd": {"filename": configuration_file},
            }
        )

        # For convenience, always do this replacement.
        bx_configurator.substitute({"output_directory": output_directory})

        return bx_configurator

    # ----------------------------------------------------------------------------------------
    async def _compose_job_summary(
        self,
        bx_job_uuid,
    ):

        bx_composer = BxComposers().build_object(
            {"type": "dls_bxflow_lib.bx_composers.text"}
        )
        bx_news_records = await bx_datafaces_get_default().get_bx_news(
            bx_job_uuid=bx_job_uuid
        )
        news_text = bx_composer.compose_bx_news(bx_news_records)

        bx_job_record = await bx_datafaces_get_default().get_bx_job(bx_job_uuid)

        bx_jobs_bx_tasks_bx_gates_records = (
            await bx_datafaces_get_default().get_bx_jobs_bx_tasks_bx_gates(
                bx_job_uuid=bx_job_uuid
            )
        )
        details_text = bx_composer.compose_bx_job_details(
            bx_job_record, bx_jobs_bx_tasks_bx_gates_records
        )

        bx_variable_records = await bx_datafaces_get_default().get_bx_variables(
            bx_job_uuid
        )
        variables_text = bx_composer.compose_bx_variables(bx_variable_records)

        return f"{news_text}\n\n{details_text}\n\n{variables_text}"
