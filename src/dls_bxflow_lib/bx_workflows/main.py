#!/usr/bin/env python

import argparse
import asyncio
import inspect
import logging
import os
import sys
import tempfile

# Base class with methods supporting MaxIV command-line programs.
from dls_mainiac_lib.mainiac import Mainiac

# Object managers we interact with.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# The package version.
from dls_bxflow_cli.version import meta as version_meta
from dls_bxflow_cli.version import version

# Results composers.
from dls_bxflow_lib.bx_composers.bx_composers import BxComposers

# Configurator.
from dls_bxflow_lib.bx_configurators.bx_configurators import (
    BxConfigurators,
    bx_configurators_set_default,
)

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

logger = logging.getLogger(__name__)


# --------------------------------------------------------------
class Main(Mainiac):
    def __init__(self, workflow_class):
        super().__init__("bx_workflow")

        # Remember the workflow class we are running.
        self.__workflow_class = workflow_class

        # Temporary directory (created later) for the live of this main program.
        self.__temporary_directory = None

        # TODO: Add option to disable running bxworkflow in the main constructor.
        # Configure the app from command line arguments.
        self.parse_args_and_configure_logging()

        # Run the main wrapped in a try/catch.
        self.try_run_catch()

    # ----------------------------------------------------------
    def run(self):
        """"""

        # Run the command line main in asyncio event loop.
        asyncio.run(self.run_coro())

    # ----------------------------------------------------------
    async def run_coro(self):
        """
        Run the workflow builder and enable it for scheduling.
        """

        # Define the configuration source.
        bx_configurator = self.get_bx_configurator()

        # For convenience, make a temporary directory for this run.
        self.__temporary_directory = tempfile.TemporaryDirectory()

        # Make the temporary directory available to the configurator.
        bx_configurator.substitute(
            {"temporary_directory": self.__temporary_directory.name}
        )

        # Actually load the configuration into a dict.
        context_specification = await bx_configurator.load()

        # Make a context from the configuration.
        bx_context = BxContexts().build_object(context_specification)

        # Open the context (servers).
        async with bx_context:
            # Start a news consumer.
            await bx_context.add_news_consumer(self._consume_bx_news)

            # Make a dict from the args namespace.
            vars_args = vars(self._args)
            kwargs = {}
            # Loop through the workflow constructor's defined kwargs.
            for name, value in self.__workflow_class.constructor_kwargs.items():
                # Grab the value that was parsed from the command line by mainiac.
                kwargs[name] = vars_args[name]

            # Instantiate the workflow builder class.
            workload_object = self.__workflow_class(**kwargs)

            try:
                # Let the builder build the workflow.
                if inspect.iscoroutinefunction(workload_object.build):
                    await workload_object.build()
                else:
                    workload_object.build()
            except Exception:
                raise RuntimeError("unable to build the workflow")

            if self._args.visualize:
                # Create a png and html files for visualizing.
                filenames = workload_object.visualize()
                logger.info("wrote visualization files %s" % (filenames))
                return

            # Register all the variables that have been added.
            await workload_object.bx_variables.register(workload_object.bx_job.uuid())

            # Schedule the bx_job to run.
            await workload_object.start()

            logger.info("job submitted... waiting for it to finish")

            try:
                # Wait for bx_job to finish.
                await workload_object.wait(timeout=None, naptime=1.0)
            except KeyboardInterrupt:
                pass

            # Summarize the job's execution.
            job_summary_text = await self._compose_job_summary(
                workload_object.bx_job.uuid()
            )

        # -------------------------------------------------------------
        # Context is now closed, database and all other services are unavailable.
        # Print the job summary to the console.
        logger.info(f"job summary\n{job_summary_text}")

    # ----------------------------------------------------------------------------------------
    async def _consume_bx_news(self, topic, headline, details):
        """ """
        logger.info(f"{headline}")

    # ----------------------------------------------------------
    def build_parser(self, arglist=None):
        """
        Method called from mainiac command line parsing.
        Should return argparser for this program.
        """
        # Make a parser.
        parser = argparse.ArgumentParser()

        # Using the workflow builder's constructor's defined kwargs, add argparse arguments.
        # These are always keyword.
        for name, value in self.__workflow_class.constructor_kwargs.items():
            # TODO: Use BxSettings to better create command line arguments.
            if isinstance(value, dict):
                value = value.get("default_value", "")
            data_type = type(value).__name__
            metavar = f"{name} ({data_type}, default {value})"
            parser.add_argument(
                f"--{name}",
                type=type(value),
                dest=name,
                metavar=metavar,
                required=False,
                default=value,
            )

        parser.add_argument(
            "--visualize",
            help="Build workflow visualization and exit.",
            action="store_true",
            dest="visualize",
        )

        parser.add_argument(
            "--visit",
            help="Visit name.",
            dest="visit",
        )

        return parser

    # ----------------------------------------------------------
    def version(self):
        """
        Method called from mainiac command line parsing.
        Should return string in form of N.N.N.
        """
        return version()

    # ----------------------------------------------------------
    def about(self):
        """
        Method called from mainiac command line parsing.
        Should return dict which can be serialized by json.
        """

        return {"versions": version_meta()}

    # ----------------------------------------------------------------------------------------
    def get_bx_configurator(self, environ=None):

        # Just return the vanilla one.
        # This method should normally be called by the beamline-specific Main.
        bx_configurator = BxConfigurators().build_object_from_environment(
            environ=environ
        )

        # Add various things from the environment into the configurator.
        bx_configurator.substitute(
            {
                "sys.prefix": sys.prefix,
                "CWD": os.getcwd(),
                "HOME": os.environ.get("HOME", "HOME"),
                "USER": os.environ.get("USER", "USER"),
                "PATH": os.environ.get("PATH", "PATH"),
                "PYTHONPATH": os.environ.get("PYTHONPATH", "PYTHONPATH"),
            }
        )

        bx_configurators_set_default(bx_configurator)

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


# # --------------------------------------------------------------------------------
# class _matplotlib_logging_filter:
#     """
#     Python logging filter to remove annoying matplotlib messages.
#     These are not super useful to see all the time at the INIT level.
#     """

#     def filter(self, record):
#         if "loaded modules" in record.msg:
#             return 0

#         return 1
