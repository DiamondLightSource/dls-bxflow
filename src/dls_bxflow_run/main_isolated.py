import asyncio
import logging
import multiprocessing
import os
import sys

import yaml

# Formatter for python logging.
from dls_logformatter.dls_logformatter import DlsLogformatter

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.explain import explain2, explain_cause_chain

# BxDataface context.
from dls_bxflow_api.bx_datafaces.context import Context as BxDatafaceContext

# BxTask manager.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

logger = logging.getLogger(__name__)


# --------------------------------------------------------------
class MainIsolated:
    """
    Class to run a bx_task, executing within a python main program.
    Set up logging according to the bxflow standard logging.
    Read the yaml configuration supplied, always in ./bxflow.yaml.
    Instantiate the bx_task object given in the configuration.
    Run the task.

    Note there is no try/catch here.
    It is expected that the process has been started to pipe stdout and sterr.
    An uncaught exception should naturally return exit_code 1.
    On success, the exit_code should naturally be 0.
    """

    def run(self):
        """
        This is the main program which calls the actual code using asyncio.
        """

        multiprocessing.current_process().name = "taskmain"

        # Run main code in asyncio event loop.
        return asyncio.run(self.run_coroutine())

    # ----------------------------------------------------------
    async def run_coroutine(self):
        """"""

        # Set up logging according to the bxflow standard logging.
        self.configure_logging()

        logger.debug("logging configured")
        try:
            filename = f"{os.getcwd()}/.bxflow/main.yaml"
            # Read the yaml configuration supplied, always in ./bxflow.yaml.
            with open(filename, "r") as yaml_stream:
                configuration = yaml.safe_load(yaml_stream)
        except Exception as exception:
            logger.error(
                explain2(
                    exception, f"reading the isolation configuration file {filename}"
                ),
                exc_info=exception,
            )
            return 1

        logger.debug("configuration loaded")

        try:
            # Instantiate the bx_task object given in the configuration.
            bx_task_specification = configuration["bx_task_specification"]
            bx_task = BxTasks().build_object(bx_task_specification)
        except Exception as exception:
            logger.error(
                explain2(
                    exception, f"instantating the bx_task specified in {filename}"
                ),
                exc_info=exception,
            )
            return 1

        logger.debug(f"main_isolated has built {callsign(bx_task)} object")

        # Task needs a dataface context?
        # For example, runtime variables, dynamic gate changes, or news events.
        if "bx_dataface_specification" in configuration:
            return await self.__run_in_dataface_context(
                filename, configuration, bx_task
            )
        else:
            return await self.__run_without_dataface_context(bx_task)

    # ----------------------------------------------------------
    async def __run_in_dataface_context(self, filename, configuration, bx_task):
        """
        Run the task inside a dataface client context.

        For example when the task needs runtime variables, dynamic gate changes, or news events.
        """

        # -------------------------------------------------------------
        try:
            # Load the bx_variables from the configuration.
            bx_variables_dict = configuration["bx_variables"]
            for bx_variable in bx_variables_dict:
                bx_task.variables.add(bx_variable["name"], bx_variable["value"])
        except Exception as exception:
            logger.error(
                explain2(
                    exception, f"loading the bx_variables specified in {filename}"
                ),
                exc_info=exception,
            )
            return 1
        logger.debug("bx_task variables added")

        # -------------------------------------------------------------
        try:
            # Make a client-only dataface context.
            bx_dataface_specification = configuration["bx_dataface_specification"]
            if "context" not in bx_dataface_specification:
                bx_dataface_specification["context"] = {}
            bx_dataface_specification["context"]["start_as"] = "client"
            bx_dataface_context = BxDatafaceContext(bx_dataface_specification)

        except Exception as exception:
            logger.error(
                explain2(
                    exception,
                    f"creating the bx_dataface context specified in {filename}",
                ),
                exc_info=exception,
            )
            return 1

        # Enable a dataface context in case the task needs to talk to it.
        # For example, by run-time variables, runtime gate changes, or runtime news.
        # TODO: In main_isolated, ensure dataface client is passive until used.
        # TODO: In main_isolated, only create a dataface context on demand
        async with bx_dataface_context:
            try:
                # Call the run method of the bx_task class.
                exit_code = await bx_task.run()
            except Exception as exception:
                logger.error(
                    explain_cause_chain(
                        exception, f"in the {callsign(bx_task)} run method"
                    ),
                    exc_info=exception,
                )
                return 1

        logger.debug(
            f"main_isolated sees {callsign(bx_task)} finished exit_code {exit_code}"
        )

        return exit_code

    # ----------------------------------------------------------
    async def __run_without_dataface_context(self, bx_task):
        """
        Run the task with no dataface client context.

        Thus it cannot use runtime variables, dynamic gate changes, or news events.
        """

        try:
            # Call the run method of the bx_task class.
            exit_code = await bx_task.run()
        except Exception as exception:
            logger.error(
                explain_cause_chain(
                    exception, f"in the {callsign(bx_task)} run method"
                ),
                exc_info=exception,
            )
            return 1

        logger.debug(
            f"main_isolated sees {callsign(bx_task)} finished exit_code {exit_code}"
        )

        return exit_code

    # --------------------------------------------------------------------------
    def configure_logging(self):
        """
        Set up logging according to the bxflow standard logging.
        """

        # Clear existing handlers.
        # Sometimes high level packages will establish their own handlers when imported.
        # TODO: Ddoes removing logging handlers has adverse effects in main_isolated?
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.DEBUG)

        # Make a handler to write file.
        # TODO: Consider that main_isolated should use RotatingFileHandler.
        file_handler = logging.FileHandler("main.log")
        file_handler.setFormatter(DlsLogformatter())
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

        # Make a handler to write console (stderr) for errors only.
        stderr_handler = logging.StreamHandler(stream=sys.stderr)
        stderr_handler.setFormatter(DlsLogformatter())
        stderr_handler.setLevel(logging.ERROR)
        root_logger.addHandler(stderr_handler)

        # Make a handler to write console (stdout) for info only, in short format.
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setFormatter(DlsLogformatter(type="bare"))
        stdout_handler.setLevel(logging.INFO)
        root_logger.addHandler(stdout_handler)


# ---------------------------------------------------------------
def main():

    # Instantiate the program class.
    main = MainIsolated()

    # Run the main.
    exit_code = main.run()

    return exit_code


# ---------------------------------------------------------------
# From command line, invoke the main method.
if __name__ == "__main__":
    exit_code = main()

    sys.exit(exit_code)
