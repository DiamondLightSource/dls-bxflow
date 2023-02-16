import logging
import os
import subprocess

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Base class for a dawn task.
from dls_bxflow_run.bx_tasks.base import Base

# Constants about error lines extraction.
from dls_bxflow_run.bx_tasks.constants import ExtractionErrorLinesMessages

# States.
from dls_bxflow_run.bx_tasks.states import States

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_run.bx_tasks.shell"


class Shell(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.state(States.PREPARED)

    # ----------------------------------------------------------------------------------------
    async def run(self):
        """Run by shell."""

        # Presume we are operating in our private task directory
        runtime_directory = os.getcwd()

        # TODO: In DAWN task, figure out correct executable binary.
        type_specific_tbd = require(
            f"{callsign(self)} specification", self.specification(), "type_specific_tbd"
        )
        # Get DAWN executable, allowing for configured override (for unit testing).
        command = require(
            f"{callsign(self)} specification type_specific_tbd",
            type_specific_tbd,
            "command",
        )

        if not isinstance(command, list):
            raise RuntimeError(
                f"{callsign(self)} specification command keyword is not a list"
            )

        if not os.path.exists(command[0]):
            raise RuntimeError(
                f"{callsign(self)} specification command executable path not found {command[0]}"
            )

        # Replace certain symbols in the command with runtime values.
        for index, value in enumerate(command):
            if hasattr(value, "replace"):
                command[index] = value.replace(
                    "{bx_task_directory}", self.get_directory()
                )
            else:
                command[index] = str(value)

        stdout_filename = "%s/shell_stdout.txt" % (runtime_directory)
        stderr_filename = "%s/shell_stderr.txt" % (runtime_directory)

        # Split the command into arguments/values for readability in the debug.
        readable = (" ".join(command)).replace(" -", " \\\n    -")
        logger.debug(f"{callsign(self)} running command\n{readable}")

        with open(stdout_filename, "wt") as stdout_handle:
            with open(stderr_filename, "wt") as stderr_handle:
                try:
                    # Start but don't wait.
                    process = subprocess.Popen(
                        command,
                        shell=False,
                        cwd=runtime_directory,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                    )
                except Exception:
                    # If we get an exception doing the Popen, it is due to missing or non-executable file.
                    # In this case the stdout and stderr are always going to be emtpy.
                    # Remove them so as not to confuse the post-run logic that the launch was successful.

                    if os.path.exists(stdout_filename):
                        os.remove(stdout_filename)
                    if os.path.exists(stderr_filename):
                        os.remove(stderr_filename)
                    raise

                logger.debug(f"{callsign(self)} submitted process pid {process.pid}")

                # TODO: Replace dawn process.wait() with asyncio.
                process.wait(timeout=None)

                # The executable must properly return a code of non-zero if it failed.
                exit_code = process.returncode

                logger.debug(
                    f"{callsign(self)} process.returncode {process.returncode}"
                )

                return exit_code

    # ------------------------------------------------------------------------------------------
    def extract_error_lines(self):
        """
        Get task post-run fields after the task finished running.
        """

        # For clarity.
        bx_task = self

        # Runtime directory where the isolated task wrote its files.
        runtime_directory = bx_task.get_directory()

        # Name of of the stderr filename.
        # TODO: Centralize naming convention of stderr.txt filename.
        filename = f"{runtime_directory}/shell_stderr.txt"

        # Died before getting to shell?
        # For example the assertions in the workflow run before running the executable.
        if not os.path.exists(filename):
            return self.extract_error_lines_from_dls_logformatter()

        return self.extract_error_lines_from_file(filename)

    # ------------------------------------------------------------------------------------------
    def extract_error_lines_from_file(self, filename):
        """
        Try to find the pertinent error in the output file.
        """

        error_lines = []
        if not os.path.exists(filename):
            error_lines.append(f"could not find errors file {filename}")
        else:
            try:
                # Open and read the first chunk of it.
                with open(filename, "r") as stream:
                    first_line = stream.readline()
                    error_lines.append(first_line.strip())

                error_lines.append(f"for more, please see: {filename}")
            except Exception as exception:
                s = str(exception)
                s = s.replace(filename, "")
                s = s.replace("''", "")
                s = s.replace('""', "")
                s = s.rstrip(": ")
                error_lines.append(
                    f"{ExtractionErrorLinesMessages.PROBLEM_READING} {filename}: {s}"
                )

        return error_lines
