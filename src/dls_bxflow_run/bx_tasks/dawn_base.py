import logging
import os
import subprocess

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Base class for a bx_task.
from dls_bxflow_run.bx_tasks.base import Base

# Constants about error lines extraction.
from dls_bxflow_run.bx_tasks.constants import ExtractionErrorLinesMessages

# States.
from dls_bxflow_run.bx_tasks.states import States

logger = logging.getLogger(__name__)

PROBLEM_READING = "problem reading"
EXISTS_BUT_IS_EMPTY = "exists but is empty"
DOES_NOT_EXIST = "does not exist"


class DawnBase(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, thing_type, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.state(States.PREPARED)

    # ----------------------------------------------------------------------------------------
    def launch_and_wait(self, json_filename):
        """Run DAWN with json config file.

        https://confluence.diamond.ac.uk/display/I22/Headless+DAWN+Operation

        shellScriptPartTwo = (
            "$DAWN_RELEASE_DIRECTORY/dawn -noSplash
            -configuration "
            + tmpDirectoryPath
            + sep
            + ".eclipse -application org.dawnsci.commandserver.processing.processing -data @none -path "
            + jsonFilePath
            + " --launcher.appendVmargs -vmargs -Dgraylog.host=graylog2.diamond.ac.uk -Dgraylog.port=12206 -Ddawn.version=2.11.0 -Xmx"
            + str(self.maxMegsOfRam)
            + "m"
        )

        dawn -noSplash -application org.dawnsci.commandserver.processing.processing
        -configuration /path/to/work/area/.eclipse
        -data @none
        -path /path/to/json/file
        --launcher.appendVmargs -vmargs -Xmx4096m

        """

        # Presume we are operating in our private task directory
        runtime_directory = os.getcwd()

        command = []
        # TODO: In DAWN task, figure out correct executable binary.
        type_specific_tbd = require(
            f"{callsign(self)} specification", self.specification(), "type_specific_tbd"
        )
        # Get DAWN executable, allowing for configured override (for unit testing).
        dawn_executable = type_specific_tbd.get(
            "dawn_executable", "/dls_sw/apps/DawnDiamond/2.26/builds/release-linux/dawn"
        )

        # Build up the DAWN command line.
        command.extend(["/bin/bash", "-c", dawn_executable])
        command.extend(["-noSplash"])
        command.extend(
            ["-application", "org.dawnsci.commandserver.processing.processing"]
        )
        command.extend(["-configuration", f"{runtime_directory}/.eclipse"])
        command.extend(["-data", "@none"])
        command.extend(["-path", f"{json_filename}"])
        command.extend(["--launcher.appendVmargs"])
        command.extend(["-vmargs"])
        command.extend(["-Xmx4096m"])

        stdout_filename = "%s/dawn_stdout.txt" % (runtime_directory)
        stderr_filename = "%s/dawn_stderr.txt" % (runtime_directory)

        # Split the command into arguments/values for readability in the debug.
        readable = (" ".join(command)).replace(" -", " \\\n    -")
        logger.debug(f"{callsign(self)} running dawn with command\n{readable}")

        with open(stdout_filename, "wt") as stdout_handle:
            with open(stderr_filename, "wt") as stderr_handle:
                # Start but don't wait.
                process = subprocess.Popen(
                    command,
                    shell=False,
                    cwd=runtime_directory,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )

                logger.debug(f"{callsign(self)} submitted process pid {process.pid}")

                # TODO: Replace dawn process.wait() with asyncio.
                process.wait(timeout=None)

                exit_code = process.returncode

                # Good return code from dawn process itself?
                if exit_code == 0:
                    # Still need to check the file for errors.
                    # DAWN writes its errors to stdout.
                    error_lines = self.extract_error_lines_from_file(stdout_filename)

                    # Empty file is ok.
                    if len(error_lines) >= 1:
                        if (
                            ExtractionErrorLinesMessages.EXISTS_BUT_IS_EMPTY
                            not in error_lines[0]
                        ):
                            exit_code = 1
                    logger.debug(
                        f"{callsign(self)} process finished with returncode {process.returncode}, adjusted to {exit_code}"
                    )
                else:
                    logger.debug(
                        f"{callsign(self)} process finished with returncode {exit_code}"
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
        filename = f"{runtime_directory}/dawn_stdout.txt"

        # Died before getting to DAWN?
        # For example the assertions in the workflow run before running the DAWN executable.
        if not os.path.exists(filename):
            return self.extract_error_lines_from_dls_logformatter()

        return self.extract_error_lines_from_file(filename)

    # ------------------------------------------------------------------------------------------
    def extract_error_lines_from_file(self, filename):
        """
        Get task post-run fields after the task finished running.
        """

        error_lines = []

        logger.debug(f"[EXTERR] looking for error lines in {filename}")

        # There is a stderr file?
        if os.path.exists(filename):
            line_count = 0
            try:
                # Open and read the first chunk of it.
                with open(filename, "r") as stream:
                    for line in stream.readlines():
                        # Line is an error line?
                        # 08:14:47.091 ERROR Could not read model values
                        if line[12:19] == " ERROR ":
                            error_lines.append(line[19:].strip())
                        line_count += 1

                if line_count == 0:
                    error_lines.append(
                        f"{filename} {ExtractionErrorLinesMessages.EXISTS_BUT_IS_EMPTY}"
                    )
            except Exception as exception:
                s = str(exception)
                s = s.replace(filename, "")
                s = s.replace("''", "")
                s = s.replace('""', "")
                s = s.rstrip(": ")
                error_lines.append(
                    f"{ExtractionErrorLinesMessages.PROBLEM_READING} {filename}: {s}"
                )
        else:
            # There is no stderr file?
            error_lines.append(
                f"{filename} {ExtractionErrorLinesMessages.DOES_NOT_EXIST}"
            )

        return error_lines
