import json
import logging
import os
import re
import subprocess

import yaml

# Utilities.
from dls_utilpack.bash_composer import BashComposer
from dls_utilpack.callsign import callsign
from dls_utilpack.describe import describe
from dls_utilpack.explain import explain
from dls_utilpack.require import require
from dls_utilpack.substitute import substitute_dict

# Remote execution.
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Base class for a bx_task.
from dls_bxflow_run.bx_tasks.base import Base

# Task constants.
from dls_bxflow_run.bx_tasks.constants import ExtractionErrorLinesMessages
from dls_bxflow_run.bx_tasks.constants import Types as BxTaskTypes
from dls_bxflow_run.bx_tasks.states import States

logger = logging.getLogger(__name__)

PROBLEM_READING = "problem reading"
EXISTS_BUT_IS_EMPTY = "exists but is empty"
DOES_NOT_EXIST = "does not exist"

thing_type = BxTaskTypes.PTYREX_MPI


class PtyrexMpi(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(
            self,
            thing_type,
            specification,
            predefined_uuid=predefined_uuid,
        )

        self.state(States.PREPARED)

    # ----------------------------------------------------------------------------------------
    def get_propagated_filename(self):
        """Filename where we promise to write the propagated output file."""

        s = f"{self.label()} specification"
        type_specific_tbd = require(s, self.specification(), "type_specific_tbd")

        # Propagate MUST be in the specification.
        propagate = require(s, type_specific_tbd, "propagate")
        # However, value of propagate is allowed to be None.
        if propagate is None:
            raise RuntimeError(
                f"{self.callsign()} is not configured to provide propagated output"
            )

        # Use fileno to predict where ptyrex_mpi_recipe writes the file.
        fileno = self.get_fileno()
        return f"{self.get_directory()}/scan_{fileno}_propagated.nxs"

    # ----------------------------------------------------------------------------------------
    def get_output_filenames(self):
        """Return dict with filenames where we promise to write the mag and phase output files."""

        # Use fileno to predict where ptyrex_mpi_recipe writes the file.
        fileno = self.get_fileno()

        ptyr_filename = f"{self.get_directory()}/scan_{fileno}.ptyr"
        mag_filename = f"{self.get_directory()}/scan_{fileno}SI14G00_mag.nxs"
        phase_filename = f"{self.get_directory()}/scan_{fileno}SI14G00_phase.nxs"

        return {
            "ptyr": ptyr_filename,
            "mag": mag_filename,
            "phase": phase_filename,
        }

    # ----------------------------------------------------------------------------------------
    def get_fileno(self):
        """Scan number derived from specification data_filename."""

        s = f"{self.label()} specification"
        type_specific_tbd = require(s, self.specification(), "type_specific_tbd")
        data_filename = require(s, type_specific_tbd, "data_filename")

        # Get just the scan number as wanted for substitution by ptyrex_configfile.
        try:
            fileno = os.path.basename(data_filename)
            fileno = os.path.splitext(fileno)[0]
            fileno = fileno.split("-")[1]
        except Exception:
            raise ValueError(
                f"fileno pattern bbb-NNNNNN.nxs not matched in data_filename {data_filename}"
            )

        return fileno

    # ----------------------------------------------------------------------------------------
    def __customize_ptyrex_configfile(
        self, ptyrex_configfile, updated_fields, substitutions
    ):
        """
        Replace symbols in the given ptyrex_configfile and
        the updated file to the task's directory.
        """

        try:
            # Read file as a dict.
            with open(ptyrex_configfile, "r") as yaml_stream:
                loaded_dict = yaml.safe_load(yaml_stream)
        except Exception as exception:
            raise RuntimeError(
                explain(exception, f"reading ptyrex_configfile {ptyrex_configfile}")
            )

        # Customize the fields from a dict of the same form factor.
        loaded_dict.update(updated_fields)

        # Substitute ${token}-like things in the dict.
        substitute_dict(loaded_dict, substitutions)

        # Write as new yaml file in task's directory.
        resolved_filename = f"{self.get_directory()}/ptyrex_configfile.json"

        with open(resolved_filename, "w") as stream:
            json.dump(loaded_dict, stream, indent=4, sort_keys=False)

        return resolved_filename

    # ----------------------------------------------------------------------------------------
    def validate_specification(self):
        """
        Validate the task's specification.

        Can be called at build time, or at run time.

        Look for things which might make it fail.
        """

        s = f"{self.label()}"

        type_specific_tbd = require(s, self.specification(), "type_specific_tbd")

        # Make sure the data_filename is provide and the file itself is accessible.
        self.__data_filename = require(s, type_specific_tbd, "data_filename")
        if self.__data_filename is None:
            raise RuntimeError(f"{s} data_filename is None")
        if self.__data_filename == "":
            raise RuntimeError(f"{s} data_filename is blank")
        # if not os.path.exists(self.__data_filename):
        #     raise RuntimeError(f"cannot find {s} data_filename {self.__data_filename}")

        # Make sure the configfile is good.
        self.__ptyrex_configfile = require(s, type_specific_tbd, "ptyrex_configfile")
        if self.__ptyrex_configfile is None:
            raise RuntimeError(f"{s} ptyrex_configfile is None")
        if self.__ptyrex_configfile == "":
            raise RuntimeError(f"{s} ptyrex_configfile is blank")
        if not os.path.exists(self.__ptyrex_configfile):
            raise RuntimeError(
                f"cannot find {s} ptyrex_configfile {self.__ptyrex_configfile}"
            )

        self.__ptyrex_configfile_updated_fields = require(
            s, type_specific_tbd, "ptyrex_configfile_updated_fields"
        )
        self.__ptyrex_configfile_substitutions = require(
            s, type_specific_tbd, "ptyrex_configfile_substitutions"
        )

        # Configuration for modules that are to be loaded
        # in the script which runs the reconstruction.
        # This allows the reconstruction to run in a different
        # environment from the one where bxflow is running.
        self.__load_modules = require(s, type_specific_tbd, "load_modules")

        # Assign values to symbol definitions in the ptyrex_configfile template.
        self.__ptyrex_configfile_substitutions["data_filename"] = self.__data_filename
        self.__ptyrex_configfile_substitutions["task_directory"] = self.get_directory()

        self.__remex_hints = require(
            "task specification",
            self.specification(),
            "remex_hints",
        )
        logger.debug(f"remex_hints\n{json.dumps(self.__remex_hints, indent=4)}")

        self.__pe = require(
            "task remex_hints", self.__remex_hints, RemexKeywords.PARALLEL_ENVIRONMENT
        )

        parts = re.split("[ ]+", self.__pe.strip())
        if len(parts) < 2:
            raise RuntimeError(
                f"task remex_hints {RemexKeywords.PARALLEL_ENVIRONMENT}"
                f" '{self.__pe}' is invalid because not enough tokens"
            )

        if not parts[-1].isdigit():
            raise RuntimeError(
                f"task remex_hints {RemexKeywords.PARALLEL_ENVIRONMENT}"
                f" '{self.__pe}' is invalid because last token is not an integer"
            )

        self.__mpi_number_of_processors = int(parts[-1])

    # ----------------------------------------------------------------------------------------
    async def run(self):
        """Run ptyrex_recon launcher."""

        # Validate the task specifications we are being provided.
        # Look for things which might make it fail.
        self.validate_specification()

        # Resolve all the tokens in the ptyrex_configfile's contents.
        # For example ${visit.directory}.
        self.__ptyrex_configfile = self.__customize_ptyrex_configfile(
            self.__ptyrex_configfile,
            self.__ptyrex_configfile_updated_fields,
            self.__ptyrex_configfile_substitutions,
        )

        # Build an on-the-fly script to run the pytrex_recon.
        # This script is run with no environment variables set.
        bash_composer = BashComposer()

        bash_composer.add_load_modules(
            self.__load_modules.get("directories", []),
            self.__load_modules.get("modules"),
        )

        # Put a few things in the log.
        bash_composer.add_command("lscpu")
        bash_composer.add_command("nvidia-smi")

        # Run the actual command.
        bash_composer.add_command(
            f"mpirun -np {self.__mpi_number_of_processors} ptyrex_recon -c {self.__ptyrex_configfile}",
        )

        bash_filename = "%s/ptyrex_mpi.sh" % (self.get_directory())
        bash_composer.write(bash_filename)

        # Build the ptyrex_mpi_recipe command.
        command = []
        command.append(bash_filename)

        # These are the outputs by the various scripts leading up to the ptyrex itself.
        stdout_filename = "%s/ptyrex_mpi_stdout.txt" % (self.get_directory())
        stderr_filename = "%s/ptyrex_mpi_stderr.txt" % (self.get_directory())

        # Split the command into arguments/values for readability in the debug.
        readable = (" ".join(command)).replace(" -", " \\\n    -")
        logger.debug(f"{callsign(self)} starting command\n{readable}")

        with open(stdout_filename, "wt") as stdout_handle:
            with open(stderr_filename, "wt") as stderr_handle:
                # Start but don't wait.
                process = subprocess.Popen(
                    command,
                    shell=False,
                    cwd=self.get_directory(),
                    env={},
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )

                logger.debug(
                    f"{callsign(self)} running process pid {process.pid}, waiting for it"
                )

                # TODO: Replace ptyrex_mpi process.wait() with asyncio.
                process.wait(timeout=None)

                exit_code = process.returncode

        # Good return code from ptyrex_mpi process itself?
        if exit_code == 0:
            # Still need to check the file for errors.
            error_lines = self.extract_error_lines_from_file(stderr_filename)

            logger.debug(describe("error_lines", error_lines))

            if len(error_lines) > 0:
                exit_code = 1
                logger.debug(
                    f"{callsign(self)} some error lines found so setting exit_code {exit_code}"
                )
        else:
            logger.debug(
                f"{callsign(self)} process finished with returncode {exit_code}"
            )

        if os.path.exists(stderr_filename):
            self.propose_artefact(stderr_filename)

        # Good return code from ptyrex_mpi process itself?
        # if exit_code == 0:
        #     pattern = f"{self.get_directory()}/scan_*/**"
        #     outputs = glob.glob(pattern, recursive=True)

        #     if len(outputs) == 0:
        #         error_lines.append(f"found no outputs like {pattern}")
        #         logger.debug(
        #             f"{callsign(self)} outputs not found so setting exit_code {exit_code}"
        #         )
        #         exit_code = 1
        #     else:
        #         logger.info(describe("outputs", outputs))

        #         # Move all files out of their scan subdirectories.
        #         for output in outputs:
        #             if not os.path.isdir(output):
        #                 move_to = f"{self.get_directory()}/{os.path.basename(output)}"
        #                 os.rename(output, move_to)
        #                 self.propose_artefact(move_to)

        #         # Remove the scan subdirectories.
        #         for output in outputs:
        #             if os.path.isdir(output):
        #                 os.rmdir(output)

        return exit_code

    # ------------------------------------------------------------------------------------------
    def extract_error_lines(self):
        """
        Get the most obvious error line from among the log files written.
        """

        # The ptyrex_mpi_recipe script normally produces a stderr file.
        # This has the "module load" blather and seems some stuff from ptyrex.
        ptyrex_mpi_stderr_filename = f"{self.get_directory()}/ptyrex_mpi_stderr.txt"
        ptyrex_log_filename = f"{self.get_directory()}/ptyrex.log"

        # Died before getting to ptyrex_mpi?
        # For example the assertions in the workflow run before running the ptyrex_mpi executable.
        if not os.path.exists(ptyrex_mpi_stderr_filename):
            error_lines = [f"could not find {ptyrex_mpi_stderr_filename}"]
            # Check the stdout file which is written by main.sh.
            error_lines.extend(self.extract_error_lines_from_dls_logformatter())

        elif os.path.getsize(ptyrex_mpi_stderr_filename) == 0:
            error_lines = [f"nothing in {ptyrex_mpi_stderr_filename}"]
            # Check the stdout file which is written by main.sh.
            error_lines.extend(self.extract_error_lines_from_dls_logformatter())

        else:
            regexs = []
            regexs.append(re.compile(r"^.*exception.*$", re.IGNORECASE))
            regexs.append(re.compile(r"^.*error.*$", re.IGNORECASE))
            error_lines = self.extract_error_lines_from_file(
                ptyrex_mpi_stderr_filename, regexs
            )

            if len(error_lines) == 0:
                if not os.path.exists(ptyrex_log_filename):
                    error_lines.append(f"could not find {ptyrex_log_filename}")

                elif os.path.getsize(ptyrex_log_filename) == 0:
                    error_lines.append(f"nothing in {ptyrex_log_filename}")

                else:
                    error_lines = self.extract_error_lines_from_file(
                        ptyrex_log_filename, regexs
                    )

        return error_lines

    # ------------------------------------------------------------------------------------------
    def extract_error_lines_from_file(self, filename, regexs=[]):
        """
        Try to find the pertinent error in the output file.
        """

        error_lines = []
        if not os.path.exists(filename):
            error_lines.append(f"could not find errors file {filename}")
        elif os.path.getsize(filename) == 0:
            # Empty file was not good for pytpy, but for ptyrex is ok for now.
            # error_lines.append(
            #     f"{filename} {ExtractionErrorLinesMessages.EXISTS_BUT_IS_EMPTY}"
            # )
            pass
        else:
            try:
                # Open and scan for first line in file to match any regex.
                matched = False
                with open(filename, "r") as stream:
                    while not matched:
                        line = stream.readline()
                        # End of file?
                        if line == "":
                            break
                        line = line.strip()
                        for regex in regexs:
                            if regex.match(line):
                                error_lines.append(line)
                                # Stop at the first matched line in the file.
                                matched = True

                if matched:
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
