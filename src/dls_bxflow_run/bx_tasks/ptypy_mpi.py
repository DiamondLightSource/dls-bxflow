import glob
import logging
import os
import re
import subprocess

import yaml

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.describe import describe
from dls_utilpack.explain import explain
from dls_utilpack.module import module_get_environ
from dls_utilpack.require import require
from dls_utilpack.substitute import substitute_dict

# Base class for a bx_task.
from dls_bxflow_run.bx_tasks.base import Base
from dls_bxflow_run.bx_tasks.constants import ExtractionErrorLinesMessages
from dls_bxflow_run.bx_tasks.constants import Types as BxTaskTypes

# Task constants.
from dls_bxflow_run.bx_tasks.states import States

logger = logging.getLogger(__name__)

PROBLEM_READING = "problem reading"
EXISTS_BUT_IS_EMPTY = "exists but is empty"
DOES_NOT_EXIST = "does not exist"

thing_type = BxTaskTypes.PTYPY_MPI


class PtypyMpi(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.state(States.PREPARED)

    # ----------------------------------------------------------------------------------------
    def get_propagated_filename(self):
        """Filename where we promise to write the propagated output file."""

        s = f"{callsign(self)} specification"
        type_specific_tbd = require(s, self.specification(), "type_specific_tbd")

        # Propagate MUST be in the specification.
        propagate = require(s, type_specific_tbd, "propagate")
        # However, value of propagate is allowed to be None.
        if propagate is None:
            raise RuntimeError(
                f"{self.callsign()} is not configured to provide propagated output"
            )

        # Use fileno to predict where ptypy_mpi_recipe writes the file.
        fileno = self.get_fileno()
        return f"{self.get_directory()}/scan_{fileno}_propagated.nxs"

    # ----------------------------------------------------------------------------------------
    def get_output_filenames(self):
        """Return dict with filenames where we promise to write the mag and phase output files."""

        # Use fileno to predict where ptypy_mpi_recipe writes the file.
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

        s = f"{callsign(self)} specification"
        type_specific_tbd = require(s, self.specification(), "type_specific_tbd")
        data_filename = require(s, type_specific_tbd, "data_filename")

        # Get just the scan number as wanted for substitution by ptypy_configfile.
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
    def customize_ptypy_configfile(
        self, ptypy_configfile, updated_fields, substitutions
    ):
        """
        Replace symbols in the given ptypy_configfile and
        the updated file to the task's directory.
        """

        try:
            # Read file as a dict.
            with open(ptypy_configfile, "r") as yaml_stream:
                loaded_dict = yaml.safe_load(yaml_stream)
        except Exception as exception:
            raise RuntimeError(
                explain(exception, f"reading ptypy_configfile {ptypy_configfile}")
            )

        # Customize the fields from a dict of the same form factor.
        loaded_dict.update(updated_fields)

        # Substitute ${token}-like things in the dict.
        substitute_dict(loaded_dict, substitutions)

        # Write as new yaml file in task's directory.
        resolved_filename = f"{self.get_directory()}/ptypy_configfile.yaml"

        with open(resolved_filename, "w") as stream:
            yaml.dump(loaded_dict, stream, default_flow_style=False, sort_keys=False)

        return resolved_filename

    # ----------------------------------------------------------------------------------------
    async def run(self):
        """Run ptypy_mpi_recipe as a shell command."""

        s = f"{callsign(self)} specification"
        type_specific_tbd = require(s, self.specification(), "type_specific_tbd")

        remex_hints = self.specification().get("remex_hints", {})

        # Get the parameters of the task.
        fileno = self.get_fileno()

        # Make sure the configfile is good.
        ptypy_configfile = require(s, type_specific_tbd, "ptypy_configfile")
        if ptypy_configfile is None:
            raise RuntimeError("task configured ptypy_configfile is None")
        if ptypy_configfile == "":
            raise RuntimeError("task configured ptypy_configfile is blank")
        if not os.path.exists(ptypy_configfile):
            raise RuntimeError(
                f"cannot find task configured ptypy_configfile {ptypy_configfile}"
            )

        ptypy_configfile_updated_fields = require(
            s, type_specific_tbd, "ptypy_configfile_updated_fields"
        )
        ptypy_configfile_substitutions = require(
            s, type_specific_tbd, "ptypy_configfile_substitutions"
        )

        # Propagate MUST be in the specification.
        # However, value of propagate is allowed to be None.
        propagate = require(s, type_specific_tbd, "propagate")

        single_threaded = type_specific_tbd.get("single_threaded", False)

        # Resolve all the tokens in the ptypy_configfile's contents.
        # For example ${visit.directory}.
        ptypy_configfile = self.customize_ptypy_configfile(
            ptypy_configfile,
            ptypy_configfile_updated_fields,
            ptypy_configfile_substitutions,
        )

        # This is the module to be loaded by ptychotools.ptypy_mpi_recipe when it runs.
        ptypy_version = type_specific_tbd.get("ptypy_version", "ptycho-tools/stable")

        old_path = os.environ["PATH"]

        try:
            # We might want to to override where to find the ptychotools.ptypy_mpi_recipe.
            # Note this is just a bash script with no dependencies on the ptypy package.
            if "ptychotools_path" in type_specific_tbd:
                ptychotools_path = type_specific_tbd["ptychotools_path"]
                logger.debug(f"updating PATH with ptychotools_path {ptychotools_path}")
                os.environ[
                    "PATH"
                ] = f"{ptychotools_path}/scripts:{os.environ.get('PATH')}"
                os.environ[
                    "PYTHONPATH"
                ] = f"{ptychotools_path}:{os.environ.get('PYTHONPATH', '')}"
            else:
                # Get the environment the module would like to set when loading.
                ptypy_environ_dict = module_get_environ(ptypy_version)
                logger.debug(
                    describe(f"{ptypy_version} ptypy_environ_dict", ptypy_environ_dict)
                )
                # Set just the path to ptychotools.ptypy_mpi_recipe from the module.
                # When the ptychotools.ptypy_mpi_recipe bash script runs,
                # it will do a module load on ptypy_version anyway.
                os.environ["PATH"] = ptypy_environ_dict["PATH"]
                os.environ["PYTHONPATH"] = ptypy_environ_dict.get("PYTHONPATH", "")

            logger.debug(
                describe(
                    "PATH",
                    os.environ.get("PATH").split(":"),
                )
            )
            logger.debug(
                describe(
                    "PYTHONPATH",
                    os.environ.get("PYTHONPATH").split(":"),
                )
            )

            # Build the ptypy_mpi_recipe command.
            command = []
            command.append("ptychotools.ptypy_mpi_recipe")
            command.extend(["-i", fileno])
            command.extend(["-j", str(ptypy_configfile)])
            command.extend(["-o", self.get_directory()])
            if propagate is not None:
                command.extend(["-p", propagate])
            command.extend(["-z", "ptypy.log"])
            command.extend(["-n", "1"])
            command.extend(["-v", ptypy_version])

            if single_threaded:
                command.extend(["-s", "true"])
            else:
                command.extend(["-s", "false"])

            # Task indicates it wants some number of gpu?
            num_gpu = remex_hints.get("ptypy_mpi", {}).get("num_gpu", 4)
            if num_gpu > 0:
                # This will engage the remex_hint ptypy_mpi num_gpu.
                command.extend(["-g"])

            # These are the outputs by the various scripts leading up to the ptypy itself.
            stdout_filename = "%s/ptypy_mpi_stdout.txt" % (self.get_directory())
            stderr_filename = "%s/ptypy_mpi_stderr.txt" % (self.get_directory())
            # This is output by the ptypy python package.
            ptypy_log_filename = "%s/ptypy.log" % (self.get_directory())

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
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                    )

                    logger.debug(
                        f"{callsign(self)} running process pid {process.pid}, waiting for it"
                    )

                    # TODO: Replace ptypy_mpi process.wait() with asyncio.
                    process.wait(timeout=None)

                    exit_code = process.returncode

            # Good return code from ptypy_mpi process itself?
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
            if os.path.exists(ptypy_log_filename):
                self.propose_artefact(ptypy_log_filename)

            # Good return code from ptypy_mpi process itself?
            if exit_code == 0:
                pattern = f"{self.get_directory()}/scan_*/**"
                outputs = glob.glob(pattern, recursive=True)

                if len(outputs) == 0:
                    error_lines.append(f"found no outputs like {pattern}")
                    logger.debug(
                        f"{callsign(self)} outputs not found so setting exit_code {exit_code}"
                    )
                    exit_code = 1
                else:
                    logger.info(describe("outputs", outputs))

                    # Move all files out of their scan subdirectories.
                    for output in outputs:
                        if not os.path.isdir(output):
                            move_to = (
                                f"{self.get_directory()}/{os.path.basename(output)}"
                            )
                            os.rename(output, move_to)
                            self.propose_artefact(move_to)

                    # Remove the scan subdirectories.
                    for output in outputs:
                        if os.path.isdir(output):
                            os.rmdir(output)

            return exit_code

        finally:
            os.environ["PATH"] = old_path

    # ------------------------------------------------------------------------------------------
    def extract_error_lines(self):
        """
        Get the most obvious error line from among the log files written.
        """

        # The ptypy_mpi_recipe script normally produces a stderr file.
        # This has the "module load" blather and seems some stuff from ptypy.
        ptypy_mpi_stderr_filename = f"{self.get_directory()}/ptypy_mpi_stderr.txt"
        ptypy_log_filename = f"{self.get_directory()}/ptypy.log"

        # Died before getting to ptypy_mpi?
        # For example the assertions in the workflow run before running the ptypy_mpi executable.
        if not os.path.exists(ptypy_mpi_stderr_filename):
            error_lines = [f"could not find {ptypy_mpi_stderr_filename}"]
            # Check the stdout file which is written by main.sh.
            error_lines.extend(self.extract_error_lines_from_dls_logformatter())

        elif os.path.getsize(ptypy_mpi_stderr_filename) == 0:
            error_lines = [f"nothing in {ptypy_mpi_stderr_filename}"]
            # Check the stdout file which is written by main.sh.
            error_lines.extend(self.extract_error_lines_from_dls_logformatter())

        else:
            regexs = []
            regexs.append(re.compile(r"^.*exception.*$", re.IGNORECASE))
            regexs.append(re.compile(r"^.*error.*$", re.IGNORECASE))
            error_lines = self.extract_error_lines_from_file(
                ptypy_mpi_stderr_filename, regexs
            )

            if len(error_lines) == 0:
                if not os.path.exists(ptypy_log_filename):
                    error_lines.append(f"could not find {ptypy_log_filename}")

                elif os.path.getsize(ptypy_log_filename) == 0:
                    error_lines.append(f"nothing in {ptypy_log_filename}")

                else:
                    error_lines = self.extract_error_lines_from_file(
                        ptypy_log_filename, regexs
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
            error_lines.append(
                f"{filename} {ExtractionErrorLinesMessages.EXISTS_BUT_IS_EMPTY}"
            )
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
