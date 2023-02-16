import copy
import logging
import os

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.describe import describe
from dls_utilpack.explain import explain2
from dls_utilpack.require import require
from dls_utilpack.sanitize import sanitize

from dls_bxflow_run.bx_tasks.base import Base

# Base class for a bx_task.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

# Contants about error lines extraction.
from dls_bxflow_run.bx_tasks.constants import ExtractionErrorLinesMessages

# States.
from dls_bxflow_run.bx_tasks.states import States

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_run.bx_tasks.jupyter"


class Jupyter(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.state(States.PREPARED)

        self.__modify_cells = None

    # ----------------------------------------------------------------------------------------
    def modify_notebook(self, notebook):
        """
        Offer opportunity to modify notebook.
        """

        if self.__modify_cells is None:
            return notebook

        for index, contents in self.__modify_cells.items():
            index = int(index)
            notebook["cells"][index]["source"] = contents

        return notebook

    # ----------------------------------------------------------------------------------------
    async def run(self):
        """
        Dummy will read infile_variable and write outfile_variable if given.
        """

        logger.debug("importing jupyter code")

        # Avoid: "DeprecationWarning: Jupyter is migrating its paths to use standard platformdirs"
        # Which appears when using a newer jupyter_core with an older nbclient.

        if "JUPYTER_PLATFORM_DIRS" not in os.environ:
            os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

        # Avoid: " Note: Debugging will proceed. Set PYDEVD_DISABLE_FILE_VALIDATION=1 to disable this validation."
        if "PYDEVD_DISABLE_FILE_VALIDATION" not in os.environ:
            os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

        # Jupyter.
        import nbformat
        from nbconvert import HTMLExporter

        type_specific_tbd = require(
            f"{callsign(self)} specification", self.specification(), "type_specific_tbd"
        )

        ipynb_filename = require(
            f"{callsign(self)} specification", type_specific_tbd, "ipynb_filename"
        )

        self.__modify_cells = type_specific_tbd.get("modify_cells")

        # Read the notebook into memory.
        logger.debug(f"reading notebook template from {ipynb_filename}")
        with open(ipynb_filename, "r") as stream:
            notebook = nbformat.read(stream, as_version=4)

        # Modify notebook.
        notebook = self.modify_notebook(notebook)

        # Where to save the executed notebook.
        sanitized_label = sanitize(self.label())
        # cwd_ipynb_filename = f"{sanitized_label}-{os.path.basename(ipynb_filename)}"
        cwd_ipynb_filename = f"{sanitized_label}.ipynb"

        execution_exception = None

        # The task specification contains a command?
        if "command" in self.specification()["type_specific_tbd"]:
            try:
                # Execute the notebook from within a shell script.
                await self.execute_in_script(notebook, cwd_ipynb_filename)
            except Exception as exception:
                logger.warning(
                    explain2(exception, "executing notebook in a shell script"),
                    exc_info=exception,
                )
                execution_exception = exception
            finally:
                # Read the notebook which has been modified in place.
                logger.debug(
                    f"reading executed in-place notebook from {cwd_ipynb_filename}"
                )
                with open(cwd_ipynb_filename, "r") as stream:
                    notebook = nbformat.read(stream, as_version=4)
        else:
            try:
                self.execute_direct(notebook, cwd_ipynb_filename)
            except Exception as exception:
                logger.warning(
                    explain2(exception, "executing notebook directly"),
                    exc_info=exception,
                )
                execution_exception = exception

            finally:
                # Always save the processed notebook.
                # If it had an error, it will contain partial results and
                # the error message about the failing cell.
                sanitized_label = sanitize(self.label())
                cwd_ipynb_filename = f"{sanitized_label}.ipynb"
                nbformat.write(notebook, cwd_ipynb_filename)

        # Export the processed notebook to html.
        html_exporter = HTMLExporter(template_name="classic")

        logger.debug("exporting html to memory")
        (body, resources) = html_exporter.from_notebook_node(notebook)

        # Save the html formatted notebook.
        html_filename = os.path.splitext(cwd_ipynb_filename)[0]
        html_filename = f"{html_filename}.html"

        with open(html_filename, "w", encoding="utf-8") as stream:
            stream.write(body)

        # The output filenames will be a candidates for catalog attachments.
        # Presume we are operating in our private task directory
        runtime_directory = os.getcwd()
        self.propose_artefact(f"{runtime_directory}/{cwd_ipynb_filename}")
        self.propose_artefact(f"{runtime_directory}/{html_filename}")

        logger.debug("finished exporting to html")

        # If there were errors executing the notebook,
        # raise here to be caught by the main_isolated.
        if execution_exception is not None:
            raise execution_exception

        # Exit code by this time is always a success.
        return 0

    # ------------------------------------------------------------------------------------------
    def execute_direct(self, notebook, cwd_ipynb_filename):
        """ """
        # Jupyter.
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        logger.debug(
            describe("constructing direct ExecutePreprocessor for input", notebook)
        )
        # TODO: Consider if we want to make jupyter ExecutePreprocessor timeput configurable.
        execution_processor = ExecutePreprocessor(timeout=None, kernel_name="python3")

        try:
            logger.debug("preprocessing with no timeout")
            execution_processor.preprocess(notebook, {})
            logger.debug("preprocessing finished with no error")
        finally:
            nbformat.write(notebook, cwd_ipynb_filename)

    # ------------------------------------------------------------------------------------------
    async def execute_in_script(self, notebook, cwd_ipynb_filename):
        """
        Execute the notebook by running the provided command in a shell of its own.
        This is necessary if the jupyter notebook needs a different version of python than bxflow itself.
        Or if it needs other special preparations not work encoding in to a BxTask overload.
        """
        # Jupyter.
        import nbformat

        # Write the notebook to disk so we can execute it from the shell script.
        nbformat.write(notebook, cwd_ipynb_filename)

        specification = copy.deepcopy(self.specification())

        # Modify the command to give the notebook filename.
        command = specification["type_specific_tbd"]["command"]
        if not isinstance(command, list):
            command = [command]
        command.append(cwd_ipynb_filename)

        specification["type_specific_tbd"]["command"] = command

        # Make a bxtask to run a shell command.
        specification["type"] = "dls_bxflow_run.bx_tasks.shell"
        shell_task = BxTasks().build_object(specification, predefined_uuid=self.uuid())
        shell_task.set_directory(self.get_directory())

        exit_code = await shell_task.run()

        if exit_code != 0:
            raise RuntimeError(f"error executing jupyter notebook {cwd_ipynb_filename}")

        return exit_code

    # ------------------------------------------------------------------------------------------
    def extract_error_lines(self):
        """
        Get error lines from log file produced by the task.
        Here there are two cases:
        1. launch from script: the output file has been written by the nbconvert command.
        2. launch internally: the output file has been written by bxflow using dls_logformatter.
        """

        # Specification contains a command?
        if "command" in self.specification()["type_specific_tbd"]:
            # Let the command execute the notebook.
            return self._extract_error_lines_from_script()
        else:
            # This task expects to run in python setup using dls-logformatter.
            # Use the base-class method to get error lines from logging formatter logs.
            return self.extract_error_lines_from_dls_logformatter()

    # ------------------------------------------------------------------------------------------
    def _extract_error_lines_from_script(self):
        """
        Get task post-run fields after the task finished running.
        """

        # For clarity.
        bx_task = self

        # Runtime directory where the isolated task wrote its files.
        runtime_directory = bx_task.get_directory()

        logger.debug(f"[EXTRLIN] runtime_directory {runtime_directory}")
        # Name of of the stderr filename.
        # TODO: Centralize naming convention of stderr.txt filename.
        filename = f"{runtime_directory}/shell_stderr.txt"

        # Died before getting to shell?
        # For example the assertions in the workflow run before running the executable.
        if not os.path.exists(filename):
            logger.debug(f"[EXTRLIN] {filename} doesn't exist, using dls_logformatter")
            return self.extract_error_lines_from_dls_logformatter()

        logger.debug(
            f"[EXTRLIN] {filename} exists, using _extract_error_lines_from_script_std_file"
        )
        return self._extract_error_lines_from_script_std_file(filename)

    # ------------------------------------------------------------------------------------------
    def _extract_error_lines_from_script_std_file(self, filename):
        """
        Try to find the pertinent error in the output file.
        """

        error_lines = []
        if not os.path.exists(filename):
            error_lines.append(f"could not find script std file {filename}")
        else:
            try:
                # Open and read the first chunk of it.
                with open(filename, "r") as stream:
                    first_line = stream.readline()
                    error_lines.append(first_line.strip())

                logger.debug(f"[EXTRLIN] first line is {first_line}")
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
