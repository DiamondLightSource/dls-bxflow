import json
import logging
import re

import yaml

# Utilities.
from dls_utilpack.search_file import search_file

# Settings manager.
from dls_bxflow_lib.bx_settings.bx_settings import BxSettings

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class NotebookHelper:
    """
    This workflow takes a notebook.
    """

    def __init__(self, notebook_paths, notebook_name):
        self.__notebook_paths = notebook_paths
        self.__notebook_name = notebook_name

    # ------------------------------------------------------------------
    # Method to get settings from notebook cell.
    def get_settings(self, cell_index):
        """
        Get settings from a notebook cell.
        """

        # Make full path to the notebook to run.
        ipynb_filename = search_file(
            self.__notebook_paths,
            f"{self.__notebook_name}.ipynb",
        )

        logger.debug("importing jupyter code")
        # Jupyter.
        import nbformat

        # Read the notebook into memory.
        logger.debug(f"reading notebook from {ipynb_filename}")
        with open(ipynb_filename, "r") as stream:
            notebook = nbformat.read(stream, as_version=4)

        source = notebook["cells"][cell_index]["source"].strip()

        if len(source) == 0:
            raise RuntimeError(
                f"notebook {self.__notebook_name} cell {cell_index} is blank"
            )

        # Replace some markdown things that might be in there.
        source = re.sub(r"^(```yaml)(\n)?", "", source)
        source = re.sub(r"^(```json)(\n)?", "", source)
        source = re.sub(r"^(```)(\n)?", "", source)
        source = re.sub(r"(\n)?(```)$", "", source)

        if source[0] == "{":
            try:
                settings_dicts = json.loads(source)
            except Exception:
                raise RuntimeError(
                    f"notebook {self.__notebook_name} cell {cell_index} failed to parse as json"
                )
        else:
            try:
                settings_dicts = yaml.safe_load(source)
            except Exception:
                raise RuntimeError(
                    f"notebook {self.__notebook_name} cell {cell_index} failed to parse as yaml"
                )

        bx_settings = BxSettings(f"{self.__notebook_name} cell {cell_index}")
        bx_settings.load_from_dicts(settings_dicts)

        return bx_settings

    # ------------------------------------------------------------------
    # Override the method which returns the job's desired label.
    def get_job_label(self):
        return self.__notebook_name
