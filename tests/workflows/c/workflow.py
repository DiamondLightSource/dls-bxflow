import logging

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.describe import describe
from dls_utilpack.require import require

# Objects we may need to access
from dls_bxflow_lib.bx_configurators.bx_configurators import (
    bx_configurators_get_default,
)

# Workflow base class.
from dls_bxflow_lib.bx_workflows.base import Base as BxWorkflowBase
from dls_bxflow_lib.bx_workflows.main import Main
from dls_bxflow_lib.bx_workflows.notebook_helper import NotebookHelper

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class C(BxWorkflowBase):
    """
    This workflow runs a single notebook.
    The name of the notebook is changed to the name of the task.
    """

    # Describe the inputs to the workflow's constructor.
    # These also become:
    # - attributes of the workflow object
    # - command line arguments to the workflow's main
    # - gui input fields
    # Data label is the basic part of the input data file.
    constructor_kwargs = {
        "notebook": "workflow_c_notebook",
        "data_label": "",
    }

    # ------------------------------------------------------------------
    def __init__(self, **kwargs):

        # Get more settings from the notebook cell.
        notebook_helper = NotebookHelper(
            bx_configurators_get_default().require("testing.notebook_paths"),
            require(f"{callsign(self)} constructor kwargs", kwargs, "notebook"),
        )
        notebook_settings = notebook_helper.get_settings(1)

        # Add the settings to the constructor kwargs.
        for setting in notebook_settings.list():
            self.constructor_kwargs[setting.uuid()] = setting.specification()

        logger.debug(describe("self.constructor_kwargs", self.constructor_kwargs))

        BxWorkflowBase.__init__(self, **kwargs)

    # ------------------------------------------------------------------------------------
    def build(self):
        """
        Build a workflow by chaining tasks together.
        """

        # Set job label from notebook's name.
        self.set_job_label(self.notebook)

        # Tell the job about the data_label.
        self.bx_job.set_data_label(self.data_label)

        # Add a task.
        self.add_dummy_task("C.1")


# ---------------------------------------------------------------
# From command line, invoke the main method.
if __name__ == "__main__":

    # Instantiate and run workflow.
    Main(C)
