import logging

# Workflow base class.
from dls_bxflow_lib.bx_workflows.base import Base as BxWorkflowBase
from dls_bxflow_lib.bx_workflows.main import Main

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class B(BxWorkflowBase):
    """
    This workflow has one task.
    """

    # Describe the inputs to the workflow's constructor.
    # These also become:
    # - attributes of the workflow object
    # - command line arguments to the workflow's main
    # - gui input fields
    # Data label is the basic part of the input data file.
    constructor_kwargs = {
        "data_label": "",
    }

    # ------------------------------------------------------------------------------------
    def build(self):
        """
        Build a workflow by chaining tasks together.
        """

        logger.info(f'data_label is from kwargs is "{self.data_label}"')

        # Tell the job about the data_label.
        self.bx_job.set_data_label(self.data_label)

        # Insert the data label to make
        self.add_dummy_task("B")


# ---------------------------------------------------------------
# From command line, invoke the main method.
if __name__ == "__main__":

    # Instantiate and run workflow.
    Main(B)
