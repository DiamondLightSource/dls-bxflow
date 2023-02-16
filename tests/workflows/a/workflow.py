import logging

# Workflow base class.
from dls_bxflow_lib.bx_workflows.base import Base as BxWorkflowBase
from dls_bxflow_lib.bx_workflows.main import Main

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class A(BxWorkflowBase):
    """
    This workflow has one task.
    """

    # Describe the inputs to the workflow's constructor.
    # These also become:
    # - attributes of the workflow object
    # - command line arguments to the workflow's main
    # - gui input fields
    constructor_kwargs = {"outfile": None}

    # ------------------------------------------------------------------------------------
    def build(self):
        """
        Build a workflow by chaining tasks together.
        """

        # if not hasattr(self, "outfile"):
        #     self.outfile = None
        logger.info(f"outfile is {self.outfile}")
        self.add_dummy_task(
            "A", outfile=self.outfile, remex_hints="workflow_A_dummy_task"
        )


# ---------------------------------------------------------------
# From command line, invoke the main method.
if __name__ == "__main__":

    # Instantiate and run workflow.
    Main(A)
