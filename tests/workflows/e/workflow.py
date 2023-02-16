import logging

from dls_bxflow_lib.bx_workflows.main import Main

# Workflow base class.
from tests.workflows.base import Base

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class E(Base):
    """
    This workflow has one task.
    """

    # Describe the inputs to the workflow's constructor.
    # These also become:
    # - attributes of the workflow object
    # - command line arguments to the workflow's main
    # - gui input fields
    constructor_kwargs = {
        # Data label is the basic part of the input data file.
        "data_label": "",
    }

    # ------------------------------------------------------------------
    def __init__(self, **kwargs):
        Base.__init__(self, **kwargs)

        # # Add some cosmetic meta info about the energy setting.
        # self.add_setting(
        # )

    # ------------------------------------------------------------------------------------
    def build(self):
        """
        Build a workflow which sets its data label.
        Used in by CollectorScraperTester.
        """

        logger.info(f'data_label is from kwargs is "{self.data_label}"')

        # Tell the job about the data_label.
        self.bx_job.set_data_label(self.data_label)

        task = self.add_dummy_task("E.task1")

        logger.info(f"task added with directory {task.get_directory()}")


# ---------------------------------------------------------------
# From command line, invoke the main method.
if __name__ == "__main__":

    # Instantiate and run workflow.
    Main(E)
