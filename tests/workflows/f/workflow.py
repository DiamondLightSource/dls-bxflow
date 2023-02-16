import logging

# Workflow base class.
from dls_bxflow_lib.bx_workflows.base import Base as BxWorkflowBase
from dls_bxflow_lib.bx_workflows.main import Main

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class F(BxWorkflowBase):
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
        # Example of a setting.
        "energy": {
            "type": "dls_bxflow_lib.bx_settings.string",
            "uuid": "energy",
            "prompt": "energy setting",
            "value": "12KeV",
        },
    }

    # ------------------------------------------------------------------
    def __init__(self, **kwargs):
        BxWorkflowBase.__init__(self, **kwargs)

        # # Add some cosmetic meta info about the energy setting.
        # self.add_setting(
        # )

    # ------------------------------------------------------------------------------------
    def build(self):
        """
        Build a workflow which sets its data label.
        Used in by CollectorScraperTester.
        TODO: Move workloads.base.build() into CollectorScraperTester to keep base virtual.
        """

        if self.energy == "":
            raise RuntimeError(f'cannot build workflow for energy "{self.energy}"')

        logger.info(f'data_label is from kwargs is "{self.data_label}"')

        # Tell the job about the data_label.
        self.bx_job.set_data_label(self.data_label)

        task = self.add_dummy_task("F.task1")

        logger.info(f"task added with directory {task.get_directory()}")


# ---------------------------------------------------------------
# From command line, invoke the main method.
if __name__ == "__main__":

    # Instantiate and run workflow.
    Main(F)
