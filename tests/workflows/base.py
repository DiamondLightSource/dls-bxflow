import logging

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Filestore interface.
from dls_bxflow_lib.bx_filestores.bx_filestores import bx_filestores_get_default

# Object managers.
from dls_bxflow_lib.bx_workflows.base import Base as BxWorkflowBase

# Output location convention.
from tests.workflows.utilities import data_label_2_filestore_directory

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------


class Base(BxWorkflowBase):
    """
    This class provides some test-specific helper methods.
    """

    # ------------------------------------------------------------------
    def __init__(self, **kwargs):
        # On this test rig, all workflows should have a data_label in the kwargs.
        data_label = require("workflow kwargs", kwargs, "data_label")

        # Derive a place to put the output files from the job execution.
        # This usually uses the data_label to save output files in a similar location to the orignal data file.
        # This needs configurator to give current beamline, year and visit.
        filestore_directory = data_label_2_filestore_directory(data_label)

        # Tell the workflow builder where the output files should go.
        bx_filestores_get_default().set_directory(filestore_directory)

        logger.info(
            f"{callsign(self)} sets filestore_directory to {filestore_directory}"
        )

        # -----------------------------------------------------------------------------
        # Init the base class only AFTER the filestore_directory is set.
        BxWorkflowBase.__init__(self, **kwargs)

        # Modify the job's data_label.
        self.bx_job.set_data_label(self.data_label)
