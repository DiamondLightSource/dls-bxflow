import logging
import os

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Base class for a bx_task.
from dls_bxflow_run.bx_tasks.base import Base
from dls_bxflow_run.bx_tasks.constants import Types as BxTaskTypes

# States.
from dls_bxflow_run.bx_tasks.states import States

logger = logging.getLogger(__name__)

thing_type = BxTaskTypes.SYMLINK


class Symlink(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.state(States.PREPARED)

        self.__modify_cells = None

    # ----------------------------------------------------------------------------------------
    async def run(self):
        """
        Run this task by simply cloning the outputs from a previously run task.
        """

        type_specific_tbd = require(
            f"{callsign(self)} specification", self.specification(), "type_specific_tbd"
        )

        source_directory = require(
            f"{callsign(self)} specification", type_specific_tbd, "source_directory"
        )

        # List files (base name only) in the current directory.
        filenames = os.listdir(source_directory)

        target_directory = self.get_directory()

        for filename in filenames:
            target_filename = f"{target_directory}/{filename}"
            # The running task doesn't already have this?
            # For example, .bxflow which was created for isolated running.
            if not os.path.exists(target_filename):
                os.symlink(f"{source_directory}/{filename}", target_filename)

        # Exit code by this time is always a success.
        return 0

    # ------------------------------------------------------------------------------------------
    def extract_error_lines(self):
        """
        Get error lines from log file.
        """

        # This task expects to run in python setup using dls-logformatter.
        # Use the base-class method to get error lines from logging formatter logs.
        return self.extract_error_lines_from_dls_logformatter()
