import logging
import re

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Base class for filestore things.
from dls_bxflow_lib.bx_filestores.base import Base as BxFilestoreBase

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_filestores.explicit"


class Scandir(BxFilestoreBase):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        BxFilestoreBase.__init__(self, thing_type, specification)

        self.__type_specific_tbd = require(
            f"{callsign(self)} specification", self.specification(), "type_specific_tbd"
        )

        # Pattern for allowable years.
        self.__year_re = re.compile(r"^20[0-9][0-9]$")
        self.__visit_re = re.compile(r"^[a-z][a-z][0-9][0-9][0-9][0-9][0-9][-][0-9]$")

        self.set_scandir(
            require(
                f"{callsign(self)} specification scandir",
                self.__type_specific_tbd,
                "scandir",
            )
        )

    # ----------------------------------------------------------------------------------------
    def set_scandir(self, scandir):

        # Use any scandir given.
        self.__scandir = scandir

        # /dls/i22/data/2022/cm12345-1/i22_000001.nxs.
        parts = scandir.split("/")

        # If properly formatted, then use beamline and visit also.
        if (
            len(parts) >= 6
            and parts[1] == "dls"
            and parts[3] == "data"
            and self.__year_re.match(parts[4])
            and self.__visit_re.match(parts[5])
        ):
            self.set_beamline(parts[2])
            self.set_visit(parts[5])
        # else:
        #     logger.debug(f"[WORKRUN] no visit because parts are %s" % (parts))

    # ----------------------------------------------------------------------------------------
    def get_scandir(self):
        return self.__scandir

    # ----------------------------------------------------------------------------------------
    def get_workflows_anchor(self):
        """ """

        return self.get_scandir()
