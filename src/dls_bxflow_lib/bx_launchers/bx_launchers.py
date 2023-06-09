# Use standard logging in this module.
import logging

# Class managing list of things.
from dls_utilpack.things import Things

from dls_bxflow_api.bx_launchers.constants import ClassTypes

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

logger = logging.getLogger(__name__)


class BxLaunchers(Things):
    """
    List of available bx_launchers.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name="bx_launchers"):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification, predefined_uuid=None):
        """"""

        bx_launcher_class = self.lookup_class(specification["type"])

        try:
            bx_launcher_object = bx_launcher_class(
                specification, predefined_uuid=predefined_uuid
            )
        except Exception as exception:
            raise RuntimeError(
                "unable to build bx_launcher object of class %s"
                % (bx_launcher_class.__name__)
            ) from exception

        return bx_launcher_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == ClassTypes.AIOHTTP:
            from dls_bxflow_lib.bx_launchers.aiohttp import Aiohttp

            return Aiohttp

        elif class_type == ClassTypes.ISLAND:
            from dls_bxflow_lib.bx_launchers.island import Island

            return Island

        elif class_type == ClassTypes.POPENER:
            from dls_bxflow_lib.bx_launchers.popener import Popener

            return Popener

        elif class_type == ClassTypes.QSUBBER:
            from dls_bxflow_lib.bx_launchers.qsubber import Qsubber

            return Qsubber

        elif class_type == ClassTypes.SLURMER:
            from dls_bxflow_lib.bx_launchers.slurmer import Slurmer

            return Slurmer

        raise NotFound("unable to get bx_launcher class for %s" % (class_type))
