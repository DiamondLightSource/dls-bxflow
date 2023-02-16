# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_bx_collector = None


def bx_collectors_set_default(bx_collector):
    global __default_bx_collector
    __default_bx_collector = bx_collector


def bx_collectors_get_default():
    global __default_bx_collector
    if __default_bx_collector is None:
        raise RuntimeError("bx_collectors_get_default instance is None")
    return __default_bx_collector


class BxCollectors(Things):
    """
    List of available bx_collectors.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name="bx_collectors"):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification, predefined_uuid=None):
        """"""

        bx_collector_class = self.lookup_class(specification["type"])

        try:
            bx_collector_object = bx_collector_class(
                specification, predefined_uuid=predefined_uuid
            )
        except Exception as exception:
            raise RuntimeError(
                "unable to build bx_collector object of class %s"
                % (bx_collector_class.__name__)
            ) from exception

        return bx_collector_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_collectors.aiohttp":
            from dls_bxflow_lib.bx_collectors.aiohttp import Aiohttp

            return Aiohttp

        elif class_type == "dls_bxflow_lib.bx_collectors.gdascan":
            from dls_bxflow_lib.bx_collectors.gdascan import Gdascan

            return Gdascan

        elif class_type == "dls_bxflow_lib.bx_collectors.manual":
            from dls_bxflow_lib.bx_collectors.manual import Manual

            return Manual

        elif class_type == "dls_bxflow_lib.bx_collectors.scraper":
            from dls_bxflow_lib.bx_collectors.scraper import Scraper

            return Scraper

        else:
            try:
                RuntimeClass = Things.lookup_class(self, class_type)
                return RuntimeClass
            except NotFound:
                raise NotFound("unable to get bx_collector class for %s" % (class_type))
