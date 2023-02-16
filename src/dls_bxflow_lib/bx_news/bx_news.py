# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_bx_news = None


def bx_news_set_default(bx_news):
    global __default_bx_news
    __default_bx_news = bx_news


def bx_news_get_default():
    global __default_bx_news
    if __default_bx_news is None:
        raise RuntimeError("bx_news_get_default instance is None")
    return __default_bx_news


# -----------------------------------------------------------------------------------------
# Default specification for applications which want news from the infrastructure.
__default_bx_news_specification = None


def bx_news_set_default_specification(bx_news_specification):
    global __default_bx_news_specification
    __default_bx_news_specification = bx_news_specification


def bx_news_get_default_specification():
    global __default_bx_news_specification
    if __default_bx_news_specification is None:
        raise RuntimeError("bx_news_get_default_specification instance is None")
    return __default_bx_news_specification


# -----------------------------------------------------------------------------------------


class BxNews(Things):
    """
    List of available bx_news.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification):
        """"""

        bx_news_class = self.lookup_class(specification["type"])

        try:
            bx_news_object = bx_news_class(specification)
        except Exception as exception:
            raise RuntimeError(
                "unable to build bx_news object for type %s" % (bx_news_class)
            ) from exception

        return bx_news_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_news.aiohttp":
            from dls_bxflow_lib.bx_news.aiohttp import Aiohttp

            return Aiohttp

        if class_type == "dls_bxflow_lib.bx_news.aio_pika":
            from dls_bxflow_lib.bx_news.aio_pika import Aiopika

            return Aiopika

        if class_type == "dls_bxflow_lib.bx_news.pika":
            from dls_bxflow_lib.bx_news.pika import Pika

            return Pika

        if class_type == "dls_bxflow_lib.bx_news.zmq_pubsub":
            from dls_bxflow_lib.bx_news.zmq_pubsub import ZmqPubsub

            return ZmqPubsub

        raise NotFound("unable to get bx_news class for type %s" % (class_type))
