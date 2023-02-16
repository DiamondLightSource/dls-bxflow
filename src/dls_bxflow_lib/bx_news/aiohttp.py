import logging
import multiprocessing
import threading

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Basic things.
from dls_bxflow_api.thing import Thing

# Base class for an aiohttp server.
from dls_bxflow_lib.base_aiohttp import BaseAiohttp

# News object manager.
from dls_bxflow_lib.bx_news.bx_news import BxNews

# News protocolj things.
from dls_bxflow_lib.bx_news.constants import Commands, Keywords

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_news.aiohttp"


# ------------------------------------------------------------------------------------------
class Aiohttp(BaseAiohttp, Thing):
    """
    Object implementing remote procedure calls for bx_news methods.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        Thing.__init__(self, thing_type, specification)
        BaseAiohttp.__init__(
            self,
            specification["type_specific_tbd"]["aiohttp_specification"],
            calling_file=__file__,
        )
        self.__local_bx_news = None

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""
        return "%s %s" % ("BxNews.Aiohttp", BaseAiohttp.callsign(self))

    # ----------------------------------------------------------------------------------------
    def activate_process(self):
        """"""

        try:
            multiprocessing.current_process().name = "bx_news"

            self.activate_process_base()

        except Exception as exception:
            logger.exception(
                f"unable to start {callsign(self)} process", exc_info=exception
            )

    # ----------------------------------------------------------------------------------------
    def activate_thread(self, loop):
        """
        Called from inside a newly created thread.
        """

        try:
            threading.current_thread().name = "bx_news"

            self.activate_thread_base(loop)

        except Exception as exception:
            logger.exception(
                f"unable to start {callsign(self)} thread", exc_info=exception
            )

    # ----------------------------------------------------------------------------------------
    async def activate_coro(self):
        """"""

        try:
            # No special routes, we will use protocolj dispathcing only
            route_tuples = []

            # Build a local bx_news for out back-end.
            self.__local_bx_news = BxNews().build_object(
                self.specification()["type_specific_tbd"]["local_bx_news_specification"]
            )

            # Start the http server.
            # Must be called before activating the zmq producer in caes it has to kill an old process.
            await self.activate_coro_base(route_tuples)

            # Get the local implementation started.
            self.__local_bx_news.activate_producer()

        except Exception:
            raise RuntimeError(f"unable to start {callsign(self)} server coro")

    # ----------------------------------------------------------------------------------------
    async def disconnect_producer(self):
        """"""

        if self.__local_bx_news is not None:
            self.__local_bx_news.disconnect()

    # ----------------------------------------------------------------------------------------
    async def direct_shutdown(self):
        """"""

        # Disconnect from our private client to the news implementation.
        self.__local_bx_news.disconnect()

        # Let the base class stop the server listener.
        await Aiohttp.base_direct_shutdown(self)

    # ----------------------------------------------------------------------------------------
    async def consume(self, consumer_callback):
        """
        Start consuming, blocking, with callback on each received message.
        """

        # No local object built for client side yet?
        if self.__local_bx_news is None:
            self.__local_bx_news = BxNews().build_object(
                self.specification()["type_specific_tbd"]["local_bx_news_specification"]
            )

        await self.__local_bx_news.consume(consumer_callback)

    # ----------------------------------------------------------------------------------------
    async def request_stop(self):
        """
        Request top consuming, blocking, with callback on each received message.
        """

        # No local object built for client side yet?
        if self.__local_bx_news is None:
            self.__local_bx_news = BxNews().build_object(
                self.specification()["type_specific_tbd"]["local_bx_news_specification"]
            )

        await self.__local_bx_news.request_stop()

    # ----------------------------------------------------------------------------------------
    async def produce(self, topic, headline, details):
        """"""
        return await self.__send_protocolj("produce", topic, headline, details)

    # ----------------------------------------------------------------------------------------
    async def test_uncaught(self, test_message):
        """"""
        return await self.__send_protocolj("test_uncaught", test_message)

    # ----------------------------------------------------------------------------------------
    async def __send_protocolj(self, function, *args, **kwargs):
        """"""

        return await self.client_protocolj(
            {
                Keywords.COMMAND: Commands.EXECUTE,
                Keywords.DETAILS: {
                    "function": function,
                    "args": args,
                    "kwargs": kwargs,
                },
            },
        )

    # ----------------------------------------------------------------------------------------
    async def __do_locally(self, function, args, kwargs):
        """"""

        # logger.info(describe("function", function))
        # logger.info(describe("args", args))
        # logger.info(describe("kwargs", kwargs))

        function = getattr(self.__local_bx_news, function)

        response = function(*args, **kwargs)

        return response

    # ----------------------------------------------------------------------------------------
    async def dispatch(self, request_dict, opaque):
        """"""

        command = require("request json", request_dict, Keywords.COMMAND)

        if command == Commands.EXECUTE:
            details = require("request json", request_dict, Keywords.DETAILS)
            response = await self.__do_locally(
                details["function"], details["args"], details["kwargs"]
            )
        else:
            raise RuntimeError("invalid command %s" % (command))

        return response
