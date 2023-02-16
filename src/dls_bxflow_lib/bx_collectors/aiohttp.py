import asyncio
import logging
import multiprocessing
import threading

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Base class which maps flask bx_tasks to methods.
from dls_bxflow_api.thing import Thing

# Base class for an aiohttp server.
from dls_bxflow_lib.base_aiohttp import BaseAiohttp

# Factory to make a BxCollector.
from dls_bxflow_lib.bx_collectors.bx_collectors import BxCollectors

# BxCollector protocolj things.
from dls_bxflow_lib.bx_collectors.constants import Commands, Keywords

# News object.
from dls_bxflow_lib.bx_news.bx_news import BxNews, bx_news_get_default
from dls_bxflow_lib.bx_news.constants import Topics

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_collectors.aiohttp"


# ------------------------------------------------------------------------------------------
class Aiohttp(Thing, BaseAiohttp):
    """
    Object representing a bx_collector which receives bx_tasks from aiohttp.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Thing.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)
        BaseAiohttp.__init__(
            self,
            specification["type_specific_tbd"]["aiohttp_specification"],
            calling_file=__file__,
        )

        self.__bx_news = None
        self.__bx_news_consumer_future = None

        self.__actual_bx_collector = None

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""
        return "%s %s" % ("BxCollector.Aiohttp", BaseAiohttp.callsign(self))

    # ----------------------------------------------------------------------------------------
    def activate_process(self):
        """"""

        try:
            multiprocessing.current_process().name = "bx_collector"

            self.activate_process_base()

        except Exception as exception:
            logger.exception("exception in bx_collector process", exc_info=exception)

    # ----------------------------------------------------------------------------------------
    def activate_thread(self, loop):
        """
        Called from inside a newly created thread.
        """

        try:
            threading.current_thread().name = "bx_collector"

            self.activate_thread_base(loop)

        except Exception as exception:
            logger.exception(
                f"unable to start {callsign(self)} thread", exc_info=exception
            )

    # ----------------------------------------------------------------------------------------
    async def activate_coro(self):
        """"""
        try:
            # Build a local bx_collector for our back-end.
            self.__actual_bx_collector = BxCollectors().build_object(
                self.specification()["type_specific_tbd"][
                    "actual_bx_collector_specification"
                ]
            )

            # Get the local implementation started.
            await self.__actual_bx_collector.activate()

            # ----------------------------------------------
            # Build a bx_news to listen for events.
            # This has to be a new instance because we will run a receive loop on it.
            self.__bx_news = BxNews().build_object(
                bx_news_get_default().specification()
            )

            # Start a news consumer receive loop.
            self.__bx_news_consumer_future = asyncio.create_task(
                self.__bx_news.consume(self.__consume_bx_news)
            )

            logger.info("[COLNEWS] listening for news")

            # ----------------------------------------------
            await BaseAiohttp.activate_coro_base(self)

        except Exception as exception:
            raise RuntimeError(
                "exception while starting bx_collector server"
            ) from exception

    # ----------------------------------------------------------------------------------------
    async def direct_shutdown(self):
        """"""

        # ----------------------------------------------
        if self.__actual_bx_collector is not None:
            # Disconnect our local dataface connection, i.e. the one which holds the database connection.
            await self.__actual_bx_collector.deactivate()

        # ----------------------------------------------
        if self.__bx_news is not None:
            # Disconnect from news server we have been using.
            # This will set the flag to stop the consumer receive loop.
            await self.__bx_news.request_stop()

            # Stop the asyncio task whcih is listening for news.
            if self.__bx_news_consumer_future is not None:
                # logger.info("waiting for consumuer future to stop")
                await self.__bx_news_consumer_future
                self.__bx_news_consumer_future = None

        # ----------------------------------------------
        # Let the base class stop the server listener.
        await self.base_direct_shutdown()

    # ----------------------------------------------------------------------------------------
    async def __consume_bx_news(self, topic, headline, payload):
        """ """

        # Job was deleted?
        if topic == Topics.BXJOB_WAS_DELETED:
            try:
                await self.__actual_bx_collector.job_was_deleted(payload)
            except Exception as exception:
                logger.warning(
                    f"some error responding to news {headline}",
                    exc_info=exception,
                )

    # ----------------------------------------------------------------------------------------
    # From http client, request server to submit bx_task for execution.

    # async def fire(self, message):
    #     """"""
    #     # Build a local bx_collector for our client side.
    #     actual_bx_collector = BxCollectors().build_object(
    #         self.specification()["type_specific_tbd"][
    #             "actual_bx_collector_specification"
    #         ]
    #     )

    #     logger.debug(f"[DMOTF] firing actual {callsign(actual_bx_collector)}")
    #     await actual_bx_collector.fire(message)
    #     logger.debug("[DMOTF] firing complete")

    # ----------------------------------------------------------------------------------------
    async def fire(self, message):
        """"""
        return await self.__send_protocolj("fire", message)

    # ----------------------------------------------------------------------------------------
    async def __send_protocolj(self, function, *args, **kwargs):
        """"""

        return await self.client_protocolj(
            {
                Keywords.COMMAND: Commands.EXECUTE,
                Keywords.PAYLOAD: {
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

        function = getattr(self.__actual_bx_collector, function)

        response = await function(*args, **kwargs)

        return response

    # ----------------------------------------------------------------------------------------
    async def dispatch(self, request_dict, opaque):
        """"""

        command = require("request json", request_dict, Keywords.COMMAND)

        if command == Commands.EXECUTE:
            payload = require("request json", request_dict, Keywords.PAYLOAD)
            response = await self.__do_locally(
                payload["function"], payload["args"], payload["kwargs"]
            )
        else:
            raise RuntimeError("invalid command %s" % (command))

        return response
