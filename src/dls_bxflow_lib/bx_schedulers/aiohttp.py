import asyncio
import logging
import multiprocessing
import threading

# Utilities.
from dls_utilpack.callsign import callsign

# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Base class which maps flask bx_tasks to methods.
from dls_bxflow_api.thing import Thing

# Base class for an aiohttp server.
from dls_bxflow_lib.base_aiohttp import BaseAiohttp

# News object.
from dls_bxflow_lib.bx_news.bx_news import BxNews, bx_news_get_default
from dls_bxflow_lib.bx_news.constants import Topics

# BxScheduler manager.
from dls_bxflow_lib.bx_schedulers.bx_schedulers import BxSchedulers

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_schedulers.aiohttp"


# ------------------------------------------------------------------------------------------
class Aiohttp(Thing, BaseAiohttp):
    """
    Object implementing remote procedure calls for bx_scheduler methods.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Thing.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)
        BaseAiohttp.__init__(
            self,
            specification["type_specific_tbd"]["aiohttp_specification"],
            calling_file=__file__,
        )

        self.__actual_bx_scheduler = None

        self.__bx_news = None
        self.__bx_news_consumer_future = None

        # This flag will stop the ticking async task.
        self.__keep_ticking = True
        self.__tick_future = None

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""
        return "%s %s" % ("BxScheduler.Aiohttp", BaseAiohttp.callsign(self))

    # ----------------------------------------------------------------------------------------
    def activate_process(self):
        """"""

        try:
            multiprocessing.current_process().name = "bx_scheduler"

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
            threading.current_thread().name = "bx_scheduler"

            self.activate_thread_base(loop)

        except Exception as exception:
            logger.exception(
                f"unable to start {callsign(self)} thread", exc_info=exception
            )

    # ----------------------------------------------------------------------------------------
    async def activate_coro(self):
        """"""
        try:

            # ----------------------------------------------
            # Build the actual launcher.
            self.__actual_bx_scheduler = BxSchedulers().build_object(
                self.specification()["type_specific_tbd"][
                    "actual_bx_scheduler_specification"
                ]
            )

            # Get the local implementation started.
            await self.__actual_bx_scheduler.activate()

            # ----------------------------------------------
            # No special routes, we will use protocolj dispathcing only
            route_tuples = []

            # Build a bx_news to listen for events.
            # This has to be a new instance because we will run a receive loop on it.
            self.__bx_news = BxNews().build_object(
                bx_news_get_default().specification()
            )

            # Start a news consumer receive loop.
            self.__bx_news_consumer_future = asyncio.create_task(
                self.__bx_news.consume(self.__consume_bx_news)
            )
            await BaseAiohttp.activate_coro_base(self, route_tuples)

            # self.__tick_future = asyncio.get_event_loop().create_task(self.tick())

            # Match once when we come up to sync with any launchers already running.
            await self.match("initializing")

        except Exception:
            raise RuntimeError(f"unable to start {callsign(self)} server coro")

    # ----------------------------------------------------------------------------------------
    async def direct_shutdown(self):
        """"""

        if self.__actual_bx_scheduler is not None:
            # Disconnect our local dataface connection, i.e. the one which holds the database connection.
            await self.__actual_bx_scheduler.deactivate()

        if self.__tick_future is not None:
            # Set flag to stop the periodic ticking.
            self.__keep_ticking = False
            # Wait for the ticking to stop.
            await self.__tick_future

        if self.__tick_future is not None:
            # Set flag to stop the periodic ticking.
            self.__keep_ticking = False
            # Wait for the ticking to stop.
            await self.__tick_future

        # We are not running in our own event loop?
        if self.owned_event_loop2 is not None:
            # Disconnect from dataface we have been using.
            await bx_datafaces_get_default().close_client_session()

        if self.__bx_news is not None:
            # Disconnect from news server we have been using.
            # This will set the flag to stop the consumer receive loop.
            await self.__bx_news.request_stop()

            # Stop the asyncio task whcih is listening for news.
            if self.__bx_news_consumer_future is not None:
                # logger.info("waiting for consumuer future to stop")
                await self.__bx_news_consumer_future
                self.__bx_news_consumer_future = None

        # Let the base class stop the server looping.
        await self.base_direct_shutdown()

    # ----------------------------------------------------------------------------------------
    async def __consume_bx_news(self, topic, headline, payload):
        """ """

        # One of the topics which can release a task for execution?
        if (
            topic == Topics.BXJOB_WAS_ENABLED
            or topic == Topics.BXGATE_WAS_OPENED
            or topic == Topics.BXLAUNCHER_WAS_UPDATED
        ):
            await self.match(f"handling {headline} news")

    # ----------------------------------------------------------------------------------------
    async def tick(self):
        """
        Periodic ticking to check for new work.
        """

        while self.__keep_ticking:
            await asyncio.sleep(1.0)
            logger.debug("tick")
            await self.__actual_bx_scheduler.match("ticking")

    # ----------------------------------------------------------------------------------------
    async def match(self, doing):
        """
        Match tasks to launchers.
        """

        try:
            await self.__actual_bx_scheduler.match()
        except Exception as exception:
            logger.warning(
                f"{callsign(self)} match failed {doing}: {str(exception)}",
                exc_info=exception,
            )
