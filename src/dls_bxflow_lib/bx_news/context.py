import asyncio
import logging

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.ignore import ignore

# Base class for an asyncio context
from dls_bxflow_lib.bx_contexts.base import Base as ContextBase

# Things created in the context.
from dls_bxflow_lib.bx_news.bx_news import BxNews, bx_news_set_default

logger = logging.getLogger(__name__)


thing_type = "dls_bxflow_lib.bx_news.context"


class Context(ContextBase):
    """
    Asyncio context for a bx_news object.
    On entering, it creates the object according to the specification (a dict).
    If configured, it starts the server as a coroutine, thread or process.
    On exiting, it commands the server to shut down and closes client connection.

    The enter and exit methods are exposed for use during testing.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification):
        ContextBase.__init__(self, thing_type, specification)

        self.__bx_news_consumer_future = None

    # ----------------------------------------------------------------------------------------
    async def aenter(self):
        """ """

        # Build the object according to the specification.
        self.server = BxNews().build_object(self.specification())

        # If there is more than one news, the last one defined will be the default.
        bx_news_set_default(self.server)

        if self.context_specification.get("start_as") == "coro":
            await self.server.activate_coro()

        elif self.context_specification.get("start_as") == "thread":
            await self.server.start_thread()

        elif self.context_specification.get("start_as") == "process":
            await self.server.start_process()

    # ----------------------------------------------------------------------------------------
    async def aexit(self):
        """ """

        if self.server is not None:
            start_as = self.context_specification.get("start_as")
            if start_as in ["coro", "thread", "process"]:
                # Put in request to shutdown the server.
                await self.server.client_shutdown()
            else:
                await self.server.close_client_session()

        # Since we may have been producing news, request to stop the producer looping.
        try:
            await self.server.disconnect_producer()
        except Exception as exception:
            ignore(logger, exception, "client disconnecting producer")

        if self.__bx_news_consumer_future is not None:
            await self.__bx_news_consumer_future
            self.__bx_news_consumer_future = None

        # Clear the global variable.  Important between pytests.
        bx_news_set_default(None)

    # ----------------------------------------------------------------------------------------
    def add_news_consumer(self, consumer_callback):
        """ """

        # TODO: Figure out BxContext add_news_consumer for multiple consumers.
        if self.__bx_news_consumer_future is not None:
            raise RuntimeError(
                f"{callsign(self)} does not support multiple news consumers at this time"
            )

        self.__bx_news_consumer_future = asyncio.create_task(
            self.server.consume(consumer_callback)
        )
