import asyncio
import logging

# Utilities.
from dls_utilpack.require import require

# Default for bx_news in the infrastructure.
from dls_bxflow_lib.bx_news.bx_news import BxNews, bx_news_set_default

logger = logging.getLogger(__name__)


class NewsConsumer:
    def __init__(self, configuration):
        self.__configuration = configuration
        self.__bx_news = None
        self.__bx_news_consumer_future = None

    # ----------------------------------------------------------------------------------------
    async def __aenter__(self):
        """ """
        bx_news_specification = require(
            "configuration", self.__configuration, "bx_news_specification"
        )

        self.__bx_news = BxNews().build_object(bx_news_specification)

        bx_news_set_default(self.__bx_news)

        # Start a news consumer receive loop.
        self.__bx_news_consumer_future = asyncio.create_task(
            self.__bx_news.consume(self.__consume_bx_news)
        )

        logger.info("position B")

    # ----------------------------------------------------------------------------------------
    async def __aexit__(self, type, value, traceback):
        """ """
        # Disconnect from news server we have been using.
        # This will set the flag to stop the consumer receive loop.
        await self.__bx_news.request_stop()

        # Stop the asyncio task whcih is listening for news.
        if self.__bx_news_consumer_future is not None:
            # logger.info("waiting for consumuer future to stop")
            await self.__bx_news_consumer_future
            self.__bx_news_consumer_future = None

    # ----------------------------------------------------------------------------------------
    def __consume_bx_news(self, topic, headline, payload):
        """ """

        logger.info(f"news! {topic}")
