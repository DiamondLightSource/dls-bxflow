import asyncio
import logging

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing news.
from dls_bxflow_lib.bx_news.bx_news import bx_news_get_default
from dls_bxflow_lib.bx_news.constants import Topics

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestNews:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        NewsTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class NewsTester(BaseContextTester):
    """
    Class to test the news.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()

        # Don't build the higher level servers.
        bx_configurator.remove("bx_filestore_specification")
        bx_configurator.remove("bx_dataface_specification")
        bx_configurator.remove("bx_job_specification")
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_launcher_specifications")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()
        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:
            self.topic = Topics.BXJOB_WAS_ENABLED

            self.__producer_count = 2
            self.__consumer_count = 0

            # Start a producer (sender).
            producer_future = asyncio.create_task(self.produce())

            # Start a news consumer task to do the callbacks.
            await bx_context.add_news_consumer(self.__consume_bx_news)

            await producer_future

            # Wait for the messages to be received within the callback.
            await asyncio.sleep(0.1)

        assert self.__consumer_count == self.__producer_count

    # ----------------------------------------------------------------------------------------
    async def produce(self):

        for i in range(0, self.__producer_count):
            # logger.info(f"producing {i}")
            headline = f"headline {i}"
            await bx_news_get_default().produce(
                self.topic, headline, {"what": f"message {i}"}
            )
            await asyncio.sleep(0.010)

    # ----------------------------------------------------------------------------------------
    async def __consume_bx_news(self, topic, headline, details):
        """
        Callback for news messages.
        """

        # logger.info(describe(f"received {topic} details which", details))

        assert topic == self.topic
        assert headline == f"headline {self.__consumer_count}"
        assert details.get("what") is not None

        self.__consumer_count += 1
