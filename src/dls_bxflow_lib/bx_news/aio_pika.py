import json
import logging

import aio_pika
from dls_utilpack.describe import describe
from dls_utilpack.explain import explain

# Utilities.
from dls_utilpack.require import require

# Base class which maps flask requests to methods.
from dls_bxflow_lib.bx_news.base import Base

logger = logging.getLogger(__name__)


thing_type = "dls_bxflow_lib.bx_news.pika"


class QueueAndCallback:
    queue = None
    callback = None


class Aiopika(Base):
    """
    Object representing an event bx_dataface connection.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        Base.__init__(self, thing_type, specification)

        type_specific_tbd = require(
            f"{thing_type} configuration", self.specification(), "type_specific_tbd"
        )

        self.__host = require(f"{thing_type} configuration", type_specific_tbd, "host")

        self.__connection = None
        self.__channel = None
        self.__queues_and_callbacks = None

    # ----------------------------------------------------------------------------------------
    async def __establish_channel_queue(self, queue):
        """
        Establish channel.
        """

        if self.__connection is None:
            self.__connection = await aio_pika.connect_robust(
                f"amqp://guest:guest@{self.__host}/"
            )
            self.__queues_and_callbacks = {}

        if self.__channel is None:
            self.__channel = await self.__connection.channel()
            await self.__channel.set_qos(prefetch_count=10)

        if queue not in self.__queues_and_callbacks:
            queue_and_callback = QueueAndCallback()
            queue_and_callback.queue = await self.__channel.declare_queue(
                queue, auto_delete=False
            )
            self.__queues_and_callbacks[queue] = queue_and_callback

    # ----------------------------------------------------------------------------------------
    async def produce(self, queue, message):
        """
        Produce a message.
        """

        await self.__establish_channel_queue(queue)

        await self.__channel.default_exchange.publish(
            aio_pika.Message(body=json.dumps(message).encode()),
            routing_key=queue,
        )

        logger.info(describe(f"sent message on {queue} which", message))

    # ----------------------------------------------------------------------------------------
    async def add_consumer_queue(self, queue, callback):
        """
        Attach callback to incoming messages.
        """

        await self.__establish_channel_queue(queue)

        self.__queues_and_callbacks[queue].callback = callback

    # ----------------------------------------------------------------------------------------
    async def start_consuming(self):
        """
        Start consuming, non blocking.
        """

        for queue, queue_and_callback in self.__queues_and_callbacks.items():
            break

        while True:
            await queue_and_callback.queue.consume(self.__consume)
            break

        # await self.close_connection()

    # ----------------------------------------------------------------------------------------
    async def close_connection(self):
        """
        Stop consuming.
        """

        if self.__connection is not None:
            try:
                await self.__connection.close()
            except Exception as exception:
                logger.info(explain(exception, "closing connection"))
            self.__connection = None
            self.__channel = None
            self.__queues_and_callbacks = None

    # ----------------------------------------------------------------------------------------
    async def __consume(
        self,
        message,
    ) -> None:
        async with message.process():
            print(message.body)
