import asyncio
import json
import logging
import multiprocessing

# ZMQ.
from dls_pairstream_lib.pairstream import Data as PairstreamData
from dls_pairstream_lib.pairstream import new_ReaderInterface, new_WriterInterface
from dls_utilpack.callsign import callsign
from dls_utilpack.explain import explain2

# Utilities.
from dls_utilpack.require import require

# Base class which maps flask requests to methods.
from dls_bxflow_lib.bx_news.base import Base

logger = logging.getLogger(__name__)


thing_type = "dls_bxflow_lib.bx_news.zmq_pubsub"


class ZmqPubsub(Base):
    """
    Object representing a zmq bx_news connection.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        Base.__init__(self, thing_type, specification)

        type_specific_tbd = require(
            f"{thing_type} configuration", self.specification(), "type_specific_tbd"
        )

        self.__producer_configuration = require(
            f"{thing_type} configuration", type_specific_tbd, "producer"
        )
        self.__consumer_configuration = require(
            f"{thing_type} configuration", type_specific_tbd, "consumer"
        )

        self.__writer = None
        self.__reader = None

        self.__started_event = multiprocessing.Event()
        self.__request_stop_event = multiprocessing.Event()

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""
        return self.__consumer_configuration["endpoint"]

    # ----------------------------------------------------------------------------------------
    def activate_producer(self):
        """
        Establish writer.
        """

        if self.__writer is None:
            self.__writer = new_WriterInterface(self.__producer_configuration)
            self.__writer.activate()

    # ----------------------------------------------------------------------------------------
    def activate_consumer(self):
        """
        Establish reader.
        """

        if self.__reader is None:
            self.__reader = new_ReaderInterface(self.__consumer_configuration)
            self.__reader.activate()

    # ----------------------------------------------------------------------------------------
    def produce(self, topic, headline, details):
        """
        Produce a message.
        """

        self.activate_producer()

        data = PairstreamData(bytearray(0))

        logger.debug(f"{callsign(self)} produced {headline}")

        self.__writer.write(
            {"topic": topic, "headline": headline, "details": json.dumps(details)}, data
        )

    # ----------------------------------------------------------------------------------------
    async def consume(self, consumer_callback):
        """
        Start consuming, blocking, with callback on each received message.
        """

        try:
            self.activate_consumer()
            # TODO: Eliminate need for sleep after activating consumer in zmq_pubsub news.
            await asyncio.sleep(0.200)
            self.__started_event.set()

            while True:
                if self.__reader is None:
                    break
                if self.__request_stop_event.is_set():
                    break
                meta = {}
                data = PairstreamData()
                self.__reader.read(meta, data)
                if data.memoryview is not None:
                    headline = meta["headline"]
                    logger.debug(f"{callsign(self)} consumed {headline}")
                    await consumer_callback(
                        meta["topic"], headline, json.loads(meta["details"])
                    )

                # Tiny sleep to give control to event loop.
                # TODO: Convert bx_news zmq_pubsub to use zmq as asyncio.
                await asyncio.sleep(0.010)
        except Exception as exception:
            logger.error(
                explain2(exception, f"running {callsign(self)} consumer loop"),
                exc_info=exception,
            )

    # ----------------------------------------------------------------------------------------
    async def request_stop(self):
        """
        Request to stop consuming, non blocking.
        """

        self.__request_stop_event.set()

    # ----------------------------------------------------------------------------------------
    def disconnect(self):
        """
        Disconnect (client or server) from the socket.
        Releasing these references lets the pairstream destructors release the sockets.
        """

        self.__reader = None
        self.__writer = None
