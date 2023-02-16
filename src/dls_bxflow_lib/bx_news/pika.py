import json
import logging
import threading

import pika

# Utilities.
from dls_utilpack.describe import describe
from dls_utilpack.explain import explain
from dls_utilpack.require import require

# Base class which maps flask requests to methods.
from dls_bxflow_lib.bx_news.base import Base

logger = logging.getLogger(__name__)


thing_type = "dls_bxflow_lib.bx_news.pika"


class Pika(Base):
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
        self.__queues = None

    # ----------------------------------------------------------------------------------------
    def __establish_channel_queue(self, queue):
        """
        Establish channel.
        """

        if self.__connection is None:
            self.__connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.__host)
            )

        if self.__channel is None:
            self.__channel = self.__connection.channel()
            self.__queues = []

        if queue not in self.__queues:
            self.__queues.append(queue)
            self.__channel.queue_declare(queue=queue)

    # ----------------------------------------------------------------------------------------
    def produce(self, queue, message):
        """
        Produce a message.
        """

        self.__establish_channel_queue(queue)

        self.__channel.basic_publish(
            exchange="", routing_key=queue, body=json.dumps(message)
        )

        logger.info(describe(f"sent message on {queue} which", message))

    # ----------------------------------------------------------------------------------------
    def add_consumer_queue(self, queue, callback):
        """
        Attach callback to incoming messages.
        """

        self.__establish_channel_queue(queue)

        self.__channel.basic_consume(
            queue=queue, on_message_callback=callback, auto_ack=True
        )

    # ----------------------------------------------------------------------------------------
    def start_consuming(self):
        """
        Start consuming, non blocking.
        """

        self.__thread = threading.Thread(target=self.__channel.start_consuming)
        self.__thread.start()

    # ----------------------------------------------------------------------------------------
    def stop_consuming(self):
        """
        Start consuming, non blocking.
        """

        if self.__connection is not None:
            try:
                self.__connection.close()
            except Exception as exception:
                logger.info(explain(exception, "closing channel"))
            self.__channel = None

            self.__thread.join()
            logger.info("consumer thread joined")
