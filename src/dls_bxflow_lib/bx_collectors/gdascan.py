import asyncio
import json
import logging
from typing import Dict
from urllib.parse import urlparse

import stomp
from dls_utilpack.callsign import callsign

# Utilities.
from dls_utilpack.import_class import import_class
from dls_utilpack.require import require, require_environment

# Base class for bx_collector instances.
from dls_bxflow_lib.bx_collectors.base import Base as BxCollectorBase

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_collectors.popener"


class MyListener(stomp.listener.ConnectionListener):
    def __init__(self, event_loop, queue):
        stomp.listener.ConnectionListener.__init__(self)
        self.__event_loop = event_loop
        self.__queue = queue

    def on_error(self, frame):
        logger.error(f"{callsign(self)} got stomp error: %s" % (frame.body))

    def on_message(self, frame):
        try:
            self.__event_loop.call_soon_threadsafe(self.__queue.put_nowait, frame)
        except Exception as exception:
            logger.error(
                f"{callsign(self)} unable to enqueue frame", exc_info=exception
            )


# ------------------------------------------------------------------------------------------
class Gdascan(BxCollectorBase):
    """
    Object representing a bx_collector which launches a task using popen for onboard execution.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification, predefined_uuid=None):
        BxCollectorBase.__init__(
            self, thing_type, specification, predefined_uuid=predefined_uuid
        )

        self.__stomp_server_connection = None

        type_specific_tbd = require(
            "gdascan specification", self.specification(), "type_specific_tbd"
        )

        server = require(
            "gdascan specification type_specific_tbd", type_specific_tbd, "server"
        )
        self.__server_url_parts = urlparse(server)

        client = require(
            "gdascan specification type_specific_tbd", type_specific_tbd, "client"
        )
        self.__client_url_parts = urlparse(client)

        self.__activemq_topic = require(
            "gdascan specification type_specific_tbd",
            type_specific_tbd,
            "activemq_topic",
        )

        self.__queue = None
        self.__gda_parser = None
        self.__monitor_future = None
        self.__monitor_stopper_event = None
        self.__monitor_stopper_task = None

    # ----------------------------------------------------------------------------------------
    async def activate(self):
        """"""
        # Connect to the stomp server.
        self.__stomp_server_connection = stomp.Connection(
            [(self.__server_url_parts.hostname, self.__server_url_parts.port)]
        )

        self.__stomp_server_connection.connect()
        self.__stomp_server_connection.subscribe(
            destination=self.__activemq_topic, id=1, ack="auto"
        )

        # -----------------------------------------------------------------
        # Reference the gda parser section of the specification.

        filename_classname = require_environment("BXFLOW_GDA_PARSER")

        try:
            # Try to import the parser class.
            self.__gda_parser_class = import_class(filename_classname)
        except Exception:
            raise RuntimeError(
                f"value of environment variable BXFLOW_GDA_PARSER is invalid: {filename_classname}"
            )

        # Make a parser object.
        self.__gda_parser = self.__gda_parser_class()

        # -----------------------------------------------------------------
        # https://alfred.diamond.ac.uk/documentation/manuals/GDA_Developer_Guide/master/key_patterns.html?highlight=activemq
        # org.eclipse.scanning.submission.queue
        # org.eclipse.scanning.status.topic
        # org.eclipse.scanning.command.topic
        # org.eclipse.scanning.ack.topic

        # Make an event which we can use to stop the queue monitor when deactivating.
        self.__monitor_stopper_event = asyncio.Event()
        self.__monitor_stopper_task = asyncio.create_task(
            self.__monitor_stopper_event.wait()
        )

        # Queue where the stomp listener can put incoming frames.
        self.__queue = asyncio.Queue()

        # Make a listener which receives stomp messages and enqueues them.
        self.__listener = MyListener(asyncio.get_running_loop(), self.__queue)

        # Monitor to pick up queue entries and spawn the workflow.
        self.__monitor_future = asyncio.create_task(self.monitor_queue())

        # Tell the stomp connection about the listener.
        self.__stomp_server_connection.set_listener("", self.__listener)

        logger.debug(f"[DMOTF] {callsign(self)} listening on stomp topic")

    # ----------------------------------------------------------------------------------------
    async def deactivate(self):
        """"""
        # Disconnect from any further incoming stomp messages.
        if self.__stomp_server_connection is not None:
            self.__stomp_server_connection.disconnect()
            self.__stomp_server_connection = None

        # We have built a queue?
        if self.__queue is not None:
            logger.debug("[DMOTF] pushing final queue entry")
            # Put in a marker to end the monitor loop.
            self.__monitor_stopper_event.set()

            # Wait for the queue to flush.
            logger.debug("[DMOTF] joining queue")
            timeout = 5.0
            try:
                await asyncio.wait_for(self.__queue.join(), timeout)
            except asyncio.TimeoutError:
                logger.warning(f"{callsign(self)} could not join its queue gracefully")

            logger.debug("[DMOTF] joined queue")

            # Wait for the monitor thread to exit.
            if self.__monitor_future is not None:
                await self.__monitor_future

    # ----------------------------------------------------------------------------------------
    async def monitor_queue(self):
        """"""

        logger.info("[DMOTF] inside monitor queue")

        try:
            logger.debug(f"[DMOTF] {callsign(self)} monitoring queue")

            while True:
                getter = asyncio.create_task(self.__queue.get())
                (done, pending) = await asyncio.wait(
                    [getter, self.__monitor_stopper_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if self.__monitor_stopper_task in done:
                    logger.debug(f"[DMOTF] {callsign(self)} exiting queue monitor")
                    break

                frame = getter.result()
                frame = await self.handle_frame(frame)

                self.__queue.task_done()

        except Exception as exception:
            logger.error("error monitoring queue", exc_info=exception)

        logger.debug(f"[DMOTF] {callsign(self)} done monitoring queue")

    # ----------------------------------------------------------------------------------------
    async def handle_frame(self, frame):
        """
        Handle frame which GDA has sent.
        """

        try:
            message = json.loads(frame.body)
        except Exception:
            raise RuntimeError(
                f"{callsign(self)} received a message which is not valid json"
            )

        # Let the external gda parser do the beamline-specific parsing.
        parsed_stuff = self.__gda_parser.parse(message, frame.headers)

        if isinstance(parsed_stuff, str):
            logger.debug("ignoring message because parser said: %s" % (parsed_stuff))
            return

        workflow_filename_classname = require(
            "gda_message", parsed_stuff, "workflow_filename_classname"
        )
        workflow_constructor_kwargs = require(
            "gda_message",
            parsed_stuff,
            "workflow_constructor_kwargs",
        )

        # data_filename = message["filePath"]
        # workflow_constructor_kwargs["dawn_pipeline"] = "dawn_pipeline.nxs"
        # workflow_constructor_kwargs["data_filename"] = data_filename
        # workflow_constructor_kwargs["dataset_path"] = "/entry/whatever"

        try:
            await self.trigger(
                workflow_filename_classname, **workflow_constructor_kwargs
            )
        except Exception as exception:
            logger.error("unable to trigger", exc_info=exception)

    # ----------------------------------------------------------------------------------------
    async def fire(self, message: Dict):
        """"""

        stomp_connection = None
        try:
            stomp_connection = stomp.Connection(
                [(self.__client_url_parts.hostname, self.__client_url_parts.port)]
            )
            stomp_connection.connect()

            logger.debug(
                f"[DMOTF] sending on stomp {self.__client_url_parts.hostname}:{self.__client_url_parts.port}"
            )

            message_string = json.dumps(message, indent=4)
            stomp_connection.send(self.__activemq_topic, message_string)
        finally:
            if stomp_connection is not None:
                stomp_connection.disconnect()
