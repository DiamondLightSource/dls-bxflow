import asyncio
import logging
import multiprocessing
import threading

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Basic things.
from dls_bxflow_api.thing import Thing

# Base class for an aiohttp server.
from dls_bxflow_lib.base_aiohttp import BaseAiohttp

# Catalog object.
from dls_bxflow_lib.bx_catalogs.bx_catalogs import BxCatalogs

# BxCatalog protocolj things.
from dls_bxflow_lib.bx_catalogs.constants import Commands, Keywords

# News object.
from dls_bxflow_lib.bx_news.bx_news import BxNews, bx_news_get_default
from dls_bxflow_lib.bx_news.constants import Topics

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_catalogs.aiohttp"


# ------------------------------------------------------------------------------------------
class Aiohttp(Thing, BaseAiohttp):
    """
    Object implementing remote procedure calls for bx_catalog methods.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        Thing.__init__(self, thing_type, specification)
        BaseAiohttp.__init__(
            self,
            specification["type_specific_tbd"]["aiohttp_specification"],
            calling_file=__file__,
        )

        self.__bx_news = None
        self.__bx_news_consumer_future = None
        self.__actual_bx_catalog = None

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""
        return "%s %s" % ("BxCatalog.Aiohttp", BaseAiohttp.callsign(self))

    # ----------------------------------------------------------------------------------------
    def activate_process(self):
        """"""

        try:
            multiprocessing.current_process().name = "bx_catalog"

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
            threading.current_thread().name = "bx_catalog"

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

            # Build a bx_news to listen for events.
            # This has to be a new instance because we will run a receive loop on it.
            self.__bx_news = BxNews().build_object(
                bx_news_get_default().specification()
            )

            # Start a news consumer receive loop.
            self.__bx_news_consumer_future = asyncio.create_task(
                self.__bx_news.consume(self.__consume_bx_news)
            )

            # Build a actual bx_catalog for our back-end.
            self.__actual_bx_catalog = BxCatalogs().build_object(
                self.specification()["type_specific_tbd"][
                    "actual_bx_catalog_specification"
                ]
            )

            # Get the local implementation started.
            await self.__actual_bx_catalog.start()

            await BaseAiohttp.activate_coro_base(self, route_tuples)

        except Exception:
            raise RuntimeError(f"unable to start {callsign(self)} server coro")

    # ----------------------------------------------------------------------------------------
    async def direct_shutdown(self):
        """"""

        # Disconnect from the actual catalog instance.
        # logger.info("disconnecting actual catalog")
        await self.__actual_bx_catalog.disconnect()

        # We are not running in our own event loop?
        if self.owned_event_loop2 is not None:
            logger.debug("disconnecting owned loop dataface")
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

        # Let the base class stop the server listener.
        await Aiohttp.base_direct_shutdown(self)

    # ----------------------------------------------------------------------------------------
    async def __consume_bx_news(self, topic, headline, payload):
        """ """

        try:
            # logger.debug(describe(f"consuming news topic {topic}", payload))
            if topic == Topics.BXJOB_WAS_ENABLED:
                await self.__handle_bx_job_enabled(payload)
            elif topic == Topics.BXJOB_SUCCEEDED:
                await self.__handle_bx_job_finished(payload)
            elif topic == Topics.BXJOB_FAILED:
                await self.__handle_bx_job_finished(payload)
            elif topic == Topics.BXTASK_WAS_STARTED:
                await self.__handle_bx_task_started(payload)
            elif topic == Topics.BXGATE_WAS_OPENED:
                await self.__handle_bx_gate_opened(payload)

        except Exception as exception:
            logger.error(f"failed in handling {topic} news", exc_info=exception)

    # ----------------------------------------------------------------------------------------
    async def __handle_bx_job_enabled(self, payload):
        """ """

        # The payload contains information about the bx_job in question.
        bx_job_dict = require("payload", payload, "bx_job")

        # The payload contains the bx_job_uuid in question.
        bx_job_uuid = require("payload[bx_job]", bx_job_dict, "uuid")

        # Get the complete job record from the dataface.
        bx_job_record = await bx_datafaces_get_default().get_bx_job(bx_job_uuid)

        # Create the catalog workflow run.
        await self.__actual_bx_catalog.create_workflow_run(bx_job_record)

        # Update the bx_dataface with the catalog's uuid.
        # updated_bx_job_record = {
        #     "uuid": bx_job_uuid,
        #     "bx_catalog_uuid": bx_catalog_uuid,
        # }
        # await bx_datafaces_get_default().update_bx_job(updated_bx_job_record)

    # ----------------------------------------------------------------------------------------
    async def __handle_bx_job_finished(self, payload):
        """ """

        # The payload contains information about the bx_job in question.
        bx_job_dict = require("payload", payload, "bx_job")

        # The payload contains the bx_job_uuid in question.
        bx_job_uuid = require("payload[bx_job]", bx_job_dict, "uuid")

        # Get the complete job record from the dataface.
        bx_job_record = await bx_datafaces_get_default().get_bx_job(bx_job_uuid)

        # Tell the actual catalog that the job is finished.
        await self.__actual_bx_catalog.finish_workflow_run(bx_job_record)

        # Update the bx_dataface with the catalog's uuid.
        # updated_bx_job_record = {
        #     "uuid": bx_job_uuid,
        #     "bx_catalog_uuid": bx_catalog_uuid,
        # }
        # await bx_datafaces_get_default().update_bx_job(updated_bx_job_record)

    # ----------------------------------------------------------------------------------------
    async def __handle_bx_task_started(self, payload):
        """ """

        # The payload contains information about the bx_job in question.
        bx_task_dict = require("payload", payload, "bx_task")

        # The payload contains the bx_task_uuid and label of the gate in question.
        bx_task_uuid = require("payload[bx_gate]", bx_task_dict, "uuid")

        # Get the complete task record from the dataface.
        bx_task_record = await bx_datafaces_get_default().get_bx_task(bx_task_uuid)
        bx_task_label = require("bx_task_record", bx_task_record, "label")

        # Update the ispyb message.
        bx_job_uuid = require("bx_task_record", bx_task_record, "bx_job_uuid")
        await self.__actual_bx_catalog.update_workflow_run_message(
            bx_job_uuid, f"bxflow task {bx_task_label} started"
        )

    # ----------------------------------------------------------------------------------------
    async def __handle_bx_gate_opened(self, payload):
        """ """

        # The payload contains information about the bx_job in question.
        bx_gate_dict = require("payload", payload, "bx_gate")

        # The payload contains the bx_task_uuid and label of the gate in question.
        bx_task_uuid = require("payload[bx_gate]", bx_gate_dict, "bx_task_uuid")
        bx_gate_label = require("payload[bx_gate]", bx_gate_dict, "label")

        # Get the complete task record from the dataface.
        bx_task_record = await bx_datafaces_get_default().get_bx_task(bx_task_uuid)
        bx_task_label = require("bx_task_record", bx_task_record, "label")

        # Update the ispyb message.
        bx_job_uuid = require("bx_task_record", bx_task_record, "bx_job_uuid")
        await self.__actual_bx_catalog.update_workflow_run_message(
            bx_job_uuid, f"bxflow task {bx_task_label} {bx_gate_label}"
        )

    # ----------------------------------------------------------------------------------------
    async def disconnect(self):
        """"""

        # Disconnect the client we have been using.
        await self.close_client_session()

    # ----------------------------------------------------------------------------------------
    async def create_workflow_run(self, bx_job_record):
        """"""
        return await self.__send_protocolj("create_workflow_run", bx_job_record)

    # ----------------------------------------------------------------------------------------
    async def update_workflow_run_message(self, bx_job_uuid, message):
        """"""
        return await self.__send_protocolj(
            "update_workflow_run_message", bx_job_uuid, message
        )

    async def attach_workflow_run_file(self, bx_job_uuid, bx_task_uuid, filename):
        """"""
        return await self.__send_protocolj(
            "attach_workflow_run_file", bx_job_uuid, bx_task_uuid, filename
        )

    async def query_workflow_run(self, bx_job_uuid):
        """"""
        return await self.__send_protocolj("query_workflow_run", bx_job_uuid)

    async def query_workflow_run_files(self, bx_job_uuid):
        """"""
        return await self.__send_protocolj("query_workflow_run_files", bx_job_uuid)

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

        function = getattr(self.__actual_bx_catalog, function)

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
