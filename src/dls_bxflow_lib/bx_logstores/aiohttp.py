import asyncio
import logging
import multiprocessing
import threading

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Base class which maps flask bx_tasks to methods.
from dls_bxflow_api.thing import Thing

# Base class for an aiohttp server.
from dls_bxflow_lib.base_aiohttp import BaseAiohttp

# BxJobs manager.
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs

# Factory to make a BxLogstore.
from dls_bxflow_lib.bx_logstores.bx_logstores import BxLogstores

# BxLogstore protocolj things.
from dls_bxflow_lib.bx_logstores.constants import Commands, Keywords

# BxTasks manager.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_logstores.aiohttp"


# ------------------------------------------------------------------------------------------
class Aiohttp(Thing, BaseAiohttp):
    """
    Object representing a bx_logstore which receives bx_tasks from aiohttp.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Thing.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)
        BaseAiohttp.__init__(
            self,
            specification["type_specific_tbd"]["aiohttp_specification"],
            calling_file=__file__,
        )

        self.__actual_bx_logstore = None

        self.__bx_dataface = bx_datafaces_get_default()

        self._bx_jobs = BxJobs()

        self._bx_tasks = BxTasks()

        # Number of concurrent tasks we will support.
        # TODO: In aiohttp rBxLogstore, make task count max configurable.
        self.__task_count_max = specification["type_specific_tbd"].get(
            "task_count_max", 10
        )
        self.__task_count_now = 0

        # This flag will stop the ticking async task.
        self.__keep_ticking = True
        self.__tick_future = None

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""
        return "%s %s" % ("BxLogstore.Aiohttp", BaseAiohttp.callsign(self))

    # ----------------------------------------------------------------------------------------
    def activate_process(self):
        """"""

        try:
            multiprocessing.current_process().name = "bx_logstore"

            self.activate_process_base()

        except Exception as exception:
            logger.exception("exception in bx_logstore process", exc_info=exception)

    # ----------------------------------------------------------------------------------------
    def activate_thread(self, loop):
        """
        Called from inside a newly created thread.
        """

        try:
            threading.current_thread().name = "bx_logstore"

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
            # Build a local bx_dataface for our back-end.
            self.__actual_bx_logstore = BxLogstores().build_object(
                self.specification()["type_specific_tbd"][
                    "actual_bx_logstore_specification"
                ]
            )

            # Get the local implementation started.
            await self.__actual_bx_logstore.activate()

            # ----------------------------------------------
            await BaseAiohttp.activate_coro_base(self)

            self.__tick_future = asyncio.get_event_loop().create_task(self.tick())

        except Exception as exception:
            raise RuntimeError(
                "exception while starting bx_logstore server"
            ) from exception

    # ----------------------------------------------------------------------------------------
    async def direct_shutdown(self):
        """"""

        if self.__actual_bx_logstore is not None:
            # Disconnect our local dataface connection, i.e. the one which holds the database connection.
            await self.__actual_bx_logstore.deactivate()

        if self.__tick_future is not None:
            # Set flag to stop the periodic ticking.
            self.__keep_ticking = False
            # Wait for the ticking to stop.
            await self.__tick_future

        # We are not running in our own event loop?
        if self.owned_event_loop2 is not None:
            # Disconnect from dataface we have been using.
            await self.__bx_dataface.close_client_session()

        # Let the base class stop the server listener.
        await self.base_direct_shutdown()

    # ----------------------------------------------------------------------------------------
    async def tick(self):
        """
        Periodic ticking to check for new work.
        """

        while self.__keep_ticking:
            await asyncio.sleep(1.0)
            try:
                # Harvest any tasks which we started that may be done.
                await self.harvest()
            except Exception as exception:
                logger.error("failed during tick-triggered harvest", exc_info=exception)

    # ----------------------------------------------------------------------------------------
    # From http client, request server to submit bx_task for execution.

    async def submit(
        self, bx_job_uuid, bx_job_specification, bx_task_uuid, bx_task_specification
    ):
        """"""

        await self.__send_protocolj(
            "submit",
            bx_job_uuid,
            bx_job_specification,
            bx_task_uuid,
            bx_task_specification,
        )

    # ----------------------------------------------------------------------------------------
    # From http client, request server to submit bx_task for execution.

    async def harvest(self):
        """"""

        await self.__send_protocolj(
            "harvest",
        )

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

        function = getattr(self.__actual_bx_logstore, function)

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
