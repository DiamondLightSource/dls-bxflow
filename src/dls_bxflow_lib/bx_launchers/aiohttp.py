import asyncio
import json
import logging
import multiprocessing
import threading

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.explain import explain
from dls_utilpack.require import require

from dls_bxflow_api.bx_databases.constants import BxLauncherFieldnames

# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Remex things.
from dls_bxflow_api.remex import Clusters as RemexClusters

# Base class which maps flask bx_tasks to methods.
from dls_bxflow_api.thing import Thing

# Base class for an aiohttp server.
from dls_bxflow_lib.base_aiohttp import BaseAiohttp

# BxJobs manager.
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs

# Factory to make a BxLauncher.
from dls_bxflow_lib.bx_launchers.bx_launchers import BxLaunchers

# BxLauncher protocolj things.
from dls_bxflow_lib.bx_launchers.constants import Commands, Keywords

# Possible bx_launcher states.
from dls_bxflow_lib.bx_launchers.states import States as BxLauncherStates

# BxTasks manager.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_launchers.aiohttp"


# ------------------------------------------------------------------------------------------
class Aiohttp(Thing, BaseAiohttp):
    """
    Object representing a bx_launcher which receives bx_tasks from aiohttp.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Thing.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)
        BaseAiohttp.__init__(
            self,
            specification["type_specific_tbd"]["aiohttp_specification"],
            calling_file=__file__,
        )

        self.__actual_bx_launcher = None

        self.__bx_dataface = bx_datafaces_get_default()

        self._bx_jobs = BxJobs()

        self._bx_tasks = BxTasks()

        # Number of concurrent tasks we will support.
        # TODO: In aiohttp rBxLauncher, make task count max configurable.
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
        if self.__actual_bx_launcher is not None:
            uuid = self.__actual_bx_launcher.uuid()
        else:
            uuid = self.uuid()

        return "%s %s" % ("BxLauncher.Aiohttp", uuid)

    # ----------------------------------------------------------------------------------------
    def activate_process(self):
        """"""

        try:
            multiprocessing.current_process().name = "bx_launcher"

            self.activate_process_base()

        except Exception as exception:
            logger.exception("exception in bx_launcher process", exc_info=exception)

    # ----------------------------------------------------------------------------------------
    def activate_thread(self, loop):
        """
        Called from inside a newly created thread.
        """

        try:
            threading.current_thread().name = "bx_launcher"

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
            self.__actual_bx_launcher = BxLaunchers().build_object(
                self.specification()["type_specific_tbd"][
                    "actual_bx_launcher_specification"
                ],
                predefined_uuid=self.uuid(),
            )

            # Get the local implementation started.
            await self.__actual_bx_launcher.activate()

            # ----------------------------------------------
            await BaseAiohttp.activate_coro_base(self)

            # After we are receiving requests, set our own implementation advertised.
            await self.advertise()

            self.__tick_future = asyncio.get_event_loop().create_task(self.tick())

        except Exception as exception:
            raise RuntimeError(
                "exception while starting bx_launcher server"
            ) from exception

    # ----------------------------------------------------------------------------------------
    async def advertise(self):
        """"""
        try:

            self.set_state(BxLauncherStates.IDLE)

            # Make sure the launcher is advertising a valid cluster.
            try:
                remex_cluster = require(
                    f"{callsign(self)} specification",
                    self.specification(),
                    BxLauncherFieldnames.REMEX_CLUSTER,
                )

                RemexClusters.validate(remex_cluster)
            except Exception as exception:
                raise RuntimeError(explain(exception, f"advertising {callsign(self)}"))

            # Define a database entry for our instance.
            record = {}
            record["type"] = self.thing_type()
            record["uuid"] = self.__actual_bx_launcher.uuid()
            record["state"] = self.get_state()
            record["specification"] = json.dumps(self.specification())
            record[BxLauncherFieldnames.REMEX_CLUSTER] = remex_cluster

            # Remove any old launcher records in the database with this uuid.
            # We do this instead of upsert since there might be messiness with old records.
            # Probably better to make the database enforce uuid unique.
            await self.__bx_dataface.delete_bx_launcher(record["uuid"])

            # Insert new launcher records in the database.
            await self.__bx_dataface.set_bx_launcher(record)

        except Exception as exception:
            raise RuntimeError(
                "exception while starting bx_launcher server"
            ) from exception

    # ----------------------------------------------------------------------------------------
    async def unadvertise(self):
        """Remove instance record from the database so no more submissions are sent."""

        self.set_state(BxLauncherStates.SHUTDOWN)
        record = {
            "uuid": self.__actual_bx_launcher.uuid(),
            "state": self.get_state(),
        }

        await self.__bx_dataface.update_bx_launcher(
            record, launcher_callsign=callsign(self)
        )

    # ----------------------------------------------------------------------------------------
    async def direct_shutdown(self):
        """"""

        # Remove instance record from the database so no more submissions are sent.
        await self.unadvertise()

        if self.__actual_bx_launcher is not None:
            # Disconnect our local dataface connection, i.e. the one which holds the database connection.
            await self.__actual_bx_launcher.deactivate()

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

    # ------------------------------------------------------------------------------------------

    async def report_health(self):
        """
        Report server health.
        """

        # Get the base class's health report.
        report = await BaseAiohttp.report_health(self)

        # Report other things about launchers.
        report["state"] = self.get_state()

        infos = self.__actual_bx_launcher.get_unharvested_infos()

        details = []
        for info in infos:
            unserialized = json.loads(info.serialize())
            details.append(unserialized)
        report["details"] = details

        return report

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

        function = getattr(self.__actual_bx_launcher, function)

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
