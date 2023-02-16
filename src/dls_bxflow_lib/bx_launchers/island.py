import asyncio
import logging
import multiprocessing

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.explain import explain
from dls_utilpack.require import require

# Base class which maps flask bx_tasks to methods.
from dls_bxflow_api.thing import Thing

# Global bx_dataface.
from dls_bxflow_lib.bx_datafaces.bx_datafaces import BxDatafaces

# BxTask manager.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks
from dls_bxflow_run.bx_tasks.states import States as BxTaskStates

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_launchers.aiohttp"


# ------------------------------------------------------------------------------------------
class Island(Thing):
    """
    Object representing a bx_launcher which receives bx_tasks from aiohttp.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Thing.__init__(self, thing_type, specification, predefined_uuid)

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""
        return "BxLauncher.Island"

    # ----------------------------------------------------------------------------------------
    def cycle(self):
        """"""

        multiprocessing.current_process().name = "island"

        try:
            rc = asyncio.run(self.cycle_coro())
        except Exception as exception:
            logger.error(
                callsign(
                    self, explain(exception, "unable to complete launch cycle process")
                )
            )
            rc = -1

        return rc

    # ----------------------------------------------------------------------------------------
    async def cycle_coro(self):
        """"""

        bx_dataface = None

        try:
            logger.info(callsign(self, "starting"))

            type_specific_tbd = require(
                "specification", self.specification(), "type_specific_tbd"
            )

            bx_job_uuid = require(
                "type_specific_tbd bx_job_uuid", type_specific_tbd, "bx_job_uuid"
            )

            bx_task_uuid = require(
                "type_specific_tbd bx_task_uuid", type_specific_tbd, "bx_task_uuid"
            )

            bx_task_specification = require(
                "type_specific_tbd bx_task_specification",
                type_specific_tbd,
                "bx_task_specification",
            )

            bx_dataface_specification = require(
                "type_specific_tbd bx_dataface_specification",
                type_specific_tbd,
                "bx_dataface_specification",
            )

            bx_dataface = BxDatafaces().build_object(bx_dataface_specification)

            # Update the state of the bx_task we are about to run.
            await bx_dataface.update_bx_task(
                {"uuid": bx_task_uuid, "state": BxTaskStates.STARTED}
            )

            # Make a bx_task object.
            bx_task = BxTasks().build_object(
                bx_task_specification, predefined_uuid=bx_task_uuid
            )

            logger.info(callsign(self, "running task"))

            # Let the bx_task run.
            await bx_task.run()

            logger.info(callsign(self, "noticed task finished"))

            # Update the state of the bx_task we ran.
            await bx_dataface.update_bx_task(
                {"uuid": bx_task_uuid, "state": BxTaskStates.FINISHED}
            )

            # Update the bx_gate states.
            await bx_dataface.open_bx_gate(bx_task_uuid, "success")

            # Update the bx_job state if it is now blocked.
            await bx_dataface.update_bx_job_if_blocked_by_bx_gates(bx_job_uuid)

            logger.info(callsign(self, "finished"))

            return 0

        except Exception as exception:
            raise RuntimeError(
                callsign(
                    self,
                    explain(exception, "unable to complete launch cycle coroutine"),
                )
            )

        finally:
            # We managed to build a bx_dataface?
            if bx_dataface is not None:
                # Close the client connection to it.
                # This is tolerant of no client session actually set up yet.
                bx_dataface.close_client_session()
