import functools
import json
import logging
import multiprocessing
import threading
import time

import dask
from dask.delayed import Delayed
from dask.distributed import Client, LocalCluster
from dask.threaded import get

# Utilities
from dls_utilpack.describe import describe
from dls_utilpack.explain import explain

# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Base class for simple things.
from dls_bxflow_api.thing import Thing
from dls_bxflow_lib.base_aiohttp import BaseAiohttp
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs
from dls_bxflow_lib.bx_jobs.states import States as BxJobStates
from dls_bxflow_run.bx_variables.bx_variables import BxVariables

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_schedulers.dask"


# ----------------------------------------------------------------------------------------
async def wait_gate(
    *args,
    bx_gate_uuid=None,
    full_gate_label=None,
):
    # Args is the output from contributing dask tasks, not used in our scheme.
    # logger.info(describe("submitting args", args))
    print("********** waiting on gate %s" % (full_gate_label,))

    # # Launch task, don't wait for it to finish.
    # await self.launch(
    #     bx_job.uuid(),
    #     bx_task.uuid(),
    #     bx_task.specification(),
    # )

    return 1


# ----------------------------------------------------------------------------------------
async def launch_task(
    *args,
    bx_job_uuid=None,
    bx_job_label=None,
    bx_task_uuid=None,
    bx_task_label=None,
    bx_task_specification=None,
):
    # Args is the output from contributing dask tasks, not used in our scheme.
    # logger.info(describe("submitting args", args))
    print(
        "********** %s [%s] [%s] launching task %s"
        % (
            multiprocessing.current_process().pid,
            multiprocessing.current_process().name,
            threading.current_thread().name,
            bx_task_label,
        )
    )

    # # Launch task, don't wait for it to finish.
    # await self.launch(
    #     bx_job.uuid(),
    #     bx_task.uuid(),
    #     bx_task.specification(),
    # )

    return 2


# ----------------------------------------------------------------------------------------
async def block_job(
    *args,
    bx_job_uuid=None,
    bx_job_label=None,
    blocked_by_bx_gate_uuids=None,
):
    # Args is the output from contributing dask tasks, not used in our scheme.
    # logger.info(describe("submitting args", args))
    print("********** blocking job %s" % (bx_job_label))

    # # Launch task, don't wait for it to finish.
    # await self.launch(
    #     bx_job.uuid(),
    #     bx_task.uuid(),
    #     bx_task.specification(),
    # )

    return 3


# --------------------------------------------------------------------------------------------
class Dask(Thing, BaseAiohttp):
    """
    Object representing a registry which executes in dask.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        Thing.__init__(self, thing_type, specification)
        BaseAiohttp.__init__(self, specification["server"])

        self.__should_use_dask_distributed = True
        self.__dask_client = None

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""
        return "%s %s" % ("BxScheduler.Dask", BaseAiohttp.callsign(self))

    # ----------------------------------------------------------------------------------------
    async def direct_shutdown(self):
        """"""

        # Let the base class do the shutdown of the process-specific asyncio event loop.
        await BaseAiohttp.direct_shutdown(self)

    # ----------------------------------------------------------------------------------------
    async def notify(self, notification_dict):
        """"""

        logger.info("got notification\n%s" % (json.dumps(notification_dict)))

    # ----------------------------------------------------------------------------------------
    async def match(self):
        """ """

        ready_bx_job_records = await bx_datafaces_get_default().get_bx_jobs(
            [BxJobStates.READY]
        )

        logger.info(
            "found %d %s bx_jobs" % (len(ready_bx_job_records), BxJobStates.READY)
        )

        for ready_bx_job_record in ready_bx_job_records:
            try:
                await self.start_job(ready_bx_job_record)
            except Exception as exception:
                logger.info(
                    explain(exception, "start_job coroutine failed"), exc_info=exception
                )

    # ----------------------------------------------------------------------------------------
    async def start_job(self, bx_job_record):

        bx_job_uuid = bx_job_record["uuid"]

        row = {"uuid": bx_job_uuid, "state": BxJobStates.PREPARING}
        count = await bx_datafaces_get_default().update_bx_job(row)
        if count == 0:
            raise RuntimeError(f"unable to set job to state {BxJobStates.PREPARING}")

        if self.__should_use_dask_distributed and self.__dask_client is None:
            # Start local workers as clients.
            cluster = LocalCluster(nanny=False, n_workers=1)
            self.__dask_client = await Client(cluster, asynchronous=True)
            logger.info(
                "client has_what\n%s"
                % (json.dumps(self.__dask_client.has_what(), indent=4))
            )
            # self.__dask_client = await Client("localhost:8786", asynchronous=True)

        logger.info(describe("bx_job_record", bx_job_record))

        bx_job = BxJobs().build_object(
            specification=bx_job_record["specification"], predefined_uuid=bx_job_uuid
        )

        # Fetch the bx_job and all its parts from the bx_dataface.
        await bx_job.fetch()

        # Get all the variables for the bx_job.
        variables = BxVariables()
        await variables.fetch(bx_job.uuid())
        variables_dict = {}
        for variable in variables.list():
            variables_dict[variable.trait("name")] = variable.trait("value")

        # ------------------------------------------------------------------
        # Turn the bx_gates into dask tasks.

        dsk = {}
        full_gate_labels = {}
        for bx_task in bx_job.bx_tasks.list():
            for bx_gate in bx_task.controlled_bx_gates.list():
                full_gate_label = "%s.%s" % (bx_task.label(), bx_gate.label())
                full_gate_labels[bx_gate.uuid()] = full_gate_label

                gate_kwargs = {
                    "bx_gate_uuid": bx_gate.uuid(),
                    "full_gate_label": full_gate_label,
                }
                target_func = functools.partial(wait_gate, **gate_kwargs)
                dask_task = (target_func, bx_task.label())

                dsk[full_gate_label] = dask_task

        for bx_task in bx_job.bx_tasks.list():
            # This is supposed to set the task namespace for subsequent declared tasks.
            # However, it doesn't show up in the Luigi web gui.
            # Delay.task_namespace = f"{bx_job.label()}]{bx_task.label()}"

            launch_kwargs = {
                "bx_job_uuid": bx_job.uuid(),
                "bx_job_label": bx_job.label(),
                "bx_task_uuid": bx_task.uuid(),
                "bx_task_label": bx_task.label(),
                "bx_task_specification": bx_task.specification(),
            }

            target_func = functools.partial(launch_task, **launch_kwargs)
            dask_task = (target_func,)

            for bx_gate in bx_task.dependency_bx_gates.list():
                full_gate_label = full_gate_labels[bx_gate.uuid()]
                dask_task = dask_task + (full_gate_label,)

            dsk[bx_task.label()] = dask_task

        # -----------------------------------------------------------------
        blocked_by_bx_gate_uuids = []
        for bx_gate in bx_job.blocked_by_bx_gates.list():
            blocked_by_bx_gate_uuids.append(bx_gate.uuid())
        block_job_kwargs = {
            "bx_job_uuid": bx_job.uuid(),
            "bx_job_label": bx_job.label(),
            "blocked_by_bx_gate_uuids": blocked_by_bx_gate_uuids,
        }
        target_func = functools.partial(block_job, **block_job_kwargs)
        dask_task = (target_func,)

        # We put in the blocking gates for visualization only.
        # Block_job never gets called because it is an OR of its inputs.
        # for bx_gate in bx_job.blocked_by_bx_gates.list():
        #     full_gate_label = full_gate_labels[bx_gate.uuid()]
        #     dask_task = dask_task + (full_gate_label,)

        dsk["block_job"] = dask_task

        # for plotting dags using matplotlib:
        # https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/

        final_label = bx_job.bx_tasks.list()[-1].label()
        dsk_delayed = Delayed(final_label, dsk)
        dask.visualize(
            dsk_delayed,
            optimize_graph=False,
            verbose=True,
        )

        # logger.info(describe("dsk_delayed", dict(dsk_delayed.dask)))

        if self.__should_use_dask_distributed:
            # Let dask get the result.
            logger.info("getting result on final label %s" % (final_label))
            gather = self.__dask_client.get(dsk, final_label, sync=False)
            logger.info(describe("after creation, gather", gather))
            time.sleep(2.0)
            logger.info(describe("after a delay, gather", gather))
            messages = []
            for address, entries in self.__dask_client.get_worker_logs().items():
                for entry in entries:
                    messages.append(entry[1])
                break
            logger.info("client get_worker_logs\n%s" % ("\n".join(messages)))
            # asyncio.get_running_loop().create_task(gather)
            # r = await gather
            logger.info(describe("after awaited, gather", gather))
            # logger.info(describe("got result which", r))
        else:
            get(dsk, final_label)
