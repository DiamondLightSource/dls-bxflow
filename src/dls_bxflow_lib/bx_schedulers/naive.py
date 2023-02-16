import asyncio
import json
import logging

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.explain import explain
from dls_utilpack.qualname import qualname

from dls_bxflow_api.bx_databases.constants import BxLauncherFieldnames

# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Exceptions.
from dls_bxflow_api.exceptions import (
    CapacityReached,
    DlsBxflowClientConnectorError,
    TransientError,
)

# Remex things.
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Base class for simple things.
from dls_bxflow_api.thing import Thing
from dls_bxflow_lib.bx_jobs.states import States as BxJobStates

# BxLauncher manager.
from dls_bxflow_lib.bx_launchers.bx_launchers import BxLaunchers
from dls_bxflow_lib.bx_launchers.states import States as BxLauncherStates

# Base class for a scheduler.
from dls_bxflow_lib.bx_schedulers.base import Base as BxSchedulerBase

# States of things.
from dls_bxflow_run.bx_gates.states import States as BxGateStates
from dls_bxflow_run.bx_tasks.states import States as BxTaskStates

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_schedulers.naive"


class Naive(BxSchedulerBase):
    """
    Object providing logic to select a launcher for a task.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Thing.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.__match_lock = asyncio.Lock()

        # Cached list of launchers we have connected with.
        self._bx_launchers = BxLaunchers()

    # ----------------------------------------------------------------------------------------
    async def activate(self):
        """
        Activate operations.
        Called by parent when service is going live.
        """
        pass

    # ----------------------------------------------------------------------------------------
    async def deactivate(self):
        """
        Deactivate operations
        Called by parent when service is shutting down.
        """

        # Disconnect from all the launchers we have been using.
        for bx_launcher in self._bx_launchers.list():
            await bx_launcher.close_client_session()

    # ----------------------------------------------------------------------------------------
    async def match(self):
        """
        Find eligible tasks and give them to a launcher.
        """

        async with self.__match_lock:
            # Subquery to get all gates which are closed.
            closed_bx_gates_sql = (
                "SELECT uuid, label FROM bx_gates WHERE state = '%s'"
                % (BxGateStates.CLOSED)
            )

            # Subquery to get all tasks with closed dependency gates.
            bx_tasks_with_closed_dependency_bx_gates_sql = (
                "SELECT DISTINCT bx_task_dependency_bx_gates.lhs AS bx_task_uuid"
                " FROM bx_task_dependency_bx_gates"
                " JOIN (%s) AS closed_bx_gates"
                " ON bx_task_dependency_bx_gates.rhs = closed_bx_gates.uuid"
            ) % (closed_bx_gates_sql)

            # Query to get all eligible tasks.
            eligible_bx_tasks_sql = (
                "SELECT bx_tasks.*,"
                " bx_jobs.specification AS bx_job_specification"
                " FROM bx_tasks"
                " JOIN bx_jobs ON bx_jobs.uuid = bx_tasks.bx_job_uuid"
                f" WHERE bx_jobs.state = '{BxJobStates.READY}'"
                f" AND bx_tasks.state = '{BxTaskStates.PREPARED}'"
                " AND bx_tasks.uuid"
                f" NOT IN ({bx_tasks_with_closed_dependency_bx_gates_sql})"
            )

            eligible_task_records = await bx_datafaces_get_default().query(
                eligible_bx_tasks_sql
            )

            # logger.debug(describe("eligible bx_tasks", eligible_task_records))

            for eligible_task_record in eligible_task_records:
                # De-serialize the job specification stored in the database as a string.
                bx_job_specification = json.loads(
                    eligible_task_record["bx_job_specification"]
                )

                # De-serialize the task specification stored in the database as a string.
                eligible_task_specification = json.loads(
                    eligible_task_record["specification"]
                )

                # Launch task, don't wait for it to finish.
                await self.launch(
                    eligible_task_record["bx_job_uuid"],
                    bx_job_specification,
                    eligible_task_record["uuid"],
                    eligible_task_specification,
                )

    # ----------------------------------------------------------------------------------------
    async def launch(
        self, bx_job_uuid, bx_job_specification, bx_task_uuid, bx_task_specification
    ):
        """
        Give tasks to a waiting launcher.
        Raise exception if the task could not be launched.
        """

        # Try to find a client to a launcher.
        # This will raise TransientError if none.
        # No use to try to launch more tasks until next tick cycle.
        launcher_client = await self.select_launcher_client(
            bx_task_uuid, bx_task_specification
        )

        try:
            logger.debug(f"submitting task to {callsign(launcher_client)}")

            await launcher_client.submit(
                bx_job_uuid,
                bx_job_specification,
                bx_task_uuid,
                bx_task_specification,
            )

        except DlsBxflowClientConnectorError:
            try:
                # Update handler state so it won't be selected for more work.
                # TODO: Distinguish between launcher temporarily unresponsive and completely dead.
                await bx_datafaces_get_default().update_bx_launcher(
                    {
                        "uuid": launcher_client.uuid(),
                        "state": BxLauncherStates.UNRESPONSIVE,
                    },
                    launcher_callsign=callsign(launcher_client),
                )
            except Exception as exception:
                logger.warning(
                    explain(exception, "updating launcher state"), exc_info=exception
                )

            # Remove this launcher from our pool of connections.
            self._bx_launchers.remove(launcher_client.uuid())

            raise TransientError(f"unable to connect to {callsign(launcher_client)}")

        # Other exception, including if the launcher service refused the request.
        except Exception as exception:
            if qualname(CapacityReached) in str(exception):
                raise TransientError(
                    f"{callsign(launcher_client)} replies CapacityReached"
                )
            else:
                raise RuntimeError(
                    explain(
                        exception, f"submitting task to {callsign(launcher_client)}"
                    )
                )

    # ----------------------------------------------------------------------------------------
    async def select_launcher_client(self, bx_task_uuid, bx_task_specification):
        """
        Get a live launcher who has matching remex_hints with the task specified.
        """

        # Identify the database record of the launcher who can handle this task.
        launcher_record = await self.select_launcher_record(
            bx_task_uuid, bx_task_specification
        )

        launcher_uuid = launcher_record["uuid"]
        if not self._bx_launchers.has(launcher_uuid):
            launcher_specification = json.loads(launcher_record["specification"])

            launcher_client = self._bx_launchers.build_object(
                launcher_specification,
                predefined_uuid=launcher_uuid,
            )
            self._bx_launchers.add(launcher_client)
            logger.debug(f"creating new client to launcher {launcher_uuid}")
        else:
            logger.debug(f"using existing client to launcher {launcher_uuid}")
            launcher_client = self._bx_launchers.find(launcher_uuid)

        return launcher_client

    # ----------------------------------------------------------------------------------------
    async def select_launcher_record(self, bx_task_uuid, bx_task_specification):
        """
        Get a live launcher who has matching remex_hints with the task specified.
        """

        launcher_records = await bx_datafaces_get_default().get_bx_launchers(
            [BxLauncherStates.IDLE]
        )

        # No idle launchers at all?
        if len(launcher_records) == 0:
            raise TransientError(
                f"[NOLAUNCH1] no {BxLauncherStates.IDLE} launchers at all"
            )

        # Clusters the task is asking for.
        task_remex_hints = bx_task_specification.get(RemexKeywords.HINTS, {})
        task_remex_clusters = task_remex_hints.get(RemexKeywords.CLUSTER, None)

        # Allow a scalar value, turn into a list.
        if not isinstance(task_remex_clusters, list):
            task_remex_clusters = [task_remex_clusters]

        # Loop over tasks's requested clusters.
        found = False
        launcher_record = None
        for task_remex_cluster in task_remex_clusters:
            # Look for a launcher that provides the cluster.
            for launcher_record in launcher_records:
                launcher_remex_cluster = launcher_record[
                    BxLauncherFieldnames.REMEX_CLUSTER
                ]
                if task_remex_cluster == launcher_remex_cluster:
                    found = True
                    break
            if found:
                break

        if not found:
            s = "{" + ", ".join(task_remex_clusters) + "}"
            raise TransientError(
                f"[NOLAUNCH2] no {BxLauncherStates.IDLE} launchers for clusters for task {bx_task_uuid} remex_hints {s}"
            )

        return launcher_record
