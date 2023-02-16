import json
import logging
import time

# Database field names.
from dls_bxflow_api.bx_databases.constants import BxJobFieldnames

# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Global bx_filestore.
from dls_bxflow_lib.bx_filestores.bx_filestores import bx_filestores_get_default

# Base class for a Thing which has a name and traits.
from dls_bxflow_lib.bx_jobs.base import Base

# States of things.
from dls_bxflow_lib.bx_jobs.states import States as BxJobStates

# Objects managing things.
from dls_bxflow_run.bx_gates.bx_gates import BxGates

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_jobs.standard"


class Standard(Base):
    """
    Object representing a bx_job for a standard command.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.__blocked_by_bx_gates = BxGates()

        self.state(BxJobStates.PREPARING)

    # -----------------------------------------------------------------------------
    async def enable(self):
        # Register ourself.
        await self.register()

        # Register all the tasks.
        await self.register_bx_tasks()

        await bx_datafaces_get_default().enable_bx_job(self.uuid())

    # -----------------------------------------------------------------------------
    async def register(self):
        bx_filestores_get_default().pin_job_directory(self)

        bx_job_dict = {}
        bx_job_dict["uuid"] = self.uuid()
        bx_job_dict["type"] = self.thing_type()
        bx_job_dict[BxJobFieldnames.LABEL] = self.specification()["label"]
        bx_job_dict[BxJobFieldnames.DATA_LABEL] = self.get_data_label()
        bx_job_dict[BxJobFieldnames.DIRECTORY] = self.get_directory()
        bx_job_dict[BxJobFieldnames.BX_WORKFLOW_UUID] = self.get_workflow_uuid()
        bx_job_dict["specification"] = json.dumps(self.specification())

        # Beamline and visit are set from the default filestore
        # which is in play at the time of job registration.
        bx_job_dict[
            BxJobFieldnames.BEAMLINE
        ] = bx_filestores_get_default().get_beamline()
        bx_job_dict[BxJobFieldnames.VISIT] = bx_filestores_get_default().get_visit()
        await bx_datafaces_get_default().set_bx_jobs([bx_job_dict])

        bx_job_blocked_by_bx_gates = []
        for blocked_by_bx_gate in self.__blocked_by_bx_gates.list():
            bx_job_blocked_by_bx_gates.append(
                {"lhs": self.uuid(), "rhs": blocked_by_bx_gate.uuid()}
            )

        await bx_datafaces_get_default().set_bx_job_blocked_by_bx_gates(
            bx_job_blocked_by_bx_gates
        )

    # -----------------------------------------------------------------------------
    async def register_bx_tasks(self):

        bx_tasks_list = []
        bx_task_dependency_bx_gates = []
        for bx_task in self.bx_tasks.list():
            bx_filestores_get_default().pin_task_directory(self, bx_task)

            bx_task_dict = {}
            bx_task_dict["bx_job_uuid"] = self.uuid()
            bx_task_dict["type"] = bx_task.thing_type()
            bx_task_dict["uuid"] = bx_task.uuid()
            bx_task_dict["state"] = bx_task.state()
            bx_task_dict["label"] = bx_task.label()
            bx_task_dict["directory"] = bx_task.get_directory()
            # Put the directory in the specification when we register it.
            bx_task.specification()["directory"] = bx_task.get_directory()
            bx_task_dict["specification"] = json.dumps(bx_task.specification())
            bx_tasks_list.append(bx_task_dict)

            await bx_task.controlled_bx_gates.register(self.uuid(), bx_task.uuid())

            for dependency_bx_gate in bx_task.dependency_bx_gates.list():
                bx_task_dependency_bx_gates.append(
                    {"lhs": bx_task.uuid(), "rhs": dependency_bx_gate.uuid()}
                )

        await bx_datafaces_get_default().set_bx_task_dependency_bx_gates(
            bx_task_dependency_bx_gates
        )
        await bx_datafaces_get_default().set_bx_tasks(bx_tasks_list)

    # -----------------------------------------------------------------------------
    async def fetch(self):
        """
        Fetch object from bx_dataface.
        """

        # Get all the bx_tasks for the bx_job.
        bx_task_records = await bx_datafaces_get_default().get_bx_tasks(self.uuid())

        # --------------------------------------------------------------------------------
        # Get all the bx_gates controlled by the bx_tasks in the bx_job.
        controlled_bx_gate_records = (
            await bx_datafaces_get_default().get_controlled_bx_gates(self.uuid())
        )

        bx_gates = BxGates()
        for controlled_bx_gate_record in controlled_bx_gate_records:
            controlled_bx_gate_uuid = controlled_bx_gate_record["uuid"]
            bx_gate = bx_gates.build_object(
                specification=controlled_bx_gate_record["specification"],
                predefined_uuid=controlled_bx_gate_uuid,
            )
            bx_gate.bx_task_uuid(controlled_bx_gate_record["bx_task_uuid"])
            bx_gates.add(bx_gate)
            # logger.debug(
            #     "bx_gate %s controlled by task %s"
            #     % (bx_gate.uuid(), bx_gate.bx_task_uuid())
            # )

        # --------------------------------------------------------------------------------
        # Assign the in-memory objects.

        # Get all the dependency bx_gates needed by the bx_tasks.
        bx_task_dependency_bx_gates_records = (
            await bx_datafaces_get_default().get_dependency_bx_gates(self.uuid())
        )

        for bx_task_record in bx_task_records:
            bx_task_uuid = bx_task_record["uuid"]
            bx_task = self.bx_tasks.build_object(
                specification=bx_task_record["specification"],
                predefined_uuid=bx_task_uuid,
            )
            bx_task.set_directory(bx_task_record["directory"])
            bx_task.bx_job_uuid(bx_task_record["bx_job_uuid"])

            for bx_gate in bx_gates.list():
                if bx_gate.bx_task_uuid() == bx_task_uuid:
                    bx_task.controlled_bx_gates.add(bx_gate)

            for (
                bx_task_dependency_bx_gates_record
            ) in bx_task_dependency_bx_gates_records:
                if bx_task_dependency_bx_gates_record["lhs"] == bx_task_uuid:
                    bx_gate_uuid = bx_task_dependency_bx_gates_record["rhs"]
                    bx_gate = bx_gates.find(bx_gate_uuid)
                    bx_task.dependency_bx_gates.add(bx_gate)

            self.bx_tasks.add(bx_task)

        # --------------------------------------------------------------------------------
        # Get all the gates which will block the job when high.
        blocked_by_bx_gates_records = (
            await bx_datafaces_get_default().get_blocked_by_bx_gates(self.uuid())
        )

        for record in blocked_by_bx_gates_records:
            uuid = record["rhs"]
            bx_gate = bx_gates.find(uuid)
            self.blocked_by_bx_gates.add(bx_gate)

    # -----------------------------------------------------------------------------
    async def wait(self, timeout=5.0, naptime=0.1):
        is_finished = False

        t0 = time.time()
        while timeout is None or time.time() - t0 < timeout:
            bx_job_record = await bx_datafaces_get_default().get_bx_job(self.uuid())
            if bx_job_record is None:
                raise RuntimeError(f"dataface does not have bx_job {self.uuid()}")
            if (
                bx_job_record["state"] != BxJobStates.IN_PROGRESS
                and bx_job_record["state"] != BxJobStates.READY
            ):
                is_finished = True
                break
            time.sleep(naptime)

        return is_finished

    # -----------------------------------------------------------------------------
    def _get_blocked_by_bx_gates(self):
        return self.__blocked_by_bx_gates

    def _set_blocked_by_bx_gates(self, blocked_by_bx_gates):
        self.__blocked_by_bx_gates = blocked_by_bx_gates

    def _del_blocked_by_bx_gates(self):
        del self.__blocked_by_bx_gates

    blocked_by_bx_gates = property(
        fget=_get_blocked_by_bx_gates,
        fset=_set_blocked_by_bx_gates,
        fdel=_del_blocked_by_bx_gates,
        doc="The blocked_by_bx_gates property.",
    )
