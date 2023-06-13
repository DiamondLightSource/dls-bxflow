import asyncio
import json
import logging
import os
from typing import List

# Job info.
from dls_slurmjob_api.models.job_summary_model import JobSummaryModel

# Job properties.
from dls_slurmjob_api.models.openapi.v0.field_0 import (
    Field38JobProperties as OpenapiJobProperties,
)

# Slurmjob client context creator.
from dls_slurmjob_api.restds.context import Context as DlsSlurmjobRestdClientContext

# The dls_slurmjob package version.
from dls_slurmjob_cli.version import version as dls_slurmjob_version

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.explain import explain2

# Environment module loader.
from dls_utilpack.module import module_get_environ
from dls_utilpack.require import require

from dls_bxflow_api.bx_launchers.constants import ClassTypes

# Remex (remote execution) API.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords
from dls_bxflow_lib.bx_launchers.base import Base as BxLauncherBase
from dls_bxflow_lib.bx_launchers.base import BaseLaunchInfo

logger = logging.getLogger(__name__)

thing_type = ClassTypes.SLURMER


# ------------------------------------------------------------------------------------------
class SlurmerLaunchInfo(BaseLaunchInfo):
    """Launch info specific to this launcher type needed to identify a launched task."""

    def __init__(self, bx_job, bx_task):
        BaseLaunchInfo.__init__(self, bx_job, bx_task)
        self.job_id = None

    def serialize(self):
        """Serialize the launch info for storing in a persistent database."""
        return json.dumps({"job_id": self.job_id})


# ------------------------------------------------------------------------------------------
class Slurmer(BxLauncherBase):
    """
    Object representing a bx_launcher which launches a task using slurm for cluster execution.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification, predefined_uuid=None):
        BxLauncherBase.__init__(
            self, thing_type, specification, predefined_uuid=predefined_uuid
        )

        # Cluster project for accounting purposes is typically the beamline.
        self.__slurmjob_specification = require(
            f"{callsign(self)} specification type_specific_tbd",
            self.specification().get("type_specific_tbd", {}),
            "slurmjob_specification",
        )

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""
        return "%s %s" % (thing_type, self.uuid())

    # ----------------------------------------------------------------------------------------
    async def activate(self):
        """"""
        await BxLauncherBase.activate(self)

        remex_hints = self.specification().get("remex_hints", None)

        if remex_hints is None:
            logger.warning(f"{callsign(self)} specification has no remex_hints")
            return

        cluster = remex_hints.get(RemexKeywords.CLUSTER, None)

        if cluster is not None:
            # Load the environment module needed to talk to this cluster.
            os.environ.update(module_get_environ(cluster))
            logger.debug(f"successful module load {cluster}")

    # ----------------------------------------------------------------------------------------
    def __sanitize(self, uuid: str) -> str:
        return f"bx_task_{uuid}"

    # ------------------------------------------------------------------------------------------
    async def submit(
        self,
        bx_job_uuid,
        bx_job_specification,
        bx_task_uuid,
        bx_task_specification,
    ):
        """Handle request to submit bx_task for execution."""

        # Let the base class prepare the directory and build up a script to run.
        (
            bx_job,
            bx_task,
            runtime_directory,
            bash_filename,
        ) = await BxLauncherBase.presubmit(
            self,
            bx_job_uuid,
            bx_job_specification,
            bx_task_uuid,
            bx_task_specification,
        )

        job_name = self.__sanitize(bx_task_uuid)
        stdout_filename = "%s/stdout.txt" % (runtime_directory)
        stderr_filename = "%s/stderr.txt" % (runtime_directory)

        # The task may specify remex hints to help select the cluster affinity.
        remex_hints = bx_task_specification.get("remex_hints", None)
        if remex_hints is None:
            remex_hints = {}

        # The remex hints may contain slurm properties to be used directly.
        remex_slurm = remex_hints.get("slurm", {})

        # Start with properties direct from the remex hints.
        # Partition is required among these.
        properties = OpenapiJobProperties(**remex_slurm)

        # These others are also minimum required.
        properties.current_working_directory = runtime_directory
        properties.environment = {"DLS_SLURMJOB": dls_slurmjob_version()}

        # These are per-task.
        properties.name = job_name
        properties.standard_output = stdout_filename
        properties.standard_error = stderr_filename

        while True:
            try:
                # Make the slurmjob client context from the specification in the configuration.
                client_context = DlsSlurmjobRestdClientContext(
                    self.__slurmjob_specification
                )

                # Open the slurmjob client context connects to the service process.
                async with client_context as client:
                    # Get jobs.
                    job_id = await client.submit_job(bash_filename, properties)
                    break

            except Exception as exception:
                logger.warning(
                    explain2(
                        exception,
                        f'{callsign(self)} submitting job "{bx_job.label()}" task "{bx_task.label()}"',
                        exc_info=exception,
                    )
                )
                # Sleep before retrying.
                await asyncio.sleep(5)
                # TODO: In slurmer launcher, have a limit to number of retries.
                # raise RemoteSubmitFailed("; ".join(lines))

        logger.debug(
            f"{callsign(self)} submitted job_id {job_id}"
            f' for job "{bx_job.label()}" task "{bx_task.label()}"'
            f" in directory {runtime_directory}"
        )

        # Make a serializable object representing the entity which was launched.
        launch_info = SlurmerLaunchInfo(bx_job, bx_task)
        launch_info.job_id = job_id

        # Let the base class update the objects in the database.
        await self.post_submit(launch_info)

    # ------------------------------------------------------------------------------------------
    def unserialize_launch_info(self, bx_job, bx_task, serialized):
        """Given a serialized string from the database, create a launch info object for our launcher type."""
        unserialized = json.loads(serialized)

        # Create a launch info object for our launcher type.
        launch_info = SlurmerLaunchInfo(bx_job, bx_task)

        # Add the fields which were in the serialized string.
        launch_info.job_id = require(
            "SlurmerLaunchInfo unserialized", unserialized, "job_id"
        )

        return launch_info

    # ------------------------------------------------------------------------------------------
    async def are_done(self, launch_infos):
        """Check for done jobs among the list provided."""

        # Look through all the job infos which we are monitoring.
        job_id_list: List[int] = []
        for launch_info in launch_infos:
            job_id_list.append(launch_info.job_id)

        # Make the slurmjob client context from the specification in the configuration.
        client_context = DlsSlurmjobRestdClientContext(self.__slurmjob_specification)

        # Open the slurmjob client context connects to the service process.
        async with client_context as client:
            # Get jobs of interest.
            slurm_infos = await client.query_jobs(job_id_list)

        done_infos = []
        remaining_infos = []

        # Look through all the job infos which we are monitoring.
        for launch_info in launch_infos:
            job_id = launch_info.job_id

            # This job looks done according to the slurm query?
            if slurm_infos[job_id].is_finished:
                done_infos.append(launch_info)
            else:
                remaining_infos.append(launch_info)

        for info in done_infos:
            logger.debug(
                f"[LAUNDON1] cluster job {info.job_id}"
                f" seems done for bx_job {info.bx_job.uuid()}"
                f" bx_task {info.bx_task.uuid()}"
            )

        for info in remaining_infos:
            logger.debug(
                f"[LAUNDON2] cluster job {info.job_id}"
                f" still remaining for bx_job {info.bx_job.uuid()}"
                f" bx_task {info.bx_task.uuid()}"
            )

        return done_infos, remaining_infos
