import asyncio
import json
import logging
import os
import subprocess
from typing import Dict

# Utilities.
from dls_utilpack.callsign import callsign

# Environment module loader.
from dls_utilpack.module import module_get_environ
from dls_utilpack.require import require

from dls_bxflow_api.bx_launchers.constants import ClassTypes

# Remex (remote execution) API.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Describes a particular launch in terms of the job and task it belongs to.
# Base class for bx_launcher instances.
from dls_bxflow_lib.bx_launchers.base import Base as BxLauncherBase
from dls_bxflow_lib.bx_launchers.base import BaseLaunchInfo

logger = logging.getLogger(__name__)

thing_type = ClassTypes.SBATCHER


# ------------------------------------------------------------------------------------------
class SlurmerLaunchInfo(BaseLaunchInfo):
    """Launch info specific to this launcher type needed to identify a launched task."""

    FINISHED_JOB_STATES = [
        "COMPLETED",  # The job has successfully completed execution.
        "CANCELLED",  # The job was canceled by the user or the system administrator before completion.
        "FAILED",  # The job encountered an error during execution and did not complete successfully.
        "TIMEOUT",  # The job exceeded its time limit and was terminated by the system.
        "NODE_FAIL",  # One or more nodes allocated to the job have failed, leading to job termination.
        "OUT_OF_MEMORY",  # The job exceeded the memory limits and was terminated.
        "PREEMPTED",  # The job was preempted by a higher-priority job or system event.
    ]

    UNFINISHED_JOB_STATES = [
        "PENDING",  # The job is waiting to be scheduled and has not started running yet.
        "RUNNING",  # The job is currently running on a compute node.
        "SUSPENDED",  # The job has been suspended and is temporarily halted. This can happen if the job exceeds resource limits or due to user intervention.
        "COMPLETING",  # The job has finished running, but some post-processing or cleanup tasks are still in progress.
    ]

    def __init__(self, bx_job, bx_task):
        BaseLaunchInfo.__init__(self, bx_job, bx_task)
        self.job_id = None
        self.job_state = None

    def serialize(self):
        """Serialize the launch info for storing in a persistent database."""
        return json.dumps(
            {
                "job_id": self.job_id,
                "job_state": self.job_state,
            }
        )

    def is_finished(
        self,
        squeue_jobs_dict: Dict,
    ) -> bool:
        """
        Return true job is finished.

        Args:
            squeue_jobs_dict (Dict): the squeue jobs list, indexed by job_id

        Returns:
            bool: True if the job's entry in the squeue response indicates the job is finished
        """
        job = squeue_jobs_dict.get(self.job_id)

        # Job not listed in the squeue data?
        if job is None:
            logger.debug(f"[SQUJOB] {self.job_id} done because not in dict")
            # Presume it has finished.
            return True

        self.job_state = job["job_state"]
        # Job's state indicates not finished?
        if self.job_state not in self.UNFINISHED_JOB_STATES:
            logger.debug(f"[SQUJOB] {self.job_id} done because state {self.job_state}")
            return True

        # Else presume the job is still running.
        return False


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
        self.__cluster_project = require(
            f"{callsign(self)} specification type_specific_tbd",
            self.specification().get("type_specific_tbd", {}),
            "cluster_project",
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
        self, bx_job_uuid, bx_job_specification, bx_task_uuid, bx_task_specification
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

        command = []
        command.extend(["sbatch"])
        command.extend(["-N", job_name])
        if self.__cluster_project is not None:
            command.extend(["-P", self.__cluster_project])
        command.extend(["-now", "no"])
        command.extend(["-cwd"])

        # The task may specify remex hints to help select the cluster affinity.
        remex_hints = bx_task_specification.get("remex_hints", None)
        if remex_hints is None:
            remex_hints = {}

        # Options for sbatch based on the remex hints.
        sbatch_options = {}
        sbatch_l_options = {}

        # The following keywords are honored:
        # cluster
        # redhat_release
        # m_mem_free
        # h_rt
        # q
        # pe
        # gpu
        # gpu_arch
        # h

        # TODO: Check that the launcher's remex_hints match the task specification's.
        # cluster = remex_hints.get(RemexKeywords.CLUSTER, "")

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.REDHAT_RELEASE)
        if t is not None:
            sbatch_l_options["redhat_release"] = t

        # -------------------------------------------------------------------------------
        memory_limit = remex_hints.get(RemexKeywords.MEMORY_LIMIT)
        if memory_limit is not None:
            gigabytes = str(memory_limit)

            if gigabytes.endswith("G") or gigabytes.endswith("g"):
                gigabytes = gigabytes[:-1]

            if not gigabytes.isnumeric():
                raise RuntimeError(
                    f"cannot parse {callsign(bx_task)}"
                    f' remex_hints[{RemexKeywords.MEMORY_LIMIT}] "{memory_limit}"'
                )

            sbatch_l_options["m_mem_free"] = f"{gigabytes}G"

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.TIME_LIMIT)
        if t is not None:

            sbatch_l_options["h_rt"] = t

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.QUEUE)
        if t is not None:
            sbatch_options["-q"] = t

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.PARALLEL_ENVIRONMENT)
        if t is not None:
            t = t.split(" ")
            if len(t) != 2:
                raise RuntimeError(
                    f"{RemexKeywords.PARALLEL_ENVIRONMENT} should be of two parts separated by a space"
                )
            sbatch_options["-pe"] = t[0].strip()
            sbatch_options[">-pe"] = t[1].strip()

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.GPU)
        if t is not None:
            sbatch_l_options["gpu"] = t

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.GPU_ARCH)
        if t is not None:
            sbatch_l_options["gpu_arch"] = t

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.HOST)
        if t is not None:
            sbatch_l_options["h"] = t

        # -------------------------------------------------------------------------------
        # Add the sbatch_options to the command line.
        for sbatch_option, sbatch_value in sbatch_options.items():
            if not sbatch_option.startswith(">"):
                command.append(sbatch_option)
            if sbatch_value is not None and sbatch_value != "":
                command.append(sbatch_value)

        # Add the sbatch "-l" and its sub-options to the command line.
        if len(sbatch_l_options) > 0:
            sbatch_l_values = []
            for sbatch_l_option, sbatch_l_value in sbatch_l_options.items():
                sbatch_l_values.append(f"{sbatch_l_option}={sbatch_l_value}")
            command.append("-l")
            command.append(",".join(sbatch_l_values))

        # -------------------------------------------------------------------------------
        command.extend(["-o", stdout_filename])
        command.extend(["-e", stderr_filename])
        command.extend(["-terse"])
        command.extend([bash_filename])

        # command_string = bash_filename

        sbatchout_filename = "%s/.bxflow/sbatchout.txt" % (runtime_directory)
        sbatcherr_filename = "%s/.bxflow/sbatcherr.txt" % (runtime_directory)

        # Split the command into arguments/values for readability in the debug.
        readable = (" ".join(command)).replace(" -", " \\\n    -")
        logger.debug(f"{callsign(self)} running command\n{readable}")

        while True:
            with open(sbatchout_filename, "wt") as sbatchout_handle:
                with open(sbatcherr_filename, "wt") as sbatcherr_handle:
                    try:
                        # Wait until sbatch command completes.
                        # TODO: In slurmer, use asyncio to run sbatch.
                        completed = subprocess.run(
                            command,
                            shell=False,
                            # input=command_string,
                            text=True,
                            cwd=runtime_directory,
                            stdout=sbatchout_handle,
                            stderr=sbatcherr_handle,
                        )
                    except Exception:
                        raise RuntimeError("failed to execute the sbatch command")

            # The sbatch command ran ok.
            if completed.returncode == 0:
                break

            # The sbatch command ran, but it indicates there was a problem.
            lines = []
            with open(sbatcherr_filename, "r") as stream:
                for line in stream:
                    line = line.strip()
                    if line == "":
                        continue
                    if line.startswith("Waiting for "):
                        continue
                    lines.append(line)

            if len(lines) == 0:
                lines.append(f"for cause, see {sbatcherr_filename}")

            # lines.append(f"executing command: {command_string}")
            # lines.append("piped into sbatch: %s" % (" ".join(command)))

            logger.warning(callsign(self, "\n    ".join(lines)))

            # Sleep before retrying.
            await asyncio.sleep(5)

            # TODO: In slurmer launcher, have a limit to number of retries.
            # raise RemoteSubmitFailed("; ".join(lines))

        with open(sbatchout_filename, "r") as stream:
            job_id = stream.read().strip()

        logger.debug(
            f"{callsign(self)} submitted job_id {job_id} for task {bx_task_uuid}"
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

        squeue_command = []
        squeue_command.append("squeue")
        squeue_command.append("--json")

        try:
            # Wait until shell script completes.
            completed = subprocess.run(
                squeue_command,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            raise RuntimeError(
                "failed to execute the squeue command '%s'" % (" ".join(squeue_command))
            )

        # Check the return code from squeue.
        if completed.returncode != 0:
            logger.debug(
                "squeue command '%s' got unexpected returncode %s, stderr was:\n%s"
                % (
                    " ".join(squeue_command),
                    completed.returncode,
                    completed.stderr.decode(),
                )
            )
            raise RuntimeError("squeue failed")

        # logger.debug("squeue stdout was:\n%s" % (completed.stdout.decode()))

        # Parse the squeue json report.
        stdout = completed.stdout.decode()
        try:
            squeue_dict = json.loads(stdout)
        except Exception:
            logger.debug(f"squeue response:\n{stdout}")
            raise RuntimeError("squeue response cannot be parsed as json (see log)")

        errors = squeue_dict.get("errors", [])
        if len(errors) > 0:
            raise RuntimeError(
                f"squeue response errors field is not empty\n{str(errors)}"
            )

        if "jobs" not in squeue_dict:
            logger.debug(f"squeue response:\n{stdout}")
            raise RuntimeError("squeue response does not contain jobs field (see log)")

        squeue_jobs_dict = {}
        for job in squeue_dict["jobs"]:
            squeue_jobs_dict[job["job_id"]] = job

        done_infos = []
        remaining_infos = []

        # Look through all the job infos which we are monitoring.
        for launch_info in launch_infos:
            # This job looks done according to the squeue output?
            if launch_info.is_finished(squeue_jobs_dict):
                done_infos.append(launch_info)
            else:
                remaining_infos.append(launch_info)

        for info in done_infos:
            logger.debug(
                f"[LAUNDON1] cluster job {info.job_id}"
                f" state {info.job_state}"
                f" seems done for bx_job {info.bx_job.uuid()}"
                f" bx_task {info.bx_task.uuid()}"
            )

        for info in remaining_infos:
            logger.debug(
                f"[LAUNDON2] cluster job {info.job_id}"
                f" state {info.job_state}"
                f" still remaining for bx_job {info.bx_job.uuid()}"
                f" bx_task {info.bx_task.uuid()}"
            )

        return done_infos, remaining_infos
