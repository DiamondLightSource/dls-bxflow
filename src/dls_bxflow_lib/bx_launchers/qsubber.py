import asyncio
import json
import logging
import os
import subprocess

# Utilities.
from dls_utilpack.callsign import callsign

# Environment module loader.
from dls_utilpack.module import module_get_environ
from dls_utilpack.require import require

# Remex (remote execution) API.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Describes a particular launch in terms of the job and task it belongs to.
# Base class for bx_launcher instances.
from dls_bxflow_lib.bx_launchers.base import Base as BxLauncherBase
from dls_bxflow_lib.bx_launchers.base import BaseLaunchInfo

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_launchers.qsubber"


# ------------------------------------------------------------------------------------------
class QsubberLaunchInfo(BaseLaunchInfo):
    """Launch info specific to this launcher type needed to identify a launched task."""

    def __init__(self, bx_job, bx_task):
        BaseLaunchInfo.__init__(self, bx_job, bx_task)
        self.job_number = None

    def serialize(self):
        """Serialize the launch info for storing in a persistent database."""
        return json.dumps({"job_number": self.job_number})


# ------------------------------------------------------------------------------------------
class Qsubber(BxLauncherBase):
    """
    Object representing a bx_launcher which launches a task using qsub for cluster execution.
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
    async def __submit_OLD(
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
        # stdout_filename = "%s/stdout.txt" % (runtime_directory)
        # stderr_filename = "%s/stderr.txt" % (runtime_directory)

        command = []
        command.extend(["qsub"])
        command.extend(["-N", job_name])
        command.extend(["-P", self.__cluster_project])
        command.extend(["-now", "no"])
        command.extend(["-cwd"])

        # The task may specify remex hints to help select the cluster affinity.
        remex_hints = bx_task_specification.get("remex_hints", None)
        if remex_hints is None:
            remex_hints = {}

        # Options for qsub based on the remex hints.
        qsub_options = {}
        qsub_l_options = {}
        qsub_l_options["m_mem_free"] = "64G"
        qsub_l_options["h_rt"] = "8:00:00"

        cluster = remex_hints.get(RemexKeywords.CLUSTER, "")
        if cluster in [RemexClusters.SCIENCE, RemexClusters.TEST]:
            qsub_options["-q"] = "high.q"
            qsub_options["-pe"] = "smp"
            qsub_options[">-pe"] = "1"

        if cluster == RemexClusters.HAMILTON and RemexKeywords.PTYPY_MPI in remex_hints:
            # This is from https://github.com/DiamondLightSource/PtychographyTools/tree/master/ptychotools/ptychotools.ptypy_launcher
            #   TOTAL_NUM_PROCESSORS=$(( NUM_GPU * 10 ));
            #   NUM_PROCS_PER_NODE=$(( 4 < NUM_GPU ? 4 : NUM_GPU )); # can be maximum 4
            #   EXTRA_ARGS="$EXTRA_ARGS -g"
            #   JOB_NAME="ptypy_gpu"
            #   MEMORY_REQUEST=8G
            #   qsub_args="-pe openmpi $TOTAL_NUM_PROCESSORS -l gpu=$NUM_PROCS_PER_NODE,m_mem_free=$MEMORY_REQUEST,gpu_arch=$GPU_ARCH,h=!(cs05r-sc-gpu01-02.diamond.ac.uk|cs05r-sc-gpu01-01.diamond.ac.uk) -N $JOB_NAME"

            ptypy_mpi_hints = remex_hints[RemexKeywords.PTYPY_MPI]
            if not isinstance(ptypy_mpi_hints, dict):
                ptypy_mpi_hints = {}
            num_gpu = ptypy_mpi_hints.get(RemexKeywords.NUM_GPU, 1)
            if num_gpu > 4:
                num_gpu = 4
            openmpi = ptypy_mpi_hints.get(RemexKeywords.OPENMPI, num_gpu * 10)
            qsub_options["-pe"] = "openmpi"
            qsub_options[">-pe"] = str(openmpi)

            qsub_l_options["m_mem_free"] = "8G"
            qsub_l_options["gpu"] = str(num_gpu)
            qsub_l_options["gpu_arch"] = ptypy_mpi_hints.get("gpu_arch", "Volta")
            qsub_l_options[
                "h"
            ] = "!(cs05r-sc-gpu01-02.diamond.ac.uk|cs05r-sc-gpu01-01.diamond.ac.uk)"

        if (
            cluster == RemexClusters.HAMILTON
            and RemexKeywords.PTYREX_MPI in remex_hints
        ):
            # This is from communication from Mohsen:
            # $ -l h_rt=01:00:00,gpu=4
            # $ -pe openmpi 4

            ptyrex_mpi_hints = remex_hints[RemexKeywords.PTYREX_MPI]
            if not isinstance(ptyrex_mpi_hints, dict):
                ptyrex_mpi_hints = {}
            num_gpu = ptyrex_mpi_hints.get(RemexKeywords.NUM_GPU, 4)
            if num_gpu > 4:
                num_gpu = 4
            openmpi = ptyrex_mpi_hints.get(RemexKeywords.OPENMPI, num_gpu)
            qsub_options["-pe"] = "openmpi"
            qsub_options[">-pe"] = str(openmpi)

            qsub_l_options["m_mem_free"] = "64G"
            qsub_l_options["gpu"] = str(num_gpu)
            qsub_l_options["gpu_arch"] = ptyrex_mpi_hints.get("gpu_arch", "Volta")

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

            qsub_l_options["m_mem_free"] = f"{gigabytes}G"

        # -------------------------------------------------------------------------------
        time_limit = remex_hints.get(RemexKeywords.TIME_LIMIT)
        if time_limit is not None:
            parts = str(time_limit).split(":")
            if len(parts) == 1:
                parts.append("0")

            if len(parts) != 2 or not parts[0].isnumeric() or not parts[1].isnumeric():
                raise RuntimeError(
                    f"cannot parse {callsign(bx_task)}"
                    f' remex_hints[{RemexKeywords.TIME_LIMIT}] "{time_limit}"'
                )

            qsub_l_options["h_rt"] = "%d:%02d:00" % (int(parts[0]), int(parts[1]))

        # -------------------------------------------------------------------------------
        redhat_release = remex_hints.get(RemexKeywords.REDHAT_RELEASE)
        if redhat_release is not None:
            qsub_l_options["redhat_release"] = redhat_release

        # -------------------------------------------------------------------------------
        # Add the qsub_options to the command line.
        for qsub_option, qsub_value in qsub_options.items():
            if not qsub_option.startswith(">"):
                command.append(qsub_option)
            if qsub_value is not None and qsub_value != "":
                command.append(qsub_value)

        # Add the qsub "-l" and its sub-options to the command line.
        if len(qsub_l_options) > 0:
            qsub_l_values = []
            for qsub_l_option, qsub_l_value in qsub_l_options.items():
                qsub_l_values.append(f"{qsub_l_option}={qsub_l_value}")
            command.append("-l")
            command.append(",".join(qsub_l_values))

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
        command.extend(["qsub"])
        command.extend(["-N", job_name])
        if self.__cluster_project is not None:
            command.extend(["-P", self.__cluster_project])
        command.extend(["-now", "no"])
        command.extend(["-cwd"])

        # The task may specify remex hints to help select the cluster affinity.
        remex_hints = bx_task_specification.get("remex_hints", None)
        if remex_hints is None:
            remex_hints = {}

        # Options for qsub based on the remex hints.
        qsub_options = {}
        qsub_l_options = {}

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
            qsub_l_options["redhat_release"] = t

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

            qsub_l_options["m_mem_free"] = f"{gigabytes}G"

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.TIME_LIMIT)
        if t is not None:

            qsub_l_options["h_rt"] = t

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.QUEUE)
        if t is not None:
            qsub_options["-q"] = t

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.PARALLEL_ENVIRONMENT)
        if t is not None:
            t = t.split(" ")
            if len(t) != 2:
                raise RuntimeError(
                    f"{RemexKeywords.PARALLEL_ENVIRONMENT} should be of two parts separated by a space"
                )
            qsub_options["-pe"] = t[0].strip()
            qsub_options[">-pe"] = t[1].strip()

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.GPU)
        if t is not None:
            qsub_l_options["gpu"] = t

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.GPU_ARCH)
        if t is not None:
            qsub_l_options["gpu_arch"] = t

        # -------------------------------------------------------------------------------
        t = remex_hints.get(RemexKeywords.HOST)
        if t is not None:
            qsub_l_options["h"] = t

        # -------------------------------------------------------------------------------
        # Add the qsub_options to the command line.
        for qsub_option, qsub_value in qsub_options.items():
            if not qsub_option.startswith(">"):
                command.append(qsub_option)
            if qsub_value is not None and qsub_value != "":
                command.append(qsub_value)

        # Add the qsub "-l" and its sub-options to the command line.
        if len(qsub_l_options) > 0:
            qsub_l_values = []
            for qsub_l_option, qsub_l_value in qsub_l_options.items():
                qsub_l_values.append(f"{qsub_l_option}={qsub_l_value}")
            command.append("-l")
            command.append(",".join(qsub_l_values))

        # -------------------------------------------------------------------------------
        command.extend(["-o", stdout_filename])
        command.extend(["-e", stderr_filename])
        command.extend(["-terse"])
        command.extend([bash_filename])

        # command_string = bash_filename

        qsubout_filename = "%s/.bxflow/qsubout.txt" % (runtime_directory)
        qsuberr_filename = "%s/.bxflow/qsuberr.txt" % (runtime_directory)

        # Split the command into arguments/values for readability in the debug.
        readable = (" ".join(command)).replace(" -", " \\\n    -")
        logger.debug(f"{callsign(self)} running command\n{readable}")

        while True:
            with open(qsubout_filename, "wt") as qsubout_handle:
                with open(qsuberr_filename, "wt") as qsuberr_handle:
                    try:
                        # Wait until qsub command completes.
                        # TODO: In qsubber, use asyncio to run qsub.
                        completed = subprocess.run(
                            command,
                            shell=False,
                            # input=command_string,
                            text=True,
                            cwd=runtime_directory,
                            stdout=qsubout_handle,
                            stderr=qsuberr_handle,
                        )
                    except Exception:
                        raise RuntimeError("failed to execute the qsub command")

            # The qsub command ran ok.
            if completed.returncode == 0:
                break

            # The qsub command ran, but it indicates there was a problem.
            lines = []
            with open(qsuberr_filename, "r") as stream:
                for line in stream:
                    line = line.strip()
                    if line == "":
                        continue
                    if line.startswith("Waiting for "):
                        continue
                    lines.append(line)

            if len(lines) == 0:
                lines.append(f"for cause, see {qsuberr_filename}")

            # lines.append(f"executing command: {command_string}")
            # lines.append("piped into qsub: %s" % (" ".join(command)))

            logger.warning(callsign(self, "\n    ".join(lines)))

            # Sleep before retrying.
            await asyncio.sleep(5)

            # TODO: In qsubber launcher, have a limit to number of retries.
            # raise RemoteSubmitFailed("; ".join(lines))

        with open(qsubout_filename, "r") as stream:
            job_number = stream.read().strip()

        logger.debug(
            f"{callsign(self)} submitted job_number {job_number} for task {bx_task_uuid}"
        )

        # Make a serializable object representing the entity which was launched.
        launch_info = QsubberLaunchInfo(bx_job, bx_task)
        launch_info.job_number = job_number

        # Let the base class update the objects in the database.
        await self.post_submit(launch_info)

    # ------------------------------------------------------------------------------------------
    def unserialize_launch_info(self, bx_job, bx_task, serialized):
        """Given a serialized string from the database, create a launch info object for our launcher type."""
        unserialized = json.loads(serialized)

        # Create a launch info object for our launcher type.
        launch_info = QsubberLaunchInfo(bx_job, bx_task)

        # Add the fields which were in the serialized string.
        launch_info.job_number = require(
            "QsubberLaunchInfo unserialized", unserialized, "job_number"
        )

        return launch_info

    # ------------------------------------------------------------------------------------------
    async def are_done(self, launch_infos):
        """Check for done jobs among the list provided."""

        qstat = []
        qstat.append("qstat")
        launch_info_job_numbers = []
        for launch_info in launch_infos:
            launch_info_job_numbers.append(launch_info.job_number)

        qstat.extend(["-j", ",".join(launch_info_job_numbers)])

        try:
            # Wait until shell script completes.
            completed = subprocess.run(
                qstat,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            raise RuntimeError("failed to execute the qstat command")

        # You get an completioncode of 1 if there are no results.
        if completed.returncode not in [0, 1]:
            logger.debug(
                "qstat got unexpected returncode %s, stderr was:\n%s"
                % (completed.returncode, completed.stderr.decode())
            )
            raise RuntimeError("qstat failed")

        # logger.debug("qstat stdout was:\n%s" % (completed.stdout.decode()))

        lines = completed.stdout.decode().split("\n")

        qstat_job_numbers = []
        for line in lines:
            if not line.startswith("job_number"):
                continue
            parts = line.split(" ", 1)
            qstat_job_numbers.append(parts[1].strip())

        done_infos = []
        remaining_infos = []
        for launch_info in launch_infos:
            if launch_info.job_number not in qstat_job_numbers:
                done_infos.append(launch_info)
            else:
                remaining_infos.append(launch_info)

        for done_info in done_infos:
            logger.debug(
                f"[LAUNDON1] cluster job {done_info.job_number}"
                f" seems done for bx_job {done_info.bx_job.uuid()}"
                f" bx_task {done_info.bx_task.uuid()}"
            )

        for remaining_info in remaining_infos:
            logger.debug(
                f"[LAUNDON2] cluster job {remaining_info.job_number}"
                f" still remaining for bx_job {remaining_info.bx_job.uuid()}"
                f" bx_task {remaining_info.bx_task.uuid()}"
            )

        return done_infos, remaining_infos
