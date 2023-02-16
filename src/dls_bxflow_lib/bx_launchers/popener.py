import json
import logging
import os
import socket
from subprocess import Popen

import psutil

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Describes a particular launch in terms of the job and task it belongs to.
# Base class for bx_launcher instances.
from dls_bxflow_lib.bx_launchers.base import Base as BxLauncherBase
from dls_bxflow_lib.bx_launchers.base import BaseLaunchInfo

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_launchers.popener"


# ------------------------------------------------------------------------------------------
class PopenerLaunchInfo(BaseLaunchInfo):
    """Launch info specific to this launcher type needed to identify a launched task."""

    def __init__(self, bx_job, bx_task):
        BaseLaunchInfo.__init__(self, bx_job, bx_task)
        self.hostname = None
        self.pid = None

    def serialize(self):
        """Serialize the launch info for storing in a persistent database."""
        return json.dumps({"hostname": self.hostname, "pid": self.pid})


# ------------------------------------------------------------------------------------------
class Popener(BxLauncherBase):
    """
    Object representing a bx_launcher which launches a task using popen for onboard execution.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification, predefined_uuid=None):
        BxLauncherBase.__init__(
            self, thing_type, specification, predefined_uuid=predefined_uuid
        )

        self.__hostname = socket.gethostname()

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""
        return "%s %s" % ("BxLauncher.Popener", self.uuid())

    # ------------------------------------------------------------------------------------------
    async def submit(
        self, bx_job_uuid, bx_job_specification, bx_task_uuid, bx_task_specification
    ):
        """Submit bx_task for execution."""

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

        # Let the specifiation say if the current environment is to be passed.
        # This generally True for testing, but False otherwise.
        should_pass_environ = (
            self.specification()
            .get("type_specific_tbd", {})
            .get("should_pass_environ", False)
        )

        if should_pass_environ:
            environ = os.environ
        else:
            environ = {}

        stdout_filename = "%s/stdout.txt" % (runtime_directory)
        stderr_filename = "%s/stderr.txt" % (runtime_directory)
        with open(stdout_filename, "wt") as stdout_handle:
            with open(stderr_filename, "wt") as stderr_handle:
                # Start but don't wait.
                popen = Popen(
                    bash_filename,
                    shell=True,
                    env=environ,
                    cwd=runtime_directory,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )

                logger.debug(f"{callsign(self)} submitted pid {popen.pid}")

                # Make an object describing the entity which was launched.
                launch_info = PopenerLaunchInfo(bx_job, bx_task)
                launch_info.hostname = self.__hostname
                launch_info.pid = popen.pid

                # Let the base class do the post-submit stuff.
                # This will typically be to keep of list of processes it has started.
                await self.post_submit(launch_info)

    # ------------------------------------------------------------------------------------------
    def unserialize_launch_info(self, bx_job, bx_task, serialized):
        """Given a serialized string from the database, create a launch info object for our launcher type."""
        unserialized = json.loads(serialized)

        # Create a launch info object for our launcher type.
        launch_info = PopenerLaunchInfo(bx_job, bx_task)

        # Add the fields which were in the serialized string.
        launch_info.hostname = require(
            "PopenerLaunchInfo unserialized", unserialized, "hostname"
        )
        launch_info.pid = require("PopenerLaunchInfo unserialized", unserialized, "pid")
        return launch_info

    # ------------------------------------------------------------------------------------------
    async def are_done(self, launch_infos):
        """
        Check for done processes among the list provided.
        Return two lists: those that are done and those that are not done.
        """

        done_infos = []
        remaining_infos = []
        for launch_info in launch_infos:
            # TODO: Figure out what to do if popener orphan recovery hostname doesn't match.
            pid = launch_info.pid

            # TODO: Check if popener launcher can check for done processes using asyncio instead of polling, even if some are adopted orphans.
            try:
                # Try to wrap the pid in a process object.
                process = psutil.Process(pid=pid)
            except psutil.NoSuchProcess:
                done_infos.append(launch_info)
                logger.debug(
                    f"[ADDORPH] {callsign(self)} sees {launch_info.serialize()} not existing, so presumed finished"
                )
                continue

            try:
                # Wait for the process.
                # If it was a child of the current process, this will clean up the zombie.
                waitrc = process.wait(timeout=0)

                # If waitrc returns without timeout, it means the process is done.
                # We "could" get the exit_code here, but other launcher types don't have this same opportunity.
                # Instead the main_isolated will record the exit_code in a separate residual file.
                done_infos.append(launch_info)
                logger.debug(
                    f"[ADDORPH] {callsign(self)} sees {launch_info.serialize()} finished with psutil waitrc {waitrc}"
                )
            except psutil.TimeoutExpired:

                # The process is still alive, but a zombie?
                if process.status() == psutil.STATUS_ZOMBIE:

                    # If process is a zombie, that means some launcher instance started it.
                    # This is tested in test_launcher_restart.
                    # It could happen in the wild if a launcher uuid gets restarted.
                    # TODO: Deal with zombies left behind by popener launchers which die.
                    done_infos.append(launch_info)
                    logger.debug(
                        f"[ADDORPH] {callsign(self)} sees {launch_info.serialize()} is a zombine"
                    )
                else:
                    logger.debug(
                        f"[ADDORPH] {callsign(self)} sees {launch_info.serialize()} not yet finished"
                    )
                    remaining_infos.append(launch_info)

        return done_infos, remaining_infos
