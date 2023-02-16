import asyncio
import logging
import os
import stat

import yaml

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.explain import explain, explain_cause_chain_error_lines
from dls_utilpack.require import require

# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Exceptions.
from dls_bxflow_api.exceptions import CapacityReached

# Base class which maps flask bx_tasks to methods.
from dls_bxflow_api.thing import Thing

# Base class for an aiohttp server.
from dls_bxflow_lib.bx_catalogs.bx_catalogs import (
    bx_catalogs_get_default,
    bx_catalogs_has_default,
)

# BxJobs manager.
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs

# Possible bx_launcher states.
from dls_bxflow_lib.bx_launchers.states import States as BxLauncherStates

# BxTasks manager.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

# BxTask configuration keywords.
from dls_bxflow_run.bx_tasks.constants import Keywords as BxTaskKeywords
from dls_bxflow_run.bx_tasks.states import States as BxTaskStates


class EXIT_CODES:
    PREPARE_ENVIRONMENT = 252


logger = logging.getLogger(__name__)


class BaseLaunchInfo:
    """Describes a particular launch in terms of the job and task it belongs to."""

    def __init__(self, bx_job, bx_task):
        self.bx_job = bx_job
        self.bx_task = bx_task


# ------------------------------------------------------------------------------------------
class Base(Thing):
    """
    Object representing a bx_launcher which receives bx_tasks from aiohttp.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, thing_type, specification=None, predefined_uuid=None):
        if predefined_uuid is None:
            predefined_uuid = require(
                f"{thing_type} specification", specification, "uuid"
            )
            logger.debug(
                f"[LAUNUUID] launcher specified predefined_uuid is {predefined_uuid}"
            )
        else:
            logger.debug(
                f"[LAUNUUID] launcher initialized predefined_uuid is {predefined_uuid}"
            )
        Thing.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.__bx_dataface = bx_datafaces_get_default()

        self._bx_jobs = BxJobs()
        self._bx_tasks = BxTasks()

        self.__launch_infos = []

        # Number of concurrent tasks we will support.
        # TODO: In aiohttp rBxLauncher, make task count max configurable.
        self.__task_count_max = specification["type_specific_tbd"].get(
            "task_count_max", 10
        )
        self.__task_count_now = 0

        self.__harvest_lock = asyncio.Lock()

    # ----------------------------------------------------------------------------------------
    async def activate(self):
        """"""
        await self.adopt_orphans()

    # ----------------------------------------------------------------------------------------
    async def deactivate(self):
        """"""
        pass

    # ----------------------------------------------------------------------------------------
    async def adopt_orphans(self):
        """"""

        # Get all tasks launched by a previous instance of this launcher.
        task_records = await self.__bx_dataface.get_bx_tasks_launched_by(
            self.uuid(), states=[BxTaskStates.STARTED]
        )

        # Adopt orphan task.
        for task_record in task_records:
            await self.adopt_orphan(task_record)

    # ----------------------------------------------------------------------------------------
    async def adopt_orphan(self, task_record):
        """"""

        async with self.__harvest_lock:
            # Increment task count.
            self.__task_count_now += 1

            bx_job_uuid = task_record["bx_job_uuid"]
            job_record = await self.__bx_dataface.get_bx_job(bx_job_uuid)

            # Make a bx_job object.
            bx_job = self._bx_jobs.build_object(
                job_record["specification"], predefined_uuid=bx_job_uuid
            )

            # Make a bx_task object.
            bx_task = self._bx_tasks.build_object(
                task_record["specification"], predefined_uuid=task_record["uuid"]
            )

            # Associate task with job.  Why?
            bx_task.bx_job_uuid(bx_job_uuid)

            # Make a launch_info object describing the launched code.
            launch_info = self.unserialize_launch_info(
                bx_job, bx_task, task_record["launch_info"]
            )

            # Add to list of harvestable handles.
            self.__launch_infos.append(launch_info)

            logger.debug(
                f"[ADDORPH] adding {task_record['type']} orphan task {task_record['label']}"
                f" with launch_info {task_record['launch_info']}"
            )

    # ------------------------------------------------------------------------------------------
    # Handle request to submit bx_task for execution.

    async def presubmit(
        self, bx_job_uuid, bx_job_specification, bx_task_uuid, bx_task_specification
    ):
        # Increment task count.
        if self.__task_count_now == self.__task_count_max:
            # Reply to client.
            # TODO: In aiohttp rBxLauncher, give special http status to indicate task count max exceeded.
            raise CapacityReached(
                callsign(
                    self,
                    f"[TASKCNT] cannot submit a new task because {self.__task_count_max} already underway in this launcher",
                )
            )

        self.__task_count_now += 1

        if self.__task_count_now == self.__task_count_max:
            # Update our own state to busy so we won't be selected for more work.
            self.set_state(BxLauncherStates.BUSY)
            await self.__bx_dataface.update_bx_launcher(
                {"uuid": self.uuid(), "state": self.get_state()}
            )

        logger.debug(
            callsign(
                self,
                f"[TASKCNT] task count now {self.__task_count_now} of max {self.__task_count_max}",
            )
        )

        # Update the state of the bx_task we are about to run.
        await self.__bx_dataface.update_bx_task(
            {
                "uuid": bx_task_uuid,
                "state": BxTaskStates.STARTING,
                "bx_launcher_uuid": self.uuid(),
            }
        )

        # Make a bx_job object.
        bx_job = self._bx_jobs.build_object(
            bx_job_specification, predefined_uuid=bx_job_uuid
        )

        # Make a bx_task object.
        bx_task = self._bx_tasks.build_object(
            bx_task_specification, predefined_uuid=bx_task_uuid
        )
        bx_task.bx_job_uuid(bx_job_uuid)

        # Variables used to pass information into and between tasks.
        await bx_task.variables.fetch(bx_job_uuid)

        return self.presubmit2(bx_job, bx_task)

    # ------------------------------------------------------------------------------------------
    # Start a process to load and run the task inside itself and wait for it to finish.
    # A unique subdirectory will be created and use for all output files.
    def presubmit2(self, bx_job, bx_task):

        bxflow_configuration = {}

        # Provide the task's specification so the task can be instantiated on the runtime platform.
        bxflow_configuration["bx_task_specification"] = bx_task.specification()
        # Include the job uuid in the "specification".
        bxflow_configuration["bx_task_specification"]["bx_job_uuid"] = bx_job.uuid()

        needs_dataface = bxflow_configuration["bx_task_specification"].get(
            BxTaskKeywords.NEEDS_DATAFACE, False
        )
        if needs_dataface:
            # Provide the dataface's specification in case the task needs to talk to it.
            # For example, by run-time variables, runtime gate changes, or runtime news.
            bxflow_configuration[
                "bx_dataface_specification"
            ] = bx_datafaces_get_default().specification()

            # Transfer the task's variables to the yaml configuration.
            # Note you need the dataface for the variables mechanism to work.
            bx_variables = []
            for bx_variable in bx_task.variables.list():
                bx_variables.append(
                    {
                        "name": bx_variable.trait("name"),
                        "value": bx_variable.trait("value"),
                    }
                )

            bxflow_configuration["bx_variables"] = bx_variables

        # ------------------------------------------------------------------------
        # This is the directory where we run.
        # All logs, stdout, stderr, and output files should be put here.
        runtime_directory = bx_task.get_directory()

        # Create the directory.
        if os.path.exists(runtime_directory):
            raise RuntimeError(
                f"unable to run {callsign(bx_task)}"
                f" because its directory exists {runtime_directory}"
            )
        else:
            try:
                os.makedirs(f"{runtime_directory}/.bxflow")
            except Exception as exception:
                raise RuntimeError(
                    explain(exception, f"creating {runtime_directory}/.bxflow")
                )

        # Write the bxflow runtime specifications.
        main_yaml_filename = f"{runtime_directory}/.bxflow/main.yaml"
        try:
            with open(main_yaml_filename, "wt") as yaml_stream:
                yaml.dump(
                    bxflow_configuration,
                    yaml_stream,
                    default_flow_style=False,
                    sort_keys=False,
                )
        except Exception as exception:
            raise RuntimeError(explain(exception, f"writing {main_yaml_filename}"))

        bash_name = "main.sh"
        # Build a bash script which sets up the necessary module, conda, venv or whatever.
        # TODO: Make the task specification contain pythonpath, module, conda or venv.
        # TODO: Use the utilpack library for building the main.sh bash script.
        bash_lines = []
        bash_lines.append("#/bin/bash")

        keyword = BxTaskKeywords.PREPARE_ENVIRONMENT
        prepare_log = ".bxflow/prepare_environment.txt"
        exit_code_txt = ".bxflow/exit_code.txt"

        # Let the script capture some information about the host running the task.
        bash_lines.append(f"echo '-----------------' >> {prepare_log}")
        bash_lines.append(f"echo 'hostname: ' `hostname`>> {prepare_log}")
        bash_lines.append(f"echo 'uname: ' `uname -a` >> {prepare_log}")
        bash_lines.append(f"echo 'id: ' `id` >> {prepare_log}")
        bash_lines.append(
            f"echo 'date: ' `date +'%Y-%m-%d %H:%M:%S %z'` >> {prepare_log}"
        )
        bash_lines.append(f"echo 'pwd: ' `pwd` >> {prepare_log}")

        bash_lines.append(f"echo '-----------------' >> {prepare_log}")

        if keyword not in bx_task.specification():
            bash_lines.append(f"# {bx_task} specification {keyword} is not present")
        else:
            prepare_environment = bx_task.specification().get(keyword)
            if prepare_environment is None:
                bash_lines.append(f"# {bx_task} specification {keyword} is None")
            else:
                bash_lines.append(f"# {bx_task} specification {keyword}:")

                for prepare_line in prepare_environment:
                    bash_lines.append("")
                    bash_lines.append('t1=`date "+%s.%N"`')
                    bash_lines.append(f"echo '{prepare_line}' >>{prepare_log}")
                    bash_lines.append(f"{prepare_line} >>{prepare_log} 2>&1")
                    bash_lines.append("rc=$?")
                    bash_lines.append('t2=`date "+%s.%N"`')
                    # Arithmetic in POSIX shells is done with $ and double parentheses (( )):
                    bash_lines.append(
                        "echo '-----------------'"
                        " `echo $t1 $t2 | awk '{printf \"%%0.3f\", $2-$1}'` seconds >> %s"
                        % (prepare_log)
                    )
                    bash_lines.append("if [ $rc -ne 0 ]")
                    bash_lines.append("then")
                    bash_lines.append(
                        f"  echo {EXIT_CODES.PREPARE_ENVIRONMENT} >> {exit_code_txt}"
                    )
                    bash_lines.append(f"  >&2 echo error doing: {prepare_line}")
                    bash_lines.append(
                        f"  >&2 echo see {runtime_directory}/{prepare_log}"
                    )
                    bash_lines.append(f"  exit $(({EXIT_CODES.PREPARE_ENVIRONMENT}))")
                    bash_lines.append("fi")

        bash_lines.append(f"echo '-----------------' >> {prepare_log}")
        bash_lines.append(f"echo 'env:' >> {prepare_log}")
        bash_lines.append(f"env | sort >> {prepare_log}")

        bash_lines.append("")
        command_line = "python3 -m dls_bxflow_run.main_isolated"
        bash_lines.append('t1=`date "+%s.%N"`')
        bash_lines.append(f"echo '{command_line}' >>{prepare_log}")
        bash_lines.append(command_line)
        bash_lines.append("exit_code=$?")
        bash_lines.append('t2=`date "+%s.%N"`')
        bash_lines.append(
            "echo '-----------------'"
            " `echo $t1 $t2 | awk '{printf \"%%0.3f\", $2-$1}'` seconds >> %s"
            % (prepare_log)
        )

        bash_lines.append("")
        bash_lines.append(
            f"echo $exit_code > {runtime_directory}/.bxflow/exit_code.txt"
        )
        bash_lines.append("exit $exit_code")

        # Write the bash script file itself.
        bash_filename = "%s/.bxflow/%s" % (runtime_directory, bash_name)

        with open(bash_filename, "w") as bash_stream:
            bash_stream.write("\n".join(bash_lines))
            bash_stream.write("\n")

        # Give the bash script file execute permission.
        st = os.stat(bash_filename)
        os.chmod(bash_filename, st.st_mode | stat.S_IXUSR)

        return bx_job, bx_task, runtime_directory, bash_filename

    # ------------------------------------------------------------------------------------------
    #
    async def post_submit(self, launch_info):
        """
        Handle when submit is done.
        """

        async with self.__harvest_lock:

            serialized = launch_info.serialize()

            # Update the task record to have launcher handle to be used during orphan adoption.
            record = {
                "uuid": launch_info.bx_task.uuid(),
                "launch_info": serialized,
                "bx_launcher_uuid": self.uuid(),
                "state": BxTaskStates.STARTED,
            }

            try:
                await self.__bx_dataface.update_bx_task(record)
            except Exception as exception:
                explain(exception, f"updating task record {record}")

            # logger.info("[PSTUBS] post_submit handle %s" % (handle))
            self.__launch_infos.append(launch_info)

    # ------------------------------------------------------------------------------------------
    #
    async def harvest(self):
        """
        Harvest finished tasks.
        """

        # Nothing to check for?
        if len(self.__launch_infos) == 0:
            return

        async with self.__harvest_lock:
            # Check which processes in the list are done.
            done_infos, remaining_infos = await self.are_done(self.__launch_infos)

            logger.debug(
                f"[ADDORPH] harvested {len(done_infos)} done and {len(remaining_infos)} remaining"
            )

            for info in done_infos:
                await self._done_callback_async(info.bx_job, info.bx_task)

            self.__launch_infos = remaining_infos

    # ------------------------------------------------------------------------------------------
    #
    def get_unharvested_infos(self):
        """
        Return tasks not yet harvested.
        """

        return self.__launch_infos

    # ------------------------------------------------------------------------------------------
    #
    async def _done_callback_async(self, bx_job, bx_task):
        """
        Handle callback when process is done.
        """

        try:
            # Catalog any artefacts left by the task.
            await self._catalog_artefacts(bx_job, bx_task)

            # Get the task post-run information for the database record.
            exit_code, error_lines, gate_label = self.get_post_run_fields_after_run(
                bx_job,
                bx_task,
            )

            # Update the state of the bx_task we ran.
            await self.__bx_dataface.update_bx_task(
                {
                    "uuid": bx_task.uuid(),
                    "state": BxTaskStates.FINISHED,
                    "exit_code": exit_code,
                    "error_lines": "\n".join(error_lines),
                }
            )

            # Update the gate which allows other tasks to run and/or the job to block.
            await self.__bx_dataface.open_bx_gate(bx_task.uuid(), gate_label)

            # Update the bx_job state if it is now blocked.
            await self.__bx_dataface.update_bx_job_if_blocked_by_bx_gates(
                bx_task.bx_job_uuid()
            )

            # Decrement task count that this service is currently waiting on.
            self.__task_count_now -= 1
            if self.__task_count_now < 0:
                self.__task_count_now = 0

            if self.__task_count_now < self.__task_count_max:
                logger.debug(
                    callsign(
                        self,
                        f"[TASKCNT] task count decremented to {self.__task_count_now}"
                        f" out of max {self.__task_count_max}",
                    )
                )

                # Update our own state to idle again so we can accept more work.
                self.set_state(BxLauncherStates.IDLE)
                await self.__bx_dataface.update_bx_launcher(
                    {"uuid": self.uuid(), "state": self.get_state()}
                )

        except Exception as exception:
            logger.exception(
                callsign(
                    self,
                    explain(
                        exception,
                        "housekeeping after task finished",
                    ),
                ),
                exc_info=exception,
            )

    # ------------------------------------------------------------------------------------------
    def get_post_run_fields_after_run(self, bx_job, bx_task):
        """
        Get task post-run fields after the task finished running.
        """

        # Runtime directory where the isolated task wrote its files.
        runtime_directory = bx_task.get_directory()

        exit_code_filename = f"{runtime_directory}/.bxflow/exit_code.txt"
        if os.path.exists(exit_code_filename):
            with open(exit_code_filename, "r") as stream:
                exit_code = int(stream.read().strip())
            if exit_code == EXIT_CODES.PREPARE_ENVIRONMENT:
                prepare_log = f"{runtime_directory}/.bxflow/prepare_environment.txt"
                error_lines = f"error while preparing environment, see {prepare_log}"
        else:
            exit_code = -2

        # The isolated process exited with 0 returncode?
        if exit_code == 0:
            error_lines = []
            gate_label = "success"

        else:
            # Get task post-run fields after the task finished running.
            error_lines = bx_task.extract_error_lines()
            gate_label = "failure"

        return exit_code, error_lines, gate_label

    # ------------------------------------------------------------------------------------------

    def get_post_run_fields_after_launch_fail(
        self, bx_job, bx_task, arrived_future_exception
    ):
        """
        Get task post-run fields when the launch itself failed.
        """

        exit_code = -3

        error_lines = explain_cause_chain_error_lines(
            arrived_future_exception, f"launching {callsign(bx_task)}"
        )

        gate_label = "failure"

        logger.error(
            f"failure when {callsign(self)} was launching {callsign(bx_task)}",
            exc_info=arrived_future_exception,
        )

        return exit_code, error_lines, gate_label

    # ------------------------------------------------------------------------------------------
    #
    async def _catalog_artefacts(self, bx_job, bx_task):
        """
        Catalog any artefacts left by the task.
        """

        # Don't catalog artefacts if there is no current catalog in play.
        if not bx_catalogs_has_default():
            return

        # Runtime directory where the isolated task wrote its files.
        runtime_directory = bx_task.get_directory()

        # Name of the file containing the list of proposed artefacts.
        artefacts_filename = f"{runtime_directory}/.bxflow/artefacts.txt"

        if os.path.exists(artefacts_filename):
            with open(artefacts_filename, "r") as stream:
                # Loop through the artefacts listed in the file.
                logger.debug(f"[TSKART] processing artefacts {artefacts_filename}")
                for line in stream.readlines():
                    artefact_filename = line.strip()
                    if os.path.exists(artefact_filename):
                        logger.debug(
                            f"[TSKART] cataloging artefact... {artefact_filename}"
                        )

                        try:
                            # This might raise NotFound if for some reason the catalog server doesn't know the job.
                            await bx_catalogs_get_default().attach_workflow_run_file(
                                bx_job.uuid(), bx_task.uuid(), artefact_filename
                            )

                        # TODO: Pass workflow run file NotFound back through http response to bx_catalogs.
                        # except NotFound as exception:
                        except Exception as exception:
                            logger.warning(
                                callsign(
                                    self,
                                    explain(
                                        exception,
                                        "[TSKART] attaching workflow run file",
                                    ),
                                ),
                                exc_info=exception,
                            )
                    else:
                        logger.debug(
                            f"[TSKART] not finding artefact file... {artefact_filename}"
                        )
        else:
            logger.debug(f"[TSKART] task did not write {artefacts_filename}")
