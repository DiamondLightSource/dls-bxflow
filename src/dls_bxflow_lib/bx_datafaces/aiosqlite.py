import logging
import shutil

# Base class for generic things.
from dls_bxflow_api.thing import Thing

# Database manager.
from dls_bxflow_lib.bx_databases.bx_databases import BxDatabases

# News producer.
from dls_bxflow_lib.bx_datafaces.news_producer import NewsProducer

# BxJobs manager.
from dls_bxflow_lib.bx_jobs.states import States as BxJobStates
from dls_bxflow_lib.bx_jobs.states import States as BxJobkStates

# News events.
from dls_bxflow_run.bx_gates.states import States as BxGateStates

# BxTasks manager.
from dls_bxflow_run.bx_tasks.states import States as BxTaskStates

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_datafaces.aiosqlite"


class Aiosqlite(Thing):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        Thing.__init__(self, thing_type, specification)

        self.__database = None

        self.__news_producer = NewsProducer(self)

    # ----------------------------------------------------------------------------------------
    async def start(self):
        # Connect to the database to create the schemas if they don't exist already.
        await self.establish_database_connection()

    # ----------------------------------------------------------------------------------------
    async def disconnect(self):
        if self.__news_producer is not None:
            await self.__news_producer.disconnect()
            self.__news_producer = None

        if self.__database is not None:
            await self.__database.disconnect()
            self.__database = None

    # ----------------------------------------------------------------------------------------
    async def establish_database_connection(self):

        if self.__database is None:
            self.__database = BxDatabases().build_object(
                self.specification()["database"]
            )
            await self.__database.connect()

    # ----------------------------------------------------------------------------------------
    async def reinstance(self):
        """"""
        if self.__database is None:
            return

        self.__database = self.__database.reinstance()

    # ----------------------------------------------------------------------------------------
    async def set_bx_gates(self, bx_gate_dicts):
        """"""
        await self.establish_database_connection()

        await self.__database.bx_gates_table.insert(bx_gate_dicts)

    # ----------------------------------------------------------------------------------------
    async def set_bx_launcher(self, record):
        """"""
        await self.establish_database_connection()

        await self.__database.bx_launchers_table.insert([record])

        await self.__news_producer.update_bx_launcher(record)

    # ----------------------------------------------------------------------------------------
    async def set_bx_tasks(self, bx_task_dicts):
        """"""
        await self.establish_database_connection()

        await self.__database.bx_tasks_table.insert(bx_task_dicts)

    # ----------------------------------------------------------------------------------------
    async def set_bx_task_dependency_bx_gates(self, bx_task_dependency_bx_gate_dicts):
        """"""
        await self.establish_database_connection()

        await self.__database.bx_task_dependency_bx_gates_table.insert(
            bx_task_dependency_bx_gate_dicts
        )

    # ----------------------------------------------------------------------------------------
    async def set_bx_job_blocked_by_bx_gates(self, bx_job_blocked_by_bx_gate_dicts):
        """"""
        await self.establish_database_connection()

        await self.__database.bx_job_blocked_by_bx_gates_table.insert(
            bx_job_blocked_by_bx_gate_dicts
        )

    # ----------------------------------------------------------------------------------------
    async def set_bx_jobs(self, bx_job_dicts):
        """ """
        await self.establish_database_connection()

        await self.__database.bx_jobs_table.insert(bx_job_dicts)

    # ----------------------------------------------------------------------------------------
    async def set_bx_variables(self, bx_variables_dict):
        """ """
        await self.establish_database_connection()

        await self.__database.bx_variables_table.insert(bx_variables_dict)

    # ----------------------------------------------------------------------------------------
    async def set_bx_workflows(self, bx_workflows_dicts):
        """ """
        await self.establish_database_connection()

        await self.__database.bx_workflows_table.insert(bx_workflows_dicts)

    # ----------------------------------------------------------------------------------------
    async def set_bx_news(self, bx_news_dict):
        """ """
        await self.establish_database_connection()

        await self.__database.bx_news_table.insert(bx_news_dict)

    # ----------------------------------------------------------------------------------------
    async def get_bx_launcher(self, bx_launcher_uuid):
        """
        Get single bx_launcher from its uuid.
        Returns database record format.
        """
        await self.establish_database_connection()

        # Get all the bx_launchers for the bx_job.
        sql = "SELECT * FROM bx_launchers WHERE uuid = '%s'" % (bx_launcher_uuid)

        records = await self.__database.query(sql)

        if len(records) == 0:
            return None

        return records[0]

    # ----------------------------------------------------------------------------------------
    async def get_bx_launchers(self, states=None):
        """
        Get bx_launchers with given states.
        Returns database records format.
        """
        await self.establish_database_connection()

        sql = "SELECT * FROM bx_launchers"

        if states is not None:
            quoted_states = ", ".join([f"'{state}'" for state in states])
            sql = f"{sql} WHERE state IN ({quoted_states})"

        records = await self.__database.query(sql)

        return records

    # ----------------------------------------------------------------------------------------
    async def get_bx_tasks_launched_by(self, bx_launcher_uuid, states=None):
        """
        Get bx_tasks with given states which where launched by given launcher.
        Returns database records format.
        """
        await self.establish_database_connection()

        sql = f"SELECT * FROM bx_tasks WHERE bx_launcher_uuid = '{bx_launcher_uuid}'"

        if states is not None:
            if not isinstance(states, list):
                states = [states]
            quoted_states = ", ".join([f"'{state}'" for state in states])
            sql = f"{sql} AND state IN ({quoted_states})"

        records = await self.__database.query(sql)

        return records

    # ----------------------------------------------------------------------------------------
    async def get_bx_task(self, bx_task_uuid):
        """
        Get single bx_task from its uuid.
        Returns database record format.
        """
        await self.establish_database_connection()

        # Get all the bx_tasks for the bx_job.
        sql = "SELECT * FROM bx_tasks WHERE uuid = '%s'" % (bx_task_uuid)

        records = await self.__database.query(sql)

        if len(records) == 0:
            return None

        return records[0]

    # ----------------------------------------------------------------------------------------
    async def get_bx_tasks(self, bx_job_uuid):
        """
        Get all bx_tasks related to current bx_job.
        Returns database records format.
        """
        await self.establish_database_connection()

        # Get all the bx_tasks for the bx_job.
        sql = "SELECT * FROM bx_tasks WHERE bx_job_uuid = '%s' ORDER BY created_on" % (
            bx_job_uuid
        )

        records = await self.__database.query(sql)

        return records

    # ----------------------------------------------------------------------------------------
    async def get_bx_job(self, bx_job_uuid, why=None):
        """
        Get single bx_job from its uuid.
        Returns database record format.
        """
        await self.establish_database_connection()

        # Get all the bx_jobs for the bx_job.
        sql = "SELECT * FROM bx_jobs WHERE uuid = '%s'" % (bx_job_uuid)

        records = await self.__database.query(sql, why=why)

        if len(records) == 0:
            return None

        return records[0]

    # ----------------------------------------------------------------------------------------
    async def get_bx_jobs(
        self, states=None, labels=None, order_by=None, limit=None, why=None
    ):
        """
        Get bx_jobs with given states.
        Returns database records format.
        """
        await self.establish_database_connection()

        sql = (
            "SELECT bx_jobs.*,"
            " bx_workflows.filename_classname as workflow_filename_classname"
            " FROM bx_jobs"
            " JOIN bx_workflows"
            " ON bx_jobs.bx_workflow_uuid = bx_workflows.uuid"
        )

        if states is not None:
            quoted_states = ", ".join([f"'{state}'" for state in states])
            sql = f"{sql} WHERE state IN ({quoted_states})"

        # TODO: Properly escape literal label field in sql query.
        if labels is not None:
            quoted_labels = ", ".join([f"'{label}'" for label in labels])
            sql = f"{sql} WHERE label IN ({quoted_labels})"

        if order_by is not None:
            sql += f" ORDER BY {order_by}"

        if limit is not None:
            sql += f" LIMIT {limit}"

        records = await self.__database.query(sql, why=why)

        return records

    # ----------------------------------------------------------------------------------------
    async def get_controlled_bx_gates(self, bx_job_uuid):
        """
        Get all bx_gates controlled by bx_tasks related to current bx_job.
        Returns database records format.
        """
        await self.establish_database_connection()

        # Get all the bx_tasks for the bx_job.
        bx_tasks_sql = "SELECT uuid FROM bx_tasks WHERE bx_job_uuid = '%s'" % (
            bx_job_uuid
        )

        # Get all the bx_gates controlled by the bx_tasks in the bx_job.
        sql = (
            "SELECT bx_gates.*"
            " FROM bx_gates"
            f" LEFT OUTER JOIN ({bx_tasks_sql}) AS bx_tasks"
            " WHERE bx_gates.bx_task_uuid = bx_tasks.uuid"
        )

        records = await self.__database.query(sql)

        return records

    # ----------------------------------------------------------------------------------------
    async def get_dependency_bx_gates(self, bx_job_uuid):
        """
        Get all bx_gates dependency by bx_tasks related to current bx_job.
        Returns database records format.
        """
        await self.establish_database_connection()

        # Get all the bx_tasks for the bx_job.
        bx_tasks_sql = "SELECT uuid FROM bx_tasks WHERE bx_job_uuid = '%s'" % (
            bx_job_uuid
        )

        # Get all the dependency bx_gates needed by the bx_tasks.
        sql = (
            "SELECT bx_task_dependency_bx_gates.*"
            " FROM bx_task_dependency_bx_gates"
            f" LEFT OUTER JOIN ({bx_tasks_sql}) AS bx_tasks"
            " WHERE bx_tasks.uuid = bx_task_dependency_bx_gates.lhs"
        )

        records = await self.__database.query(sql)

        return records

    # ----------------------------------------------------------------------------------------
    async def get_blocked_by_bx_gates(self, bx_job_uuid):
        """
        Get all bx_gates which block the job when high.
        Returns database records format.
        """
        await self.establish_database_connection()

        # Get all the gates which block the job.
        sql = (
            "SELECT rhs"
            " FROM bx_job_blocked_by_bx_gates"
            f" WHERE lhs = '{bx_job_uuid}'"
        )

        records = await self.__database.query(sql)

        return records

    # ----------------------------------------------------------------------------------------
    async def get_bx_variables(self, bx_job_uuid, why=None):
        """
        Get all bx_variables related to current bx_job.
        Returns database records format.
        """
        await self.establish_database_connection()

        sql = f"SELECT * FROM bx_variables WHERE bx_job_uuid = '{bx_job_uuid}'"

        records = await self.__database.query(sql, why=why)

        return records

    # ----------------------------------------------------------------------------------------
    async def get_bx_workflows(self, bx_job_uuids, why=None):
        """
        Get all workflows related to given bx_jobs.
        Returns database records format.
        """
        await self.establish_database_connection()

        quoted_fields = []
        for bx_job_uuid in bx_job_uuids:
            quoted_fields.append(f"'{bx_job_uuid}'")

        quoted_fields = ", ".join(quoted_fields)

        sql = f"SELECT * FROM bx_workflows WHERE bx_job_uuid IN ({quoted_fields})"

        records = await self.__database.query(sql, why=why)

        return records

    # ----------------------------------------------------------------------------------------
    async def open_bx_gate(
        self,
        bx_task_uuid,
        controlled_bx_gate_name,
        why=None,
    ):
        """"""
        await self.establish_database_connection()

        row = {"state": BxGateStates.OPEN}
        count = await self.__database.bx_gates_table.update(
            row,
            "bx_task_uuid = '%s' AND label='%s'"
            % (bx_task_uuid, controlled_bx_gate_name),
        )

        if count > 0:
            await self.__news_producer.open_bx_gate(
                bx_task_uuid, controlled_bx_gate_name, why=why
            )

        return {"count": count}

    # ----------------------------------------------------------------------------------------
    async def close_bx_gate(self, bx_task_uuid, controlled_bx_gate_name, why=None):
        """"""
        await self.establish_database_connection()

        row = {"state": BxGateStates.CLOSED}
        count = await self.__database.bx_gates_table.update(
            row,
            "bx_task_uuid = '%s' AND label='%s'"
            % (bx_task_uuid, controlled_bx_gate_name),
        )

        if count > 0:
            await self.__news_producer.close_bx_gate(
                bx_task_uuid, controlled_bx_gate_name, why=why
            )

        return {"count": count}

    # ----------------------------------------------------------------------------------------
    async def update_bx_launcher(self, row, launcher_callsign=None):
        """"""
        await self.establish_database_connection()

        count = await self.__database.bx_launchers_table.update(
            row, "uuid = '%s'" % (row["uuid"])
        )

        if count > 0:
            await self.__news_producer.update_bx_launcher(
                row, launcher_callsign=launcher_callsign
            )

        return {"count": count}

    # ----------------------------------------------------------------------------------------
    async def update_bx_task(self, row, why=None):
        """"""
        await self.establish_database_connection()

        count = await self.__database.bx_tasks_table.update(
            row,
            "uuid = '%s'" % (row["uuid"]),
            why=why,
        )

        if count > 0:
            await self.__news_producer.update_bx_task(row, why=why)

        return {"count": count}

    # ----------------------------------------------------------------------------------------
    async def update_bx_job(self, row, why=None):
        """"""
        await self.establish_database_connection()

        count = await self.__database.bx_jobs_table.update(
            row,
            "uuid = '%s'" % (row["uuid"]),
            why=why,
        )

        if count > 0:
            await self.__news_producer.update_bx_job(row, why=why)

        return {"count": count}

    # ----------------------------------------------------------------------------------------
    async def cancel_bx_job(self, bx_job_uuid):
        """"""
        return await self.update_bx_job(
            {"uuid": bx_job_uuid, "state": BxJobStates.CANCELLED}
        )

    # ----------------------------------------------------------------------------------------
    async def delete_bx_job(self, bx_job_uuid):
        """ """

        bx_job_record = await self.get_bx_job(bx_job_uuid)
        if bx_job_record is None:
            logger.warning(f"cannot find bx_job {bx_job_uuid} to delete it")
            return

        logger.debug(
            f"deleting bx_job {bx_job_uuid} with label \"{bx_job_record['label']}\""
        )

        why = "delete job"

        # Delete relations records.
        await self.__database.execute(
            f"DELETE FROM bx_task_dependency_bx_gates WHERE lhs IN (SELECT uuid FROM bx_tasks WHERE bx_job_uuid IN ('{bx_job_uuid}'))",
            why=why,
        )
        await self.__database.execute(
            f"DELETE FROM bx_job_blocked_by_bx_gates WHERE lhs IN ('{bx_job_uuid}')",
            why=why,
        )

        # Delete direct foreign keyed records.
        await self.__database.execute(
            f"DELETE FROM bx_news WHERE bx_job_uuid IN ('{bx_job_uuid}')",
            why=why,
        )
        await self.__database.execute(
            f"DELETE FROM bx_variables WHERE bx_job_uuid IN ('{bx_job_uuid}')",
            why=why,
        )
        await self.__database.execute(
            f"DELETE FROM bx_gates WHERE bx_job_uuid IN ('{bx_job_uuid}')",
            why=why,
        )
        await self.__database.execute(
            f"DELETE FROM bx_tasks WHERE bx_job_uuid IN ('{bx_job_uuid}')",
            why=why,
        )
        await self.__database.execute(
            f"DELETE FROM bx_workflows WHERE bx_job_uuid IN ('{bx_job_uuid}')",
            why=why,
        )

        # Delete job folder.
        directory = bx_job_record["directory"]
        logger.debug(f"removing job directory {directory}")
        shutil.rmtree(directory, ignore_errors=True)

        # Delete job record itself.
        await self.__database.execute(
            f"DELETE FROM bx_jobs WHERE uuid IN ('{bx_job_uuid}')"
        )

        await self.__news_producer.delete_bx_job(bx_job_record)

    # ----------------------------------------------------------------------------------------
    async def delete_bx_launcher(self, bx_launcher_uuid):
        """ """

        why = "delete launcher"

        # Delete launcher record itself.
        await self.__database.execute(
            f"DELETE FROM bx_launchers WHERE uuid IN ('{bx_launcher_uuid}')",
            why=why,
        )

    # ----------------------------------------------------------------------------------------
    async def unblock_bx_job(self, bx_job_uuid):
        """"""

        # Find the gates blocking the job.
        sql = (
            "SELECT bx_gates.*, bx_tasks.directory AS bx_task_directory"
            " FROM bx_job_blocked_by_bx_gates"
            " JOIN bx_gates ON bx_gates.uuid = bx_job_blocked_by_bx_gates.rhs"
            " JOIN bx_tasks ON bx_tasks.uuid = bx_gates.bx_task_uuid"
            f" WHERE lhs = '{bx_job_uuid}'"
            f" AND bx_gates.state = '{BxGateStates.OPEN}'"
        )

        records = await self.__database.query(sql)

        if len(records) == 0:
            logger.warning("[UNBLOCKJOB] job has no blocking gates")

        for record in records:
            bx_task_uuid = record["bx_task_uuid"]
            bx_task_directory = record["bx_task_directory"]

            if bx_task_directory is None:
                logger.warning(
                    f"[UNBLOCKJOB] task {bx_task_uuid} directory is {bx_task_directory}"
                )
            elif "/processing/" not in bx_task_directory:
                logger.warning(
                    f"[UNBLOCKJOB] task {bx_task_uuid} directory {bx_task_directory} does not contain /processing/"
                )
            elif len(bx_task_directory.split("/processing/")[1]) < 10:
                logger.warning(
                    f"[UNBLOCKJOB] task {bx_task_uuid} directory {bx_task_directory} too short after /processing/"
                )
            else:
                try:
                    shutil.rmtree(bx_task_directory, ignore_errors=False, onerror=None)
                    logger.debug(
                        f"[UNBLOCKJOB] removed task {bx_task_uuid} directory {bx_task_directory}"
                    )
                except Exception as exception:
                    logger.warning(
                        f"[UNBLOCKJOB] task {bx_task_uuid} directory {bx_task_directory} could not be removed because {exception}"
                    )

            # Close the gates.
            await self.close_bx_gate(
                record["bx_task_uuid"],
                record["label"],
                why="[UNBLOCKJOB] unblocked",
            )

            # Allow tasks controlling those gates to be scheduled again.
            row = {
                "uuid": record["bx_task_uuid"],
                "state": BxTaskStates.PREPARED,
                "error_lines": "",
            }
            await self.update_bx_task(
                row,
                why="[UNBLOCKJOB] unblocked",
            )

        # Make job ready again.
        row = {"uuid": bx_job_uuid, "state": BxJobkStates.READY}
        await self.update_bx_job(
            row,
            why="[UNBLOCKJOB] unblocked",
        )

    # ----------------------------------------------------------------------------------------
    async def enable_bx_job(self, uuid):
        """"""
        await self.establish_database_connection()

        row = {"uuid": uuid, "state": BxJobStates.READY}

        result = await self.update_bx_job(row)

        return result

    # ----------------------------------------------------------------------------------------
    async def update_bx_job_if_blocked_by_bx_gates(self, bx_job_uuid):

        sql = (
            "SELECT DISTINCT lhs, bx_gates.label"
            " FROM bx_job_blocked_by_bx_gates"
            " JOIN bx_gates ON bx_gates.uuid = bx_job_blocked_by_bx_gates.rhs"
            f" WHERE lhs = '{bx_job_uuid}'"
            f" AND bx_gates.state = '{BxGateStates.OPEN}'"
        )

        records = await self.__database.query(sql)

        if len(records) > 0:
            state = BxJobStates.SUCCEEDED
            for record in records:
                if record["label"] == "failure":
                    state = BxJobStates.FAILED
            row = {"uuid": bx_job_uuid, "state": state}
            result = await self.update_bx_job(row)
            count = result["count"]
        else:
            count = 0

        return {"count": count}

    # ----------------------------------------------------------------------------------------
    async def get_bx_jobs_bx_tasks_bx_gates(self, bx_job_uuid=None, why=None):
        """"""
        await self.establish_database_connection()

        bx_task_controlled_bx_gate_label_sql = (
            "SELECT bx_tasks.uuid AS bx_task_uuid,"
            " (bx_tasks.label || '.' || bx_gates.label) AS label, bx_gates.state"
            " FROM bx_tasks LEFT OUTER JOIN bx_gates ON bx_tasks.uuid = bx_gates.bx_task_uuid"
        )

        bx_task_dependency_bx_gate_label_sql = (
            "SELECT dependency_bx_tasks.uuid AS bx_task_uuid,"
            " (controlled_bx_tasks.label || '.' || bx_gates.label) AS label, bx_gates.state"
            " FROM bx_tasks AS dependency_bx_tasks JOIN bx_task_dependency_bx_gates"
            " ON dependency_bx_tasks.uuid = bx_task_dependency_bx_gates.lhs"
            " JOIN bx_gates ON bx_gates.uuid = bx_task_dependency_bx_gates.rhs"
            " JOIN bx_tasks AS controlled_bx_tasks"
            " ON controlled_bx_tasks.uuid = bx_gates.bx_task_uuid"
        )

        # records = await self.__database.query(bx_task_dependency_bx_gate_label_sql)
        # logger.info(
        #     "bx_task_dependency_bx_gate_label_sql\n%s" % (json.dumps(records, indent=4))
        # )

        sql = (
            "SELECT bx_jobs.label AS bx_job_label,"
            " bx_jobs.uuid AS bx_job_uuid,"
            " bx_tasks.label AS bx_task_label,"
            " bx_tasks.directory AS bx_task_directory,"
            " bx_tasks.uuid AS bx_task_uuid,"
            " bx_tasks.state AS bx_task_state,"
            # TODO: Improved storage efficiency by separate lookup for task error lines in get_bx_jobs_bx_tasks_bx_gates.
            " bx_tasks.exit_code AS bx_task_exit_code,"
            " bx_tasks.error_lines AS bx_task_error_lines,"
            " controlled_bx_gates.label AS controlled_bx_gate_label,"
            " controlled_bx_gates.state AS controlled_bx_gate_state,"
            " dependency_bx_gates.label AS dependency_bx_gate_label,"
            " dependency_bx_gates.state AS dependency_bx_gate_state"
            " FROM bx_jobs"
            " LEFT OUTER JOIN bx_tasks ON bx_tasks.bx_job_uuid = bx_jobs.uuid"
            f" LEFT OUTER JOIN ({bx_task_controlled_bx_gate_label_sql}) AS controlled_bx_gates"
            " ON controlled_bx_gates.bx_task_uuid = bx_tasks.uuid"
            f" LEFT OUTER JOIN ({bx_task_dependency_bx_gate_label_sql}) AS dependency_bx_gates"
            " ON dependency_bx_gates.bx_task_uuid = bx_tasks.uuid"
        )

        if bx_job_uuid is not None:
            sql += f" WHERE bx_jobs.uuid IN ('{bx_job_uuid}')"

        sql += " ORDER BY bx_jobs.created_on ASC, bx_tasks.created_on ASC"

        records = await self.__database.query(sql, why=why)
        return records

    # ----------------------------------------------------------------------------------------
    async def get_bx_news(self, bx_job_uuid=None, why=None):
        """"""
        await self.establish_database_connection()

        # job_sql = (
        #     "SELECT bx_news.*, bx_jobs.label AS `job`"
        #     " FROM bx_news"
        #     " JOIN bx_tabx_jobssks ON bx_jobs.uuid = bx_news.bx_job_uuid"
        # )

        # task_sql = (
        #     "SELECT bx_news.*, bx_tasks.label AS `task`"
        #     " FROM bx_news"
        #     " JOIN bx_tasks ON bx_tasks.uuid = bx_news.bx_task_uuid"
        # )

        # union_sql = (
        #     "SELECT DISTINCT bx_news.uuid"
        #     f" FROM ({job_sql}) AS job_news"
        #     f" UNION ALL ({task_sql}) AS task_new"
        # )

        news_sql = "SELECT bx_news.* FROM bx_news"

        if bx_job_uuid is not None:
            news_sql += f" WHERE bx_news.bx_job_uuid = '{bx_job_uuid}'"
            news_sql += f" OR bx_news.bx_task_uuid IN (SELECT uuid FROM bx_tasks WHERE bx_job_uuid = '{bx_job_uuid}')"

        if bx_job_uuid is None:
            news_sql += " LIMIT 10"

        task_sql = (
            "SELECT bx_tasks.uuid,"
            " bx_tasks.label AS task_label,"
            " bx_jobs.label AS job_label"
            " FROM bx_tasks"
            " JOIN bx_jobs ON bx_jobs.uuid = bx_tasks.bx_job_uuid"
        )

        sql = (
            "SELECT news.*,"
            " COALESCE(bx_jobs.label, tasks.job_label, '') AS `job`,"
            " COALESCE(tasks.task_label, '') AS `task`"
            f" FROM ({news_sql}) AS news"
            f" LEFT OUTER JOIN bx_jobs ON bx_jobs.uuid = news.bx_job_uuid"
            f" LEFT OUTER JOIN ({task_sql}) AS tasks ON tasks.uuid = news.bx_task_uuid"
            " ORDER BY news.created_on DESC"
        )

        records = await self.__database.query(sql, why=why)

        return records

    # ----------------------------------------------------------------------------------------
    async def query(self, sql, subs=None, why=None):
        """"""
        await self.establish_database_connection()

        return await self.__database.query(sql, subs=subs, why=why)

    # ----------------------------------------------------------------------------------------
    async def report_health(self):
        """"""

        report = {}

        report["alive"] = True

        return report
