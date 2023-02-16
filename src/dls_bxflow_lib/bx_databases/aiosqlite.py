# This class produces log entries.
import logging
import os
from collections import OrderedDict
from datetime import datetime

import aiosqlite

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.explain import explain

# ----------------------------------------------------------------------------------------
from dls_bxflow_api.bx_databases.constants import (
    BxGateFieldnames,
    BxJobFieldnames,
    BxLauncherFieldnames,
    BxTaskFieldnames,
    BxVariableFieldnames,
    BxWorkflowFieldnames,
    NewsFieldnames,
    RelationFieldnames,
    RevisionFieldnames,
)

logger = logging.getLogger(__name__)

LATEST_REVISION = 4


# ----------------------------------------------------------------------------------------
class Aiosqlite:
    """
    Class with coroutines for creating and querying a sqlite database.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification):
        """
        Construct object.  Do not connect to database.
        """
        self.__filename = specification["filename"]
        self.bx_news_table = None
        self.bx_gates_table = None
        self.bx_launchers_table = None
        self.bx_tasks_table = None
        self.bx_task_dependency_bx_gates_table = None
        self.bx_job_blocked_by_bx_gates_table = None
        self.bx_jobs_table = None
        self.bx_variables_table = None
        self.bx_workflows_table = None
        self.revision_table = None

        # Don't normally want to see all the debug for aiosqlite internals.
        level = specification.get("log_level", "INFO")
        logging.getLogger("aiosqlite").setLevel(level)

    # ----------------------------------------------------------------------------------------
    async def connect(self):
        """
        Connect to database at filename given in constructor.
        """

        _should_create_schemas = False

        # File doesn't exist yet?
        if not os.path.isfile(self.__filename):
            # Create directory for the file.
            await self._create_directory()
            # After connection, we must create the schemas.
            _should_create_schemas = True

        self._connection = await aiosqlite.connect(self.__filename)
        self._connection.row_factory = aiosqlite.Row

        # Schemas in our database.
        self.bx_news_table = BxNewsTable(self)
        self.bx_gates_table = BxGatesTable(self)
        self.bx_launchers_table = BxLaunchersTable(self)
        self.bx_variables_table = BxVariablesTable(self)
        self.bx_workflows_table = BxWorkflowsTable(self)
        self.bx_task_dependency_bx_gates_table = BxTaskDependencyBxGatesTable(self)
        self.bx_job_blocked_by_bx_gates_table = BxBlockedByBxGatesTable(self)
        self.bx_tasks_table = BxTasksTable(self)
        self.bx_jobs_table = BxJobsTable(self)
        self.revision_table = RevisionTable(self)

        await self.bx_news_table.setup()
        await self.bx_gates_table.setup()
        await self.bx_launchers_table.setup()
        await self.bx_tasks_table.setup()
        await self.bx_task_dependency_bx_gates_table.setup()
        await self.bx_job_blocked_by_bx_gates_table.setup()
        await self.bx_jobs_table.setup()
        await self.bx_variables_table.setup()
        await self.bx_workflows_table.setup()
        await self.revision_table.setup()

        if _should_create_schemas:
            await self._create_schemas()
            await self.revision_table.insert([{"number": LATEST_REVISION}])
            # TODO: Set permission on sqlite file from configuration.
            os.chmod(self.__filename, 0o666)
        else:
            try:
                records = await self.query(
                    "SELECT number FROM revision", why="get database revision"
                )
                old_revision = records[0]["number"]
            except Exception:
                await self.revision_table.create()
                old_revision = 0
                await self.revision_table.insert([{"number": 0}])

            if old_revision < LATEST_REVISION:
                for revision in range(old_revision, LATEST_REVISION):
                    await self.apply_revision(revision + 1)
                await self.revision_table.update(
                    {"number": LATEST_REVISION},
                    "1 = 1",
                    why="update database revision",
                )

        # Emit the name of the database file for positive confirmation on console.
        logger.info(
            f"{callsign(self)} database file is {self.__filename} revision {LATEST_REVISION}"
        )

    # ----------------------------------------------------------------------------------------
    async def apply_revision(self, revision):
        if revision == 1:
            await self.execute(
                f"ALTER TABLE bx_jobs ADD COLUMN {BxJobFieldnames.COMMENT} TEXT",
                why=f"revision 1: add job {BxJobFieldnames.COMMENT} column",
            )
        if revision == 2:
            await self.execute(
                f"ALTER TABLE bx_jobs ADD COLUMN {BxJobFieldnames.RATING} INTEGER",
                why=f"revision 2: add job {BxJobFieldnames.RATING} column",
            )
            await self.execute(
                "CREATE INDEX %s_%s ON %s(%s)"
                % ("bx_jobs", BxJobFieldnames.RATING, "bx_jobs", BxJobFieldnames.RATING)
            )
        if revision == 3:
            await self.execute(
                "DROP TABLE bx_cookies",
                why="revision 3: remove cookies table",
            )
        if revision == 4:
            await self.execute(
                f"ALTER TABLE bx_launchers ADD COLUMN {BxLauncherFieldnames.REMEX_CLUSTER} TEXT",
                why=f"revision 4: add launcher {BxLauncherFieldnames.REMEX_CLUSTER} column",
            )
            await self.execute(
                "CREATE INDEX %s_%s ON %s(%s)"
                % (
                    "bx_launchers",
                    BxLauncherFieldnames.REMEX_CLUSTER,
                    "bx_launchers",
                    BxLauncherFieldnames.REMEX_CLUSTER,
                )
            )

    # ----------------------------------------------------------------------------------------
    async def disconnect(self):

        if self._connection is not None:
            await self._connection.close()

    # ----------------------------------------------------------------------------------------
    def reinstance(self):
        """
        Clone database instance.
        """

        return Aiosqlite(self.__filename)

    # ----------------------------------------------------------------------------------------
    async def _create_directory(self):

        directory, filename = os.path.split(self.__filename)

        if not os.path.exists(directory):
            # Make sure that parent directories which get created will have public permission.
            umask = os.umask(0)
            os.umask(umask & ~0o0777)
            os.makedirs(directory)
            os.umask(umask)

    # ----------------------------------------------------------------------------------------
    async def _create_schemas(self):

        await self.bx_news_table.create()
        await self.bx_gates_table.create()
        await self.bx_launchers_table.create()
        await self.bx_tasks_table.create()
        await self.bx_task_dependency_bx_gates_table.create()
        await self.bx_job_blocked_by_bx_gates_table.create()
        await self.bx_jobs_table.create()
        await self.bx_variables_table.create()
        await self.bx_workflows_table.create()
        await self.revision_table.create()

    # ----------------------------------------------------------------------------------------
    async def execute(self, sql, why=None):

        cursor = None
        try:
            cursor = await self._connection.cursor()
            await cursor.execute(sql)
            if why is None:
                logger.debug(f"{cursor.rowcount} records affected by {sql}")
            else:
                logger.debug(f"{cursor.rowcount} records affected by {why}: {sql}")
        except aiosqlite.OperationalError:
            if why is None:
                raise RuntimeError(f"failed to execute {sql}")
            else:
                raise RuntimeError(f"failed to execute {why}: {sql}")
        finally:
            if cursor is not None:
                await cursor.close()

        # Commmit the auto-transaction.
        await self._connection.commit()

    # ----------------------------------------------------------------------------------------
    async def query(self, sql, subs=None, why=None):

        if subs is None:
            subs = {}

        cursor = None
        try:
            cursor = await self._connection.cursor()
            await cursor.execute(sql, subs)
            rows = await cursor.fetchall()
            cols = []
            for col in cursor.description:
                cols.append(col[0])

            if why is None:
                logger.debug("%d records from: %s" % (len(rows), sql))
            else:
                logger.debug("%d records from %s: %s" % (len(rows), why, sql))
            records = []
            for row in rows:
                record = OrderedDict()
                for index, col in enumerate(cols):
                    record[col] = row[index]
                records.append(record)
            return records
        except aiosqlite.OperationalError as exception:
            if why is None:
                raise RuntimeError(explain(exception, f"executing {sql}"))
            else:
                raise RuntimeError(explain(exception, f"executing {why}: {sql}"))
        finally:
            if cursor is not None:
                await cursor.close()


# ----------------------------------------------------------------------------------------
class BaseTable:
    # ----------------------------------------------------------------------------------------
    def __init__(self, database, table_name):
        self._database = database
        self._table_name = table_name
        self._connection = self._database._connection

        self._fields = OrderedDict()

        # All tables have a unique uuid field.
        self._fields["uuid"] = {
            "type": "TEXT PRIMARY KEY",
            "index": True,
        }

    # ----------------------------------------------------------------------------------------
    async def setup(self):
        """
        Perform database interaction necessary to access the table.
        """
        pass

    # ----------------------------------------------------------------------------------------
    async def create(self):
        """
        Wipe and re-create the table in the database.
        """

        await self._database.execute("DROP TABLE IF EXISTS %s" % (self._table_name))

        fields_sql = []
        indices_sql = []

        for field_name in self._fields:
            field = self._fields[field_name]
            fields_sql.append("%s %s" % (field_name, field["type"]))
            if field["index"]:
                indices_sql.append(
                    "CREATE INDEX %s_%s ON %s(%s)"
                    % (self._table_name, field_name, self._table_name, field_name)
                )

        await self._database.execute(
            "CREATE TABLE %s(%s)" % (self._table_name, ", ".join(fields_sql))
        )

        for sql in indices_sql:
            await self._database.execute(sql)

        await self._connection.commit()

    # ----------------------------------------------------------------------------------------
    async def insert(self, rows):
        """
        Insert one or more rows.
        """

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        values_rows = []

        for row in rows:
            values_row = []
            for field in self._fields:
                if field == "created_on":
                    values_row.append(row.get(field, now))
                else:
                    values_row.append(row.get(field, None))
            values_rows.append(values_row)

        qmarks = []
        for field in self._fields:
            qmarks.append("?")

        command = "INSERT INTO %s VALUES (%s)" % (
            self._table_name,
            ", ".join(qmarks),
        )

        logger.debug("%s %s" % (command, values_rows))
        cursor = None
        try:
            cursor = await self._connection.cursor()
            await cursor.executemany(command, values_rows)
        finally:
            if cursor is not None:
                await cursor.close()

        await self._connection.commit()

    # ----------------------------------------------------------------------------------------
    async def update(self, row, where, why=None):
        """
        Update specified fields to all rows matching selection.
        """

        values_row = []
        qmarks = []

        for field in self._fields:
            if field == "uuid":
                continue
            if field not in row:
                continue
            if field == "state":
                qmarks.append("%s = '%s'" % (field, row[field]))
            else:
                qmarks.append("%s = ?" % (field))
                values_row.append(row[field])

        command = "UPDATE %s SET %s WHERE %s" % (
            self._table_name,
            ", ".join(qmarks),
            where,
        )

        cursor = None
        try:
            cursor = await self._connection.cursor()
            await cursor.executemany(command, [values_row])
            rowcount = cursor.rowcount
        finally:
            if cursor is not None:
                await cursor.close()

        await self._connection.commit()

        if why is None:
            logger.debug("%d rows from %s" % (rowcount, command))
        else:
            logger.debug("%d rows from %s: %s" % (rowcount, why, command))

        return rowcount


# ----------------------------------------------------------------------------------------
class BxNewsTable(BaseTable):
    # ----------------------------------------------------------------------------------------
    def __init__(self, database):
        BaseTable.__init__(self, database, "bx_news")

        self._fields[NewsFieldnames.UUID] = {"type": "TEXT", "index": True}
        self._fields[NewsFieldnames.CREATED_ON] = {"type": "TEXT", "index": True}
        self._fields[NewsFieldnames.TOPIC] = {"type": "TEXT", "index": True}
        self._fields[NewsFieldnames.BX_JOB_UUID] = {"type": "TEXT", "index": True}
        self._fields[NewsFieldnames.BX_TASK_UUID] = {"type": "TEXT", "index": True}
        self._fields[NewsFieldnames.HEADLINE] = {"type": "TEXT", "index": False}
        self._fields[NewsFieldnames.DETAILS] = {"type": "TEXT", "index": False}


# ----------------------------------------------------------------------------------------
class BxJobsTable(BaseTable):
    # ----------------------------------------------------------------------------------------
    def __init__(self, database):
        BaseTable.__init__(self, database, "bx_jobs")

        self._fields[BxJobFieldnames.UUID] = {"type": "TEXT", "index": True}
        self._fields[BxJobFieldnames.CREATED_ON] = {"type": "TEXT", "index": True}
        self._fields[BxJobFieldnames.TYPE] = {"type": "TEXT", "index": True}
        self._fields[BxJobFieldnames.STATE] = {"type": "TEXT", "index": True}
        self._fields[BxJobFieldnames.SPECIFICATION] = {"type": "TEXT", "index": False}
        self._fields[BxJobFieldnames.LABEL] = {"type": "TEXT", "index": True}
        self._fields[BxJobFieldnames.DATA_LABEL] = {"type": "TEXT", "index": True}
        self._fields[BxJobFieldnames.DIRECTORY] = {"type": "TEXT", "index": True}
        self._fields[BxJobFieldnames.BEAMLINE] = {"type": "TEXT", "index": True}
        self._fields[BxJobFieldnames.VISIT] = {"type": "TEXT", "index": True}
        self._fields[BxJobFieldnames.BX_CATALOG_UUID] = {"type": "TEXT", "index": True}
        self._fields[BxJobFieldnames.BX_WORKFLOW_UUID] = {"type": "TEXT", "index": True}
        self._fields[BxJobFieldnames.COMMENT] = {"type": "TEXT", "index": False}
        self._fields[BxJobFieldnames.RATING] = {"type": "INTEGER", "index": True}


# ----------------------------------------------------------------------------------------
class BxGatesTable(BaseTable):
    # ----------------------------------------------------------------------------------------
    def __init__(self, database):
        BaseTable.__init__(self, database, "bx_gates")

        self._fields[BxGateFieldnames.UUID] = {"type": "TEXT", "index": True}
        self._fields[BxGateFieldnames.CREATED_ON] = {"type": "TEXT", "index": True}
        self._fields[BxGateFieldnames.TYPE] = {"type": "TEXT", "index": True}
        self._fields[BxGateFieldnames.BX_JOB_UUID] = {"type": "TEXT", "index": True}
        self._fields[BxGateFieldnames.BX_TASK_UUID] = {"type": "TEXT", "index": True}
        self._fields[BxGateFieldnames.STATE] = {"type": "TEXT", "index": True}
        self._fields[BxGateFieldnames.SPECIFICATION] = {"type": "TEXT", "index": False}
        self._fields[BxGateFieldnames.LABEL] = {"type": "TEXT", "index": False}


# ----------------------------------------------------------------------------------------
class BxLaunchersTable(BaseTable):
    # ----------------------------------------------------------------------------------------
    def __init__(self, database):
        BaseTable.__init__(self, database, "bx_launchers")

        self._fields[BxLauncherFieldnames.UUID] = {"type": "TEXT", "index": True}
        self._fields[BxLauncherFieldnames.CREATED_ON] = {"type": "TEXT", "index": True}
        self._fields[BxLauncherFieldnames.TYPE] = {"type": "TEXT", "index": True}
        self._fields[BxLauncherFieldnames.STATE] = {"type": "TEXT", "index": True}
        self._fields[BxLauncherFieldnames.SPECIFICATION] = {
            "type": "TEXT",
            "index": False,
        }
        self._fields[BxLauncherFieldnames.SUBMIT_COUNT] = {
            "type": "INTEGER",
            "index": False,
        }
        self._fields[BxLauncherFieldnames.REMEX_CLUSTER] = {
            "type": "TEXT",
            "index": True,
        }


# ----------------------------------------------------------------------------------------
class BxTasksTable(BaseTable):
    # ----------------------------------------------------------------------------------------
    def __init__(self, database):
        BaseTable.__init__(self, database, "bx_tasks")

        self._fields[BxTaskFieldnames.UUID] = {"type": "TEXT", "index": True}
        self._fields[BxTaskFieldnames.CREATED_ON] = {"type": "TEXT", "index": True}
        self._fields[BxTaskFieldnames.TYPE] = {"type": "TEXT", "index": True}
        self._fields[BxTaskFieldnames.BX_JOB_UUID] = {"type": "TEXT", "index": True}
        self._fields[BxTaskFieldnames.BX_LAUNCHER_UUID] = {
            "type": "TEXT",
            "index": True,
        }
        self._fields[BxTaskFieldnames.launch_info] = {
            "type": "TEXT",
            "index": False,
        }
        self._fields[BxTaskFieldnames.STATE] = {"type": "TEXT", "index": True}
        self._fields[BxTaskFieldnames.SPECIFICATION] = {"type": "TEXT", "index": False}
        self._fields[BxTaskFieldnames.LABEL] = {"type": "TEXT", "index": True}
        self._fields[BxTaskFieldnames.DIRECTORY] = {"type": "TEXT", "index": True}
        self._fields[BxTaskFieldnames.EXIT_CODE] = {"type": "INTEGER", "index": True}
        self._fields[BxTaskFieldnames.ERROR_LINES] = {"type": "TEXT", "index": False}


# ----------------------------------------------------------------------------------------
class BxTaskDependencyBxGatesTable(BaseTable):
    # ----------------------------------------------------------------------------------------
    def __init__(self, database):
        BaseTable.__init__(self, database, "bx_task_dependency_bx_gates")

        self._fields.pop("uuid")
        self._fields[RelationFieldnames.LHS] = {"type": "TEXT", "index": True}
        self._fields[RelationFieldnames.RHS] = {"type": "TEXT", "index": True}


# ----------------------------------------------------------------------------------------
class BxBlockedByBxGatesTable(BaseTable):
    # ----------------------------------------------------------------------------------------
    def __init__(self, database):
        BaseTable.__init__(self, database, "bx_job_blocked_by_bx_gates")

        self._fields.pop("uuid")
        self._fields[RelationFieldnames.LHS] = {"type": "TEXT", "index": True}
        self._fields[RelationFieldnames.RHS] = {"type": "TEXT", "index": True}


# ----------------------------------------------------------------------------------------
class BxVariablesTable(BaseTable):
    # ----------------------------------------------------------------------------------------
    def __init__(self, database):
        BaseTable.__init__(self, database, "bx_variables")

        self._fields[BxVariableFieldnames.UUID] = {"type": "TEXT", "index": True}
        self._fields[BxVariableFieldnames.CREATED_ON] = {"type": "TEXT", "index": True}
        self._fields[BxVariableFieldnames.TYPE] = {"type": "TEXT", "index": True}
        self._fields[BxVariableFieldnames.BX_JOB_UUID] = {"type": "TEXT", "index": True}
        self._fields[BxVariableFieldnames.STATE] = {"type": "TEXT", "index": True}
        self._fields[BxVariableFieldnames.NAME] = {"type": "TEXT", "index": True}
        self._fields[BxVariableFieldnames.VALUE] = {"type": "TEXT", "index": False}


# ----------------------------------------------------------------------------------------
class BxWorkflowsTable(BaseTable):
    # ----------------------------------------------------------------------------------------
    def __init__(self, database):
        BaseTable.__init__(self, database, "bx_workflows")

        self._fields[BxWorkflowFieldnames.UUID] = {"type": "TEXT", "index": True}
        self._fields[BxWorkflowFieldnames.CREATED_ON] = {"type": "TEXT", "index": True}
        self._fields[BxWorkflowFieldnames.BX_JOB_UUID] = {"type": "TEXT", "index": True}
        self._fields[BxWorkflowFieldnames.FILENAME_CLASSNAME] = {
            "type": "TEXT",
            "index": True,
        }
        self._fields[BxWorkflowFieldnames.BX_SETTINGS_JSON] = {
            "type": "TEXT",
            "index": False,
        }


# ----------------------------------------------------------------------------------------
class RevisionTable(BaseTable):
    # ----------------------------------------------------------------------------------------
    def __init__(self, database):
        BaseTable.__init__(self, database, "revision")

        self._fields.pop("uuid")
        self._fields[RevisionFieldnames.CREATED_ON] = {"type": "TEXT", "index": False}
        self._fields[RevisionFieldnames.NUMBER] = {"type": "INTEGER", "index": False}
