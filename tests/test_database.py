import asyncio
import logging
import multiprocessing

import pytest

# Library for accumulating tagged performance results.
from dls_bxflow_lib.bx_databases.bx_databases import BxDatabases
from dls_bxflow_lib.bx_launchers.states import States

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestDatabaseSqlite:
    def test(self, constants, logging_setup, output_directory):
        """
        Tests the sqlite implementation of BxDatabase.
        """

        database_specification = {
            "type": "dls_bxflow_lib.bx_databases.aiosqlite",
            "filename": "%s/bx_scheduler.sqlite" % (output_directory),
        }

        # Test direct SQL access to the database.
        DatabaseTesterSql().main(
            constants,
            database_specification,
            output_directory,
        )


# ----------------------------------------------------------------------------------------
class _BaseTester:
    """
    Provide asyncio loop and error checking over *Tester classes.
    """

    def main(self, constants, specification, output_directory):
        """
        This is the main program which calls the test using asyncio.
        """

        multiprocessing.current_process().name = "main"

        failure_message = None
        try:
            # Run main test in asyncio event loop.
            asyncio.run(
                self._main_coroutine(constants, specification, output_directory)
            )

        except Exception as exception:
            logger.exception(
                "unexpected exception in the test method", exc_info=exception
            )
            failure_message = str(exception)

        if failure_message is not None:
            pytest.fail(failure_message)


# ----------------------------------------------------------------------------------------
class DatabaseTesterSql(_BaseTester):
    """
    Test direct SQL access to the database.
    """

    async def _main_coroutine(
        self, constants, database_specification, output_directory
    ):
        """ """

        databases = BxDatabases()
        database = databases.build_object(database_specification)

        # Connect to database.
        await database.connect()

        # Write one record.
        await database.bx_launchers_table.insert(
            [{"type": "x", "state": States.IDLE, "uuid": "uuid1"}]
        )
        all_sql = "SELECT * FROM bx_launchers ORDER BY created_on ASC"
        records = await database.query(all_sql)
        assert len(records) == 1, "first %s count" % (all_sql)

        # Write two more records.
        await database.bx_launchers_table.insert(
            [
                {"type": "y", "state": States.IDLE, "uuid": "uuid2"},
                {"type": "z", "state": States.IDLE, "uuid": "uuid3"},
            ]
        )
        records = await database.query(all_sql)
        assert len(records) == 3, "second %s count" % (all_sql)

        # Update one record to BUSY.
        await database.bx_launchers_table.update(
            {"state": States.BUSY}, "uuid = 'uuid3'"
        )
        busy_sql = (
            "SELECT * FROM bx_launchers WHERE state = '%s' ORDER BY created_on ASC"
            % (States.BUSY)
        )
        records = await database.query(busy_sql)
        assert len(records) == 1, "%s count" % (busy_sql)

        # Update two records to DEAD.
        await database.bx_launchers_table.update(
            {"state": States.UNRESPONSIVE}, "uuid IN ('uuid1', 'uuid2')"
        )
        dead_sql = (
            "SELECT * FROM bx_launchers WHERE state = '%s' ORDER BY created_on ASC"
            % (States.UNRESPONSIVE)
        )
        records = await database.query(dead_sql)
        assert len(records) == 2, "%s count" % (dead_sql)

        # Write one variable record.
        await database.bx_variables_table.insert(
            [{"type": "x", "state": States.IDLE, "uuid": "uuid4", "value": True}]
        )

        # Connect from the database... necessary to allow asyncio loop to exit.
        await database.disconnect()
