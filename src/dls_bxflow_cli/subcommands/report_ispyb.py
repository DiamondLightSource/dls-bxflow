# Use standard logging in this module.
import logging
import os
import warnings

from prettytable import PrettyTable

# Base class for femtocheck subcommands.
from dls_bxflow_cli.subcommands.base import Base

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import ispyb
    import ispyb.model.__future__


logger = logging.getLogger()


# --------------------------------------------------------------
class ReportIspyb(Base):
    """
    Check a previously recorded file.
    """

    def __init__(self, args):
        super().__init__(args)

        self.__visit_concat = (
            "CONCAT(Proposal.proposalCode,"
            " Proposal.proposalNumber,"
            " '-',"
            " BLSession.visit_number)"
        )

        if self._args.exclude is None:
            self.__and_exclude = ""
        else:
            exclude = self._args.exclude.split(",")
            for index, not_beamline in enumerate(exclude):
                exclude[index] = f"'{not_beamline}'"
            exclude = ", ".join(exclude)
            self.__and_exclude = f" AND `BLSession`.`beamLineName` NOT IN ({exclude})"

        if self._args.include is None:
            self.__and_include = ""
        else:
            include = self._args.include.split(",")
            for index, not_beamline in enumerate(include):
                include[index] = f"'{not_beamline}'"
            include = ", ".join(include)
            self.__and_include = f" AND `BLSession`.`beamLineName` IN ({include})"

        if self._args.recipe_like is None:
            self.__and_recipe_like = ""
        else:
            self.__and_recipe_like = f" AND recipe LIKE ('{self._args.recipe_like}')"

    # ----------------------------------------------------------
    def run(self):
        """"""

        ispyb.model.__future__.enable(os.environ["ISPYB_CREDENTIALS"])

        if self._args.report == "report1":
            self._run_report1()

        if self._args.report == "report2":
            self._run_report2()

    # ----------------------------------------------------------
    def _run_report1(self):
        """"""

        fields = [
            ("", self.__visit_concat, "visit", "l"),
            ("BLSession", "beamLineName", "beamline", "l"),
            ("", "SUM(data_collection_group.count)", "scan count", "r"),
            ("", "MAX(data_collection_group.latest)", "latest scan", "l"),
        ]

        data_collection_group_sql = (
            "SELECT sessionID, COUNT(*) AS count, MAX(startTime) AS latest"
            " FROM DataCollectionGroup"
            " GROUP BY sessionID"
        )

        sql = (
            f"SELECT {', '.join(self.format_sql_fields(fields))}"
            " FROM BLSession"
            " LEFT JOIN Proposal ON Proposal.proposalId = BLSession.proposalId"
            f" LEFT JOIN ({data_collection_group_sql}) AS data_collection_group"
            " ON BLSession.sessionId = data_collection_group.sessionId"
            " WHERE data_collection_group.count IS NOT NULL"
            " AND Proposal.proposalCode != 'mx'"
            f"{self.__and_exclude}"
            f"{self.__and_include}"
            " GROUP BY visit, beamline"
            " ORDER BY `latest scan` DESC"
            f" LIMIT {self._args.limit}"
        )

        self.query_and_print("scan count by beamline proposal", fields, sql)

    # ----------------------------------------------------------
    def _run_report2(self):
        """"""

        fields = [
            ("", self.__visit_concat, "visit", "l"),
            ("BLSession", "beamLineName", "beamline", "l"),
            ("", "recipe", "recipe", "l"),
            ("", "SUM(data_collection_group.count)", "scan count", "r"),
            ("", "MAX(data_collection_group.latest)", "latest scan", "l"),
        ]

        processing_job_sql = (
            "SELECT dataCollectionId, CONCAT(recipe, ' -- ', displayName) AS recipe"
            " FROM ProcessingJob"
            f" WHERE ProcessingJob.recordTimestamp > '{self._args.since}'"
        )

        data_collection_sql = (
            "SELECT DataCollectionGroupId, recipe, COUNT(*) AS count, MAX(startTime) AS latest"
            " FROM DataCollection"
            f" JOIN ({processing_job_sql}) AS processing_job"
            " ON processing_job.dataCollectionId = DataCollection.dataCollectionId"
            f" WHERE DataCollection.startTime > '{self._args.since}'"
            " GROUP BY DataCollectionGroupId, recipe"
        )

        data_collection_group_sql = (
            "SELECT sessionId, recipe, SUM(count) as count, MAX(latest) as latest"
            " FROM DataCollectionGroup"
            f" JOIN ({data_collection_sql}) AS data_collection"
            " ON data_collection.dataCollectionGroupId = DataCollectionGroup.dataCollectionGroupId"
            " GROUP BY sessionID, recipe"
        )

        sql = (
            f"SELECT {', '.join(self.format_sql_fields(fields))}"
            " FROM BLSession"
            " LEFT JOIN Proposal ON Proposal.proposalId = BLSession.proposalId"
            f" LEFT JOIN ({data_collection_group_sql}) AS data_collection_group"
            " ON BLSession.sessionId = data_collection_group.sessionId"
            " WHERE data_collection_group.count IS NOT NULL"
            " AND Proposal.proposalCode != 'mx'"
            f"{self.__and_exclude}"
            f"{self.__and_include}"
            f"{self.__and_recipe_like}"
            " GROUP BY visit, beamline"
            f" ORDER BY {self._args.order}"
            f" LIMIT {self._args.limit}"
        )

        self.query_and_print("scan count by beamline proposal", fields, sql)

    # ----------------------------------------------------------------------------
    def format_sql_fields(self, fields):
        sql_fields = []
        for field in fields:
            if field[0] is not None and field[0] != "":
                sql_fields.append("`%s`.`%s` AS `%s`" % (field[0], field[1], field[2]))
            else:
                sql_fields.append("%s AS `%s`" % (field[1], field[2]))

        return sql_fields

    # ----------------------------------------------------------------------------
    def query_and_print(self, title, fields, sql):

        # Get the records.
        with ispyb.model.__future__._db_cc() as cursor:
            print(f"{sql}")
            cursor.run(sql)
            records = cursor.fetchall()

        # Start making the ascii table.
        pretty_table = PrettyTable()

        # Make the list of columns in the table.
        pretty_fields = []
        for field in fields:
            pretty_fields.append(field[2])
        pretty_table.field_names = pretty_fields

        # Alignment of the columns.
        for field in fields:
            pretty_table.align[field[2]] = field[3]

        for record in records:
            row = []
            for field in fields:
                value = record[field[2]]
                row.append(value)
            pretty_table.add_row(row)

        pretty_table.title = title

        logger.info("\n" + str(pretty_table))

    # ----------------------------------------------------------
    def add_arguments(parser):

        parser.add_argument(
            help="report name",
            type=str,
            metavar="report",
            dest="report",
        )

        parser.add_argument(
            "--since",
            help="earliest date",
            type=str,
            metavar="date",
            dest="since",
            default="2022-01-01",
        )

        parser.add_argument(
            "--exclude",
            help="beamlines to ignore",
            type=str,
            metavar="command-separated-list",
            dest="exclude",
            default=None,
        )

        parser.add_argument(
            "--include",
            help="beamlines to include",
            type=str,
            metavar="command-separated-list",
            dest="include",
            default=None,
        )

        parser.add_argument(
            "--recipe_like",
            help="recipe name",
            type=str,
            metavar="sql-like-expression",
            dest="recipe_like",
            default=None,
        )

        parser.add_argument(
            "--limit",
            help="maximum records",
            type=int,
            metavar="number",
            dest="limit",
            default=10,
        )

        parser.add_argument(
            "--order",
            help="order by",
            type=str,
            metavar="sql-orderby",
            dest="order",
            default="`latest scan` DESC",
        )
        return parser
