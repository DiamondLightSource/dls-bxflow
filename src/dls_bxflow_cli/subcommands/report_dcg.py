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
class ReportDcg(Base):
    """
    Report on an ispyb DataCollectionGroup.
    """

    def __init__(self, args):
        super().__init__(args)

        self.__visit_concat = (
            "CONCAT(Proposal.proposalCode,"
            " Proposal.proposalNumber,"
            " '-',"
            " BLSession.visit_number)"
        )

    # ----------------------------------------------------------
    def run(self):
        """"""

        ispyb.model.__future__.enable(os.environ["ISPYB_CREDENTIALS"])

        # ----------------------------------------------------------------------------
        dcg_fields = [
            ("", self.__visit_concat, "visit", "l"),
            ("BLSession", "beamLineName", "beamline", "l"),
            ("data_collection_group", "dataCollectionGroupId", "dcg", "l"),
        ]

        dcg_sql = (
            "SELECT *"
            " FROM DataCollectionGroup"
            f" WHERE DataCollectionGroup.dataCollectionGroupId = {self._args.dcg}"
        )

        sql = (
            f"SELECT {', '.join(self.format_sql_fields(dcg_fields))}"
            " FROM BLSession"
            " JOIN Proposal ON Proposal.proposalId = BLSession.proposalId"
            f" JOIN ({dcg_sql}) AS data_collection_group"
            " ON BLSession.sessionId = data_collection_group.sessionId"
        )

        self.query_and_print("DataCollectionGroup", dcg_fields, sql)

        # ----------------------------------------------------------------------------

        processing_job_sql = (
            "SELECT dataCollectionId, processingJobId, CONCAT(recipe, ' -- ', displayName) AS recipe"
            " FROM ProcessingJob"
        )

        dc_fields = [
            ("DataCollection", "dataCollectionId", "dc id", "l"),
            ("processing_job", "processingJobId", "job id", "l"),
            ("processing_job", "recipe", "recipe", "l"),
            ("DataCollection", "xtalSnapshotFullPath1", "snapshot", "l"),
            ("DataCollection", "datFullPath", "dat file", "l"),
        ]

        dc_sql = (
            f"SELECT {', '.join(self.format_sql_fields(dc_fields))}"
            " FROM DataCollection"
            f" JOIN ({processing_job_sql}) AS processing_job"
            " ON processing_job.dataCollectionId = DataCollection.dataCollectionId"
            f" WHERE DataCollection.dataCollectionGroupId = {self._args.dcg}"
        )

        self.query_and_print("DataCollection", dc_fields, dc_sql)

        # ----------------------------------------------------------------------------

        autoproc_program_sql = (
            "SELECT processingJobId, autoProcProgramId, processingCommandLine, processingPrograms"
            " FROM AutoProcProgram"
        )

        autoproc_program_fields = [
            ("AutoProcProgram", "processingJobId", "job id", "l"),
            ("AutoProcProgram", "autoProcProgramId", "auto proc id", "l"),
            ("AutoProcProgram", "processingCommandLine", "command line", "l"),
            ("AutoProcProgram", "processingPrograms", "programs", "l"),
        ]

        autoproc_program_sql = (
            f"SELECT {', '.join(self.format_sql_fields(autoproc_program_fields))}"
            " FROM AutoProcProgram"
            " JOIN ProcessingJob ON ProcessingJob.processingJobId = AutoProcProgram.processingJobId"
            " JOIN DataCollection ON DataCollection.dataCollectionId = ProcessingJob.dataCollectionId"
            f" WHERE DataCollection.dataCollectionGroupId = {self._args.dcg}"
        )

        self.query_and_print(
            "AutoProcProgram", autoproc_program_fields, autoproc_program_sql
        )

        # ----------------------------------------------------------------------------

        autoproc_program_attachment_sql = (
            "SELECT autoProcProgramId, fileType, fileName, filePath"
            " FROM AutoProcProgramAttachment"
        )

        autoproc_program_attachment_fields = [
            ("AutoProcProgramAttachment", "autoProcProgramId", "auto proc id", "l"),
            ("AutoProcProgramAttachment", "fileType", "file type", "l"),
            ("AutoProcProgramAttachment", "fileName", "file name", "l"),
            ("AutoProcProgramAttachment", "filePath", "file path", "l"),
        ]

        autoproc_program_attachment_sql = (
            f"SELECT {', '.join(self.format_sql_fields(autoproc_program_attachment_fields))}"
            " FROM AutoProcProgramAttachment"
            " JOIN AutoProcProgram ON AutoProcProgram.autoProcProgramId = AutoProcProgramAttachment.autoProcProgramId"
            " JOIN ProcessingJob ON ProcessingJob.processingJobId = AutoProcProgram.processingJobId"
            " JOIN DataCollection ON DataCollection.dataCollectionId = ProcessingJob.dataCollectionId"
            f" WHERE DataCollection.dataCollectionGroupId = {self._args.dcg}"
        )

        self.query_and_print(
            "AutoProcProgramAttachment",
            autoproc_program_attachment_fields,
            autoproc_program_attachment_sql,
        )

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
            help="data collection group id",
            type=str,
            metavar="number",
            dest="dcg",
        )

        return parser
