import json
import logging
import os
import warnings
from typing import Any, List, NewType, Optional, Tuple

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.describe import describe
from dls_utilpack.isodatetime import isodatetime
from dls_utilpack.require import require
from dls_utilpack.sanitize import sanitize

# Database field names.
from dls_bxflow_api.bx_databases.constants import BxJobFieldnames

# Exception we might raise.
from dls_bxflow_api.exceptions import NotFound

# Base class for generic things.
from dls_bxflow_api.thing import Thing
from dls_bxflow_lib.typing import BxSpecificationType

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import ispyb.model.__future__
    from ispyb import NoResult
    from ispyb.connector.mysqlsp.main import ISPyBMySQLSPConnector


logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_catalogs.ispyb"

SqlFieldsType = NewType("SqlFieldsType", List[Tuple[Any, Any, Any, Any]])


class WorkflowRun:
    def __init__(self):
        self.session_id = None
        self.data_collection_group_id = None
        self.data_collection_id = None
        self.processing_job_id = None
        self.autoproc_program_id = None
        self.message = None


# Co-opt a field the ispby schema to stash the bx_job_uuid.
BX_JOB_UUID_FIELD_NAME = "imageDirectory"
BX_JOB_UUID_FIELD_PARAM = "imgdir"

# Co-opt a field the ispby schema to stash the data label.
DATA_LABEL_FIELD_NAME = "imagePrefix"
DATA_LABEL_FIELD_PARAM = "imgprefix"


class Ispyb(Thing):
    """
    Class which implements BxCatalogApi interface using ispyb.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification: Optional[BxSpecificationType] = None):
        Thing.__init__(self, thing_type, specification)

        self.__ispyb: Optional[ISPyBMySQLSPConnector] = None

    # ----------------------------------------------------------------------------------------
    async def start(self) -> None:
        """
        Called to start operations.
        """

        await self.__establish_database_connection()

    # ----------------------------------------------------------------------------------------
    async def disconnect(self) -> None:
        """
        Called to stop operations.
        """

        if self.__ispyb is not None:
            self.__ispyb.disconnect()
            self.__ispyb = None

    # ----------------------------------------------------------------------------------------
    async def __establish_database_connection(self) -> None:
        """
        Connect to the ispyb database.
        """

        if self.__ispyb is None:
            try:
                ispyb.model.__future__.enable(os.environ["ISPYB_CREDENTIALS"])
                self.__ispyb = ispyb.open()
                self.__ispyb_core = self.__ispyb.core
                self.__ispyb_mxacquisition = self.__ispyb.mx_acquisition
                self.__ispyb_mxprocessing = self.__ispyb.mx_processing
            except Exception:
                await self.disconnect()
                raise

    # ----------------------------------------------------------------------------------------
    async def create_workflow_run(self, bx_job_record: dict) -> None:
        """
        Create a new workflow run in the catalog.
        """

        await self.__establish_database_connection()

        workflow_run = WorkflowRun()

        # Identifies the bx_job_record.
        bx_job_uuid = require("bx_job_record", bx_job_record, "uuid")

        # Visit comes from bx_job_record.
        visit = require("bx_job_record", bx_job_record, "visit")

        # Job record cannot provide a visit?
        if visit is None:
            logger.debug(
                "[WORKRUN] visit is none in bx_job_record\n%s"
                % (json.dumps(bx_job_record))
            )
            return

        try:
            workflow_run.session_id = self.__ispyb_core.retrieve_visit_id(visit)
        except NoResult:
            raise RuntimeError(f"visit {visit} does not exist in the ispyb database")

        logger.debug(describe("session_id", workflow_run.session_id))
        start_time = isodatetime()

        # TODO: Sanitize input to sql query when looking for data label in ispyb.
        data_label = require("job record", bx_job_record, BxJobFieldnames.DATA_LABEL)
        sanitized_data_label = sanitize(data_label)

        # Co-opt a field the ispby schema to stash the bx_job_uuid.
        bx_job_uuid_field_value = bx_job_uuid

        # Co-opt a field the ispby schema to stash the data label.
        data_label_field_value = f"bxflow/{sanitized_data_label}"

        sql = (
            f"SELECT DataCollectionGroup.dataCollectionGroupId\n"
            " FROM DataCollectionGroup\n"
            " JOIN DataCollection ON DataCollection.dataCollectionGroupId = DataCollectionGroup.dataCollectionGroupId"
            f" WHERE DataCollectionGroup.sessionId = {workflow_run.session_id}"
            f" AND DataCollection.{DATA_LABEL_FIELD_NAME} = '{data_label_field_value}'"
            " ORDER BY DataCollectionGroup.dataCollectionGroupId DESC"
        )

        records = self.__query(
            f"check for existing data collection for data label {data_label}", sql
        )

        if len(records) > 0:
            workflow_run.data_collection_group_id = records[0]["dataCollectionGroupId"]
            logger.debug(
                f"got existing ispyb dataCollectionGroupId {workflow_run.data_collection_group_id}"
            )
        else:
            params = self.__ispyb_mxacquisition.get_data_collection_group_params()
            params["parentid"] = workflow_run.session_id
            params["start_time"] = start_time
            params["comments"] = "Data label: {data_label}"
            workflow_run.data_collection_group_id = (
                self.__ispyb_mxacquisition.insert_data_collection_group(
                    list(params.values())
                )
            )
            logger.debug(
                f"inserted new ispyb dataCollectionGroupId {workflow_run.data_collection_group_id}"
            )

        params = self.__ispyb_mxacquisition.get_data_collection_params()
        params["parentid"] = workflow_run.data_collection_group_id
        params["visitid"] = workflow_run.session_id
        params["start_time"] = start_time
        # TODO: Put lock around computing datacollection_number when creating new catalog workflow run.
        params["datacollection_number"] = len(records) + 1
        params["run_status"] = "DataCollection Successful"
        params[BX_JOB_UUID_FIELD_PARAM] = bx_job_uuid_field_value
        params[DATA_LABEL_FIELD_PARAM] = data_label_field_value
        params["n_images"] = 0
        # params["img_container_sub_path"] = "datacollection/1/"
        workflow_run.data_collection_id = (
            self.__ispyb_mxacquisition.insert_data_collection(list(params.values()))
        )
        logger.debug(
            f"inserted ispyb dataCollectionId {workflow_run.data_collection_id}"
        )

        # ProcessingJob = a workflow class
        params = self.__ispyb_mxprocessing.get_job_params()
        params["datacollectionid"] = workflow_run.data_collection_id
        params["display_name"] = bx_job_record["label"]
        params["comments"] = "no comments"
        params["recipe"] = "no recipe"
        params["automatic"] = True
        workflow_run.processing_job_id = self.__ispyb_mxprocessing.upsert_job(
            list(params.values())
        )
        logger.debug(f"inserted ispyb processingJobId {workflow_run.processing_job_id}")

        # AutoProcProgram = a run of a workflow class
        workflow_run.autoproc_program_id = self.__ispyb_mxprocessing.upsert_program_ex(
            job_id=workflow_run.processing_job_id,
            # Value for "name" is saved to "processingPrograms"
            name=bx_job_record["label"],
            # Value for "command" is saved to processingCommandLine
            command="",
            environment="",
            message="bxflow workflow run registered",
            time_start=start_time,
            time_update=start_time,
        )
        logger.debug(
            f"inserted ispyb AutoprocProgramId {workflow_run.autoproc_program_id}"
        )

        logger.debug(f"[WORKRUN] adding workflow_run for bx_job_uuid {bx_job_uuid}")

    # ----------------------------------------------------------------------------------------
    async def update_workflow_run_message(self, bx_job_uuid: str, message: str) -> None:
        """
        Update the status message on the workflow run.
        """

        # Get the ispyb AutoprocProgram record for the workflow in question.
        workflow_run = await self.query_workflow_run_details(bx_job_uuid)
        workflow_run.message = message
        date_time = isodatetime()

        logger.debug(
            f"upsert_program_ex {workflow_run.autoproc_program_id} message={message}"
        )
        self.__ispyb_mxprocessing.upsert_program_ex(
            program_id=workflow_run.autoproc_program_id,
            message=workflow_run.message,
            time_update=date_time,
        )

    # ----------------------------------------------------------------------------------------
    async def finish_workflow_run(self, bx_job_record: dict) -> None:
        """
        Seal the workflow run.
        """

        bx_job_uuid = bx_job_record["uuid"]

        # Get the ispyb AutoprocProgram record for the workflow in question.
        workflow_run = await self.query_workflow_run_details(bx_job_uuid)
        date_time = isodatetime()

        # :param status: An integer describing the processing status. 1 means
        #                success, 0 means failure. If left at None then the
        #                status is left undefined or unchanged. The underlying
        #                stored procedure does not allow any more changes to the
        #                record once the status is set.
        # Synchweb shows the swirling icon and shows no files while status is None.
        # TODO: When updating ispyb autoproc progarm status, determine if job is blocked due to error.
        status = 1
        message = "bx_job finished sucessfully"
        workflow_run.message = message
        logger.debug(
            f"upsert_program_ex {workflow_run.autoproc_program_id}, status={status}"
        )
        self.__ispyb_mxprocessing.upsert_program_ex(
            program_id=workflow_run.autoproc_program_id,
            message=workflow_run.message,
            status=1,
            time_update=date_time,
        )

    # ----------------------------------------------------------------------------------------
    async def attach_workflow_run_file(
        self,
        bx_job_uuid: str,
        bx_task_uuid: str,
        filename: str,
    ) -> None:
        """
        Attach a single file to a workflow run.
        """

        # Get the ispyb AutoprocProgram record for the workflow in question.
        workflow_run = await self.query_workflow_run_details(bx_job_uuid)

        params = self.__ispyb_mxprocessing.get_program_attachment_params()
        params["parentid"] = workflow_run.autoproc_program_id
        params["file_name"] = os.path.basename(filename)
        params["file_path"] = os.path.dirname(filename)
        # TODO: Add ispby file attachment type and rank.
        params["file_type"] = "Result"
        params["importance_rank"] = 1
        result = self.__ispyb_mxprocessing.upsert_program_attachment(
            list(params.values())
        )

        logger.debug(
            f"[WORKRUN] adding program attachment {result} for bx_job_uuid {bx_job_uuid}"
        )

    # ----------------------------------------------------------------------------------------
    async def query_workflow_run_details(self, bx_job_uuid: str):
        """
        Get information about the workflow run.
        """

        sql = (
            "SELECT DataCollection.dataCollectionId,\n"
            "   ProcessingJob.processingJobId,\n"
            "   AutoProcProgram.autoProcProgramId,\n"
            "   AutoProcProgram.processingMessage\n"
            " FROM DataCollection\n"
            " JOIN ProcessingJob ON ProcessingJob.dataCollectionId = DataCollection.dataCollectionId\n"
            " JOIN AutoProcProgram ON AutoProcProgram.processingJobId = ProcessingJob.processingJobId\n"
            f" WHERE DataCollection.{BX_JOB_UUID_FIELD_NAME} = '{bx_job_uuid}'"
        )

        records = self.__query("get workflow run details for bx job", sql)

        if len(records) == 0:
            raise NotFound(
                f"[WORKRUN] no workflow_run in {callsign(self)} database for bx_job_uuid {bx_job_uuid}"
            )

        workflow_run = WorkflowRun()

        workflow_run.session_id = None
        workflow_run.data_collection_group_id = None
        workflow_run.data_collection_id = records[0]["dataCollectionId"]
        workflow_run.processing_job_id = records[0]["processingJobId"]
        workflow_run.autoproc_program_id = records[0]["autoProcProgramId"]
        workflow_run.message = records[0]["processingMessage"]

        return workflow_run

    # ----------------------------------------------------------------------------------------
    async def query_workflow_run(self, bx_job_uuid: str):
        """
        Get information about the workflow run.
        """

        workflow_run = await self.query_workflow_run_details(bx_job_uuid)

        return {"message": workflow_run.message}

    # ----------------------------------------------------------------------------------------
    async def query_workflow_run_files(self, bx_job_uuid: str) -> List[str]:
        """Get the list of files attached to the workflow run.

        Args:
            bx_job_uuid (str): uuid of the bxflow job

        Returns:
            List[str]: filenames found by the query
        """

        filename_concat = (
            "CONCAT(AutoProcProgramAttachment.filePath,"
            " '/',"
            " AutoProcProgramAttachment.fileName)"
        )

        fields: SqlFieldsType = SqlFieldsType(
            [
                ("", filename_concat, "filename", "l"),
            ]
        )

        # Get the ispyb AutoprocProgram record for the workflow in question.
        workflow_run = await self.query_workflow_run_details(bx_job_uuid)

        sql = (
            f"SELECT {', '.join(self.__format_sql_fields_for_select(fields))}\n"
            " FROM AutoProcProgramAttachment\n"
            f" WHERE autoProcProgramId = {workflow_run.autoproc_program_id}"
        )

        records = self.__query("query_workflow_run_files", sql)

        filenames = []
        for record in records:
            filenames.append(record["filename"])

        return filenames

    # ----------------------------------------------------------------------------
    def __format_sql_fields_for_select(self, fields: SqlFieldsType) -> List[str]:
        """Format fields in list into a list of strings suitable for use in a SELECT.

        Args:
            fields (SqlFieldsType): list of field specifications

        Returns:
            List[str]: list of sql-selectable fields
        """

        sql_fields = []
        for field in fields:
            if field[0] is not None and field[0] != "":
                sql_fields.append("`%s`.`%s` AS `%s`" % (field[0], field[1], field[2]))
            else:
                sql_fields.append("%s AS `%s`" % (field[1], field[2]))

        return sql_fields

    # ----------------------------------------------------------------------------
    def __query(self, title: str, sql: str) -> List[dict]:
        """Issue sql select statement and returns result records.

        Args:
            title (str): for identifying the query in logging
            sql (str): sql statement

        Returns:
            List[dict]: records containing the fields from the sql SELECT statement
        """

        with ispyb.model.__future__._db_cc() as cursor:
            cursor.run(sql)
            records = cursor.fetchall()
            logger.debug(f"{title} {len(records)} records from\n{sql}")

        return records
