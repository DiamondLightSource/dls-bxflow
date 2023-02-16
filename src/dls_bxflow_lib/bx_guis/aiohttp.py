import asyncio
import inspect
import json
import logging
import multiprocessing
import threading

# Utilities.
from dls_utilpack.callsign import callsign, callsign_html
from dls_utilpack.describe import describe
from dls_utilpack.import_class import import_class
from dls_utilpack.require import require

# Database field names.
from dls_bxflow_api.bx_databases.constants import BxJobFieldnames, BxWorkflowFieldnames
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Basic things.
from dls_bxflow_api.thing import Thing

# Base class for an aiohttp server.
from dls_bxflow_lib.base_aiohttp import BaseAiohttp
from dls_bxflow_lib.bx_catalogs.bx_catalogs import bx_catalogs_get_default
from dls_bxflow_lib.bx_collectors.bx_collectors import bx_collectors_get_default

# Object managing bx_composers.
from dls_bxflow_lib.bx_composers.bx_composers import (
    BxComposers,
    bx_composers_get_default,
    bx_composers_has_default,
    bx_composers_set_default,
)

# Object managing bx_configurators.
from dls_bxflow_lib.bx_configurators.bx_configurators import (
    bx_configurators_get_default,
)

# Gaml manager.
from dls_bxflow_lib.bx_gamls.bx_gamls import BxGamls

# BxGui protocolj things (must agree with javascript).
from dls_bxflow_lib.bx_guis.constants import Commands, Cookies, Keywords
from dls_bxflow_lib.bx_launchers.bx_launchers import BxLaunchers

# Logstore object.
from dls_bxflow_lib.bx_logstores.bx_logstores import bx_logstores_get_default

# News object.
from dls_bxflow_lib.bx_news.bx_news import BxNews, bx_news_get_default

# Global managers.
from dls_bxflow_lib.bx_schedulers.bx_schedulers import bx_schedulers_get_default

# Settings manager.
from dls_bxflow_lib.bx_settings.bx_settings import BxSettings

# Workflow finder.
from dls_bxflow_lib.bx_workflows.workflow_finder import WorkflowFinder

# BxLauncher manager.


logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_guis.aiohttp"


# ------------------------------------------------------------------------------------------
class Aiohttp(Thing, BaseAiohttp):
    """
    Object implementing remote procedure calls for bx_gui methods.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        Thing.__init__(self, thing_type, specification)

        BaseAiohttp.__init__(
            self,
            specification["type_specific_tbd"]["aiohttp_specification"],
            calling_file=__file__,
        )

        bx_gaml_specification = {
            "type": "dls_bxflow_lib.bx_gamls.html",
        }

        self.__bx_gaml = BxGamls().build_object(bx_gaml_specification)

        # List of launchers we have connected with.
        self._bx_launchers = BxLaunchers()

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""
        return "%s %s" % ("BxGui.Aiohttp", BaseAiohttp.callsign(self))

    # ----------------------------------------------------------------------------------------
    def activate_process(self):
        """"""

        try:
            multiprocessing.current_process().name = "bx_gui"

            self.activate_process_base()

        except Exception as exception:
            logger.exception(
                f"unable to start {callsign(self)} process", exc_info=exception
            )

    # ----------------------------------------------------------------------------------------
    def activate_thread(self, loop):
        """
        Called from inside a newly created thread.
        """

        try:
            threading.current_thread().name = "bx_gui"

            self.activate_thread_base(loop)

        except Exception as exception:
            logger.exception(
                f"unable to start {callsign(self)} thread", exc_info=exception
            )

    # ----------------------------------------------------------------------------------------
    async def activate_coro(self):
        """"""
        try:
            # No special routes, we will use protocolj dispathcing only
            route_tuples = []

            # Build a bx_news to listen for events.
            # This has to be a new instance because we will run a receive loop on it.
            self.__bx_news = BxNews().build_object(
                bx_news_get_default().specification()
            )

            # Start a news consumer receive loop.
            self.__bx_news_consumer_future = asyncio.create_task(
                self.__bx_news.consume(self.__consume_bx_news)
            )

            # Start the actual coro to listen for incoming http requests.
            await self.activate_coro_base(route_tuples)

            # No default composer is set up yet?
            if not bx_composers_has_default():
                # The bx_composer to use.
                bx_composer_specification = {"type": "dls_bxflow_lib.bx_composers.html"}

                # Set up the default bx_composer.
                bx_composer = BxComposers().build_object(bx_composer_specification)
                bx_composers_set_default(bx_composer)

        except Exception:
            raise RuntimeError(f"unable to start {callsign(self)} server coro")

    # ----------------------------------------------------------------------------------------
    async def direct_shutdown(self):
        """"""

        if self.__bx_news is not None:
            # Disconnect from news server we have been using.
            # This will set the flag to stop the consumer receive loop.
            await self.__bx_news.request_stop()

            # Stop the asyncio task whcih is listening for news.
            if self.__bx_news_consumer_future is not None:
                # logger.info("waiting for consumuer future to stop")
                await self.__bx_news_consumer_future
                self.__bx_news_consumer_future = None

        # Let the base class stop the server looping.
        await self.base_direct_shutdown()

    # ----------------------------------------------------------------------------------------
    async def __consume_bx_news(self, topic, headline, payload):
        """ """

        # TBD: In bx_gui, connect consumed news items to browser client by websockets.

        pass

    # ----------------------------------------------------------------------------------------
    async def dispatch(self, request_dict, opaque):
        """"""

        command = require("request json", request_dict, Keywords.COMMAND)

        if command == Commands.LOAD_TABS:
            return await self._load_tabs(opaque, request_dict)

        elif command == Commands.SELECT_TAB:
            return await self._select_tab(opaque, request_dict)

        elif command == Commands.GET_RECENT_JOBS:
            response = await self._get_recent_jobs(opaque, request_dict)

        elif command == Commands.UPDATE_JOB:
            response = await self._update_job(opaque, request_dict)

        elif command == Commands.GET_JOB_DETAILS:
            response = await self._get_job_details(opaque, request_dict)

        elif command == Commands.GET_JOB_NEWS:
            response = await self._get_job_news(opaque, request_dict)

        elif command == Commands.GET_JOB_VARIABLES:
            response = await self._get_job_variables(opaque, request_dict)

        elif command == Commands.GET_JOB_DATA_GRID:
            response = await self._get_job_data_grid(opaque, request_dict)

        elif command == Commands.GET_SYSTEM_HEALTH:
            response = await self._get_system_health(opaque, request_dict)

        elif command == Commands.SHOW_WORKFLOW_SETTINGS:
            response = await self._show_workflow_settings(opaque, request_dict)

        elif command == Commands.START_WORKFLOW:
            response = await self._start_workflow(opaque, request_dict)

        elif command == Commands.START_WORKFLOW_NOCOOKIE:
            response = await self._start_workflow_nocookie(opaque, request_dict)

        elif command == Commands.GET_WORKFLOW_CONSTRUCTOR_KWARGS:
            response = await self._get_workflow_constructor_kwargs(opaque, request_dict)

        elif command == Commands.CANCEL_JOB:
            bx_job_uuid = require("request json", request_dict, "bx_job_uuid")
            await bx_datafaces_get_default().cancel_bx_job(bx_job_uuid)
            response = await self._get_recent_jobs(opaque, request_dict)

        elif command == Commands.DELETE_JOB:
            bx_job_uuid = require("request json", request_dict, "bx_job_uuid")
            await bx_datafaces_get_default().delete_bx_job(bx_job_uuid)
            response = await self._response_for_cookie_name(
                opaque, command, request_dict
            )

        elif command == Commands.UNBLOCK_JOB:
            bx_job_uuid = require("request json", request_dict, "bx_job_uuid")
            records = await bx_datafaces_get_default().unblock_bx_job(bx_job_uuid)
            response = await self._get_recent_jobs(opaque, request_dict)

        elif command == Commands.GET_RECENT_NEWS:
            records = await bx_datafaces_get_default().get_bx_news(why="[GUIPOLL]")
            html = bx_composers_get_default().compose_bx_news(records)
            response = {"html": html}

        elif command == Commands.QUERY_LOGSTORE:
            return await self._query_logstore(opaque, request_dict)

        else:
            raise RuntimeError("invalid command %s" % (command))

        return response

    # ----------------------------------------------------------------------------------------
    async def _response_for_cookie_name(self, opaque, command, request_dict):

        response = None
        if response is None:
            try:
                opaque.cookies.find(Cookies.RECENT_JOBS_UX, trait_name="cookie_name")
                response = await self._get_recent_jobs(opaque, request_dict)
            except NotFound:
                pass

        if response is None:
            try:
                opaque.cookies.find(Cookies.JOB_DETAILS_UX, trait_name="cookie_name")
                response = await self._get_job_details(opaque, request_dict)

                logger.debug(describe("[DELJOB] response", response))
            except NotFound:
                pass

        if response is None:
            response = f"command {command} completed"

        return response

    # ----------------------------------------------------------------------------------------
    async def _handle_auto_update(self, opaque, request_dict, cookie_name):

        # Remember last posted value for auto_update_enabled.
        auto_update_enabled = request_dict.get("auto_update_enabled")
        # logger.debug(
        #     describe(
        #         f"[AUTOUP] request_dict auto_update_enabled for cookie {cookie_name}",
        #         auto_update_enabled,
        #     )
        # )
        auto_update_enabled = await self.set_or_get_cookie_content(
            opaque,
            cookie_name,
            "auto_update_enabled",
            auto_update_enabled,
            False,
        )
        # logger.debug(
        #     describe(
        #         f"[AUTOUP] request_set_or_get_cookie_content auto_update_enabled",
        #         auto_update_enabled,
        #     )
        # )

        return auto_update_enabled

    # ----------------------------------------------------------------------------------------
    async def _get_recent_jobs(self, opaque, request_dict):

        # Remember last posted value for auto_update_enabled.
        auto_update_enabled = await self._handle_auto_update(
            opaque, request_dict, Cookies.RECENT_JOBS_UX
        )

        records = await bx_datafaces_get_default().get_bx_jobs(
            order_by="created_on DESC",
        )
        html = bx_composers_get_default().compose_bx_jobs(records)
        response = {"html": html, "auto_update_enabled": auto_update_enabled}

        return response

    # ----------------------------------------------------------------------------------------
    async def _update_job(self, opaque, request_dict):

        # Start a record to update the job.
        row = {
            BxJobFieldnames.UUID: request_dict[Keywords.BX_JOB_UUID],
        }

        for field in [BxJobFieldnames.COMMENT, BxJobFieldnames.RATING]:
            # If the job field is in the request, we will update it.
            if field in request_dict:
                row[field] = request_dict[field]

        # Do the actual update.
        await bx_datafaces_get_default().update_bx_job(row)

        return {}

    # ----------------------------------------------------------------------------------------
    async def _get_job_details(self, opaque, request_dict):

        # Remember last posted value for auto_update_enabled.
        auto_update_enabled = await self._handle_auto_update(
            opaque, request_dict, Cookies.JOB_DETAILS_UX
        )

        # Remember or get the last posted value for bx_job_uuid.
        bx_job_uuid = await self.set_or_get_cookie_content(
            opaque,
            Cookies.JOB_DETAILS_UX,
            "bx_job_uuid",
            request_dict.get("bx_job_uuid"),
            False,
        )
        if not bx_job_uuid:
            bx_job_uuid = None

        # For short.
        composer = bx_composers_get_default()
        dataface = bx_datafaces_get_default()

        workflow_filename_classname = None
        data_label = None
        job_label = None
        if bx_job_uuid is not None:
            # bx_job_uuid = require("request json", request_dict, "bx_job_uuid")
            bx_job_record = await dataface.get_bx_job(bx_job_uuid)
            if bx_job_record is None:
                return {"has_been_deleted": True}

            data_label = bx_job_record.get(BxJobFieldnames.DATA_LABEL)
            job_label = bx_job_record.get(BxJobFieldnames.LABEL)

            # Look up the tasks-and-gates records from the database.
            bx_jobs_bx_tasks_bx_gates_records = (
                await dataface.get_bx_jobs_bx_tasks_bx_gates(
                    bx_job_uuid=bx_job_uuid, why="[GUIPOLL]"
                )
            )
            # Turn this into nice html.
            html = callsign_html(self)
            html += composer.compose_bx_job_details(
                bx_job_record, bx_jobs_bx_tasks_bx_gates_records
            )

            # Look up the full job record.
            record = await dataface.get_bx_job(bx_job_uuid, why="[GUIPOLL]")
            # Turn this into the job summary line html.
            job_summary_html = callsign_html(self)
            job_summary_html += composer.compose_bx_jobs([record])

            # Get the workflow record for this job.
            bx_workflow_records = await dataface.get_bx_workflows([bx_job_uuid])
            settings_html = callsign_html(self)
            if len(bx_workflow_records) > 0:
                bx_workflow_record = bx_workflow_records[0]
                workflow_filename_classname = bx_workflow_record[
                    BxWorkflowFieldnames.FILENAME_CLASSNAME
                ]

                # Get the workflow's settings from the json field in the record.
                bx_settings = BxSettings(f"workflow for job {bx_job_record['label']}")
                bx_settings.load_from_json(bx_workflow_record["bx_settings_json"])
                settings_html += bx_settings.compose_html_outputs()

            else:
                settings_html += "no workflow settings found for this job"

        else:
            html = callsign_html(self)
            html += "<div>Please select a job from one of the other tabs.</div>"
            job_summary_html = ""
            settings_html = ""

        response = {
            "html": html,
            "job_summary_html": job_summary_html,
            "settings_html": settings_html,
            "workflow_filename_classname": workflow_filename_classname,
            "data_label": data_label,
            "job_label": job_label,
            "auto_update_enabled": auto_update_enabled,
            "bx_job_uuid": bx_job_uuid,
        }

        return response

    # ----------------------------------------------------------------------------------------
    async def _get_job_news(self, opaque, request_dict):

        # Remember or get the last posted value for bx_job_uuid.
        bx_job_uuid = await self.set_or_get_cookie_content(
            opaque,
            Cookies.JOB_NEWS_UX,
            "bx_job_uuid",
            request_dict.get("bx_job_uuid"),
            False,
        )

        if bx_job_uuid:
            records = await bx_datafaces_get_default().get_bx_news(
                bx_job_uuid=bx_job_uuid, why="[GUIPOLL]"
            )
            html = bx_composers_get_default().compose_bx_news(records)
        else:
            html = "Please select a job."

        response = {
            "html": html,
            "bx_job_uuid": bx_job_uuid,
        }

        return response

    # ----------------------------------------------------------------------------------------
    async def _get_job_variables(self, opaque, request_dict):

        # Remember or get the last posted value for bx_job_uuid.
        bx_job_uuid = await self.set_or_get_cookie_content(
            opaque,
            Cookies.JOB_VARIABLES_UX,
            "bx_job_uuid",
            request_dict.get("bx_job_uuid"),
            False,
        )

        if bx_job_uuid:
            bx_variable_records = await bx_datafaces_get_default().get_bx_variables(
                bx_job_uuid, why="[GUIPOLL]"
            )
            html = bx_composers_get_default().compose_bx_variables(bx_variable_records)
            response = {"html": html}
        else:
            html = "Please select a job."

        response = {
            "html": html,
            "bx_job_uuid": bx_job_uuid,
        }

        return response

    # ----------------------------------------------------------------------------------------
    async def _get_job_data_grid(self, opaque, request_dict):
        """
        Compose the html to show the Job Data Grid.
        This has data labels as rows, and job labels as columns.
        The columns come from discover of existing jobs.
        The bx_configurator allows to add or exclude columns.
        """

        # Remember last posted value for auto_update_enabled.
        auto_update_enabled = await self._handle_auto_update(
            opaque, request_dict, Cookies.JOB_DATA_GRID_UX
        )

        # Get all jobs. This will give workflow_filename_classname also.
        records = await bx_datafaces_get_default().get_bx_jobs(
            order_by="data_label ASC",
        )

        # Get the columns as specified from the configurator.
        try:
            prepend_job_labels = bx_configurators_get_default().require(
                "gui.job_data_grid.prepend_job_labels"
            )
        except NotFound:
            prepend_job_labels = None

        try:
            append_job_labels = bx_configurators_get_default().require(
                "gui.job_data_grid.append_job_labels"
            )
        except NotFound:
            append_job_labels = None

        try:
            specific_job_labels = bx_configurators_get_default().require(
                "gui.job_data_grid.specific_job_labels"
            )
        except NotFound:
            specific_job_labels = None

        try:
            exclude_job_labels = bx_configurators_get_default().require(
                "gui.job_data_grid.exclude_job_labels"
            )
        except NotFound:
            exclude_job_labels = None

        # Compose the html using prettytable, with special cell composer method.
        html = bx_composers_get_default().compose_bx_jobs_data_grid(
            records,
            append_job_labels=append_job_labels,
            prepend_job_labels=prepend_job_labels,
            specific_job_labels=specific_job_labels,
            exclude_job_labels=exclude_job_labels,
        )

        response = {"html": html, "auto_update_enabled": auto_update_enabled}

        return response

    # ----------------------------------------------------------------------------------------
    async def _connect_registered_launchers(self):

        launcher_records = await bx_datafaces_get_default().get_bx_launchers()

        for launcher_record in launcher_records:

            launcher_uuid = launcher_record["uuid"]

            # We don't already have a session to this launcher?
            if not self._bx_launchers.has(launcher_uuid):

                # Build a client session to the launcher.
                launcher_specification = json.loads(launcher_record["specification"])

                launcher_client = self._bx_launchers.build_object(
                    launcher_specification,
                    predefined_uuid=launcher_uuid,
                )
                self._bx_launchers.add(launcher_client)

            launcher_client = self._bx_launchers.find(launcher_uuid)

    # ----------------------------------------------------------------------------------------
    async def _get_launcher_health_report(self, launcher):
        """
        Get health report for one launcher service.
        """

        try:
            health_report = await launcher.client_report_health()
        except Exception as exception:
            health_report = {"exception": str(exception)}

        if "exception" in health_report:
            health_report["state"] = "exception"
        else:
            if "state" not in health_report:
                health_report["state"] = "ok"

        health_report["name"] = f"launcher {launcher.uuid()}"

        return health_report

    # ----------------------------------------------------------------------------------------
    async def _get_service_health_report(self, name, default):
        """
        Get health report for one service.
        """

        try:
            health_report = await default().client_report_health()
        except Exception as exception:
            health_report = {"exception": str(exception)}

        if "exception" in health_report:
            health_report["state"] = "exception"
        else:
            if "state" not in health_report:
                health_report["state"] = "ok"

        health_report["name"] = name

        return health_report

    # ----------------------------------------------------------------------------------------
    async def _get_system_health(self, opaque, request_dict):
        """
        Compose the html to show the System Health display.
        This is done by interrogating each server in turn.
        TODO: Gui should get system health in parallel network requests.
        """

        # Remember last posted value for auto_update_enabled.
        auto_update_enabled = await self._handle_auto_update(
            opaque, request_dict, Cookies.SYSTEM_HEALTH_UX
        )

        await self._connect_registered_launchers()

        health_reports = []

        health_reports.append(
            await self._get_service_health_report(
                "data interface",
                bx_datafaces_get_default,
            )
        )
        health_reports.append(
            await self._get_service_health_report(
                "scheduler",
                bx_schedulers_get_default,
            )
        )

        for launcher in self._bx_launchers.list():
            health_reports.append(await self._get_launcher_health_report(launcher))

        health_reports.append(
            await self._get_service_health_report(
                "catalog",
                bx_catalogs_get_default,
            )
        )
        health_reports.append(
            await self._get_service_health_report(
                "collector",
                bx_collectors_get_default,
            )
        )
        health_reports.append(
            await self._get_service_health_report(
                "news",
                bx_news_get_default,
            )
        )

        # -----------------------------------------------------------------
        health_report = await self.report_health()
        # logger.debug("dataface health report\n%s" % (json.dumps(health_report)))

        if "exception" in health_report:
            health_report["state"] = "exception"
        else:
            if "state" not in health_report:
                health_report["state"] = "ok"

        health_report["name"] = "gui"
        health_reports.append(health_report)

        # -----------------------------------------------------------------

        html = bx_composers_get_default().compose_health_reports(health_reports)

        response = {"html": html, "auto_update_enabled": auto_update_enabled}

        return response

    # ----------------------------------------------------------------------------------------
    async def _load_tabs(self, opaque, request_dict):

        tab_id = await self.get_cookie_content(
            opaque, Cookies.TABS_MANAGER, Keywords.TAB_ID
        )
        logger.debug(f"[GUITABS] tab_id from cookie content is {tab_id}")

        # Reply with tabs.
        response = {Keywords.TAB_ID: tab_id}

        return response

    # ----------------------------------------------------------------------------------------
    async def _select_tab(self, opaque, request_dict):
        tab_id = require("request json", request_dict, Keywords.TAB_ID)

        logger.debug(f"[GUITABS] tab_id in request is {tab_id}")

        # Put the tab_id into the cookie.
        self.set_cookie_content(opaque, Cookies.TABS_MANAGER, Keywords.TAB_ID, tab_id)

        response = {}

        return response

    # ----------------------------------------------------------------------------------------
    async def _show_workflow_settings(self, opaque, request_dict):
        """
        Build response for request to show workflow settings.
        """
        workflow_filename_classname = require(
            "request json", request_dict, "workflow_filename_classname"
        )
        data_label = require("request json", request_dict, "data_label")
        job_label = require("request json", request_dict, "job_label")
        bx_job_uuid = request_dict.get("bx_job_uuid")

        if (
            workflow_filename_classname is not None
            and workflow_filename_classname != ""
            and data_label is not None
            and data_label != ""
        ):

            logger.debug(
                f"[WORKSET] importing workflow class from {workflow_filename_classname}"
            )

            class_object = import_class(workflow_filename_classname)

            workflow_constructor_kwargs = {}

            # We have been given a previous job to get the workflow from?
            if bx_job_uuid is not None and bx_job_uuid != "" and bx_job_uuid != "-":
                # Get workflow record for the job.
                records = await bx_datafaces_get_default().get_bx_workflows(
                    [bx_job_uuid]
                )

                # There is a record for the workflow's job?
                if len(records) > 0:
                    # Load up the previous settings.
                    settings = BxSettings()
                    settings.load_from_json(
                        records[0][BxWorkflowFieldnames.BX_SETTINGS_JSON]
                    )

                    settings = settings.as_dict()

                    for key, value in settings.items():
                        # The old setting is still wanted by the new workflow?
                        if key in class_object.constructor_kwargs:
                            workflow_constructor_kwargs[key] = value

            # Override workflow settings with posted values.
            for key, value in request_dict.items():
                if key in class_object.constructor_kwargs:
                    workflow_constructor_kwargs[key] = value

            if "notebook" in class_object.constructor_kwargs:
                workflow_constructor_kwargs["notebook"] = job_label

            logger.debug(
                f"[WORKSET] constructing {callsign(class_object)}"
                " with kwargs\n{json.dumps(workflow_constructor_kwargs)}"
            )

            # Construct the workflow instance.
            bx_workflow = class_object(**workflow_constructor_kwargs)

            # Let the workflow build itself.
            # This allows it to add dynamic settings.
            # if inspect.iscoroutinefunction(bx_workflow.build):
            #     await bx_workflow.build()
            # else:
            #     bx_workflow.build()

            # Compose html input fields from workflow's settings.
            settings_html = bx_workflow.compose_html_inputs()

        else:
            settings_html = "<div class='T_notfound'>The workflow or data_label"
            " needed to compose this panel are not sufficient.</div>"

        # Componse settings into html input fields.
        response = {
            "workflow_filename_classname": workflow_filename_classname,
            "data_label": data_label,
            "settings_html": settings_html,
        }

        return response

    # ----------------------------------------------------------------------------------------
    async def _show_workflow_settings_WIP(self, opaque, request_dict):

        # Remember or get the last posted value for bx_job_uuid.
        bx_job_uuid = await self.set_or_get_cookie_content(
            opaque,
            Cookies.JOB_SUBMIT_UX,
            "bx_job_uuid",
            request_dict.get("bx_job_uuid"),
            False,
        )

        # For short.
        dataface = bx_datafaces_get_default()

        workflow_filename_classname = None

        # Start rendering the html with our callsign.
        settings_html = callsign_html(self)

        if bx_job_uuid:
            bx_job_record = await dataface.get_bx_job(bx_job_uuid)

            # Get the workflow record for this job.
            bx_workflow_records = await dataface.get_bx_workflows([bx_job_uuid])

            # Found any workflow records?
            if len(bx_workflow_records) > 0:
                bx_workflow_record = bx_workflow_records[0]
                workflow_filename_classname = bx_workflow_record[
                    BxWorkflowFieldnames.FILENAME_CLASSNAME
                ]

                # Get the workflow's settings from the json field in the record.
                bx_settings = BxSettings(f"workflow for job {bx_job_record['label']}")
                bx_settings.load_from_json(bx_workflow_record["bx_settings_json"])
                settings_html += bx_settings.compose_html_inputs()

            else:
                settings_html += "no workflow settings found for this job"

        else:
            html = callsign_html(self)
            html += "<div>Please select a job from another tab.</div>"

        # Componse settings into html input fields.
        response = {
            "workflow_filename_classname": workflow_filename_classname,
            "settings_html": settings_html,
        }

        return response

    # ----------------------------------------------------------------------------------------
    async def _start_workflow(self, opaque, request_dict):
        """
        Implement API call to start a new workflow.
        HTML form data for the workflow kwargs is posted in the request dict payload.
        """
        workflow_filename_classname = require(
            "request json", request_dict, "workflow_filename_classname"
        )
        data_label = require("request json", request_dict, "data_label")

        logger.debug(
            describe(
                "[WORKSTART] received workflow_filename_classname",
                workflow_filename_classname,
            )
        )
        logger.debug(describe("[WORKSTART] received data_label", data_label))

        # No classname provided in posted data?
        if workflow_filename_classname is None or workflow_filename_classname == "":
            # Use what's in the cookie.
            workflow_filename_classname = await self.get_cookie_content(
                opaque, Cookies.JOB_SUBMIT_UX, "workflow_filename_classname"
            )
            logger.debug(
                "[WORKSTART] workflow_filename_classname"
                f" from cookie content is {workflow_filename_classname}"
            )
        # A classname is posted?
        else:
            # Remember it in the cookie.
            self.set_cookie_content(
                opaque,
                Cookies.JOB_SUBMIT_UX,
                "workflow_filename_classname",
                workflow_filename_classname,
            )

        # Same for data_label.
        if data_label is None or data_label == "":
            data_label = await self.get_cookie_content(
                opaque, Cookies.JOB_SUBMIT_UX, "data_label"
            )
            logger.debug(f"[WORKSTART] data_label from cookie content is {data_label}")
        else:
            self.set_cookie_content(
                opaque, Cookies.JOB_SUBMIT_UX, "data_label", data_label
            )

        if workflow_filename_classname is None or workflow_filename_classname == "":
            raise RuntimeError(
                "no workflow_filename_classname provided in request or in cookie"
            )

        logger.debug(
            f"[WORKSTART] triggering workflow from {workflow_filename_classname}"
        )

        # Find the workflow class.
        workflow_finder = WorkflowFinder()
        class_object = workflow_finder.find_class_object(workflow_filename_classname)

        # Put everything from the form into the workflow constructor's kwargs.
        workflow_constructor_kwargs = require(
            "request json", request_dict, Keywords.PAYLOAD
        )

        # Also put in the data_label as a kwarg if it hasn't been submitted with the settings.
        # TODO: Iron out where data_label comes from when submitting a new workflow from the gui.
        if "data_label" in class_object.constructor_kwargs:
            if data_label is None or data_label == "":
                raise RuntimeError("no data_label provided in request or in cookie")
            if "data_label" not in workflow_constructor_kwargs:
                workflow_constructor_kwargs["data_label"] = data_label

        # Construct the workflow instance.
        logger.debug(
            f"[WORKSTART] constructing {workflow_filename_classname} with kwargs"
            f"\n{json.dumps(workflow_constructor_kwargs)}"
        )
        bx_workflow = class_object(**workflow_constructor_kwargs)

        # Register the workflow.
        logger.debug(f"[WORKSTART] building {workflow_filename_classname}")
        # Let the builder build the workflow.
        if inspect.iscoroutinefunction(bx_workflow.build):
            await bx_workflow.build()
        else:
            bx_workflow.build()

        # Start the workflow.
        logger.debug(f"[WORKSTART] starting {workflow_filename_classname}")
        await bx_workflow.start()

        # Return the uuid of the job we started.
        response = {
            "bx_job_uuid": bx_workflow.bx_job.uuid(),
        }

        return response

    # ----------------------------------------------------------------------------------------
    async def _start_workflow_nocookie(self, opaque, request_dict):
        """
        Implement API call to start a new workflow, without cookies.
        The json for the workflow kwargs is posted in the request dict payload.
        """
        workflow_filename_classname = require(
            "request json", request_dict, "workflow_filename_classname"
        )

        logger.debug(
            f"[WORKSTART] triggering workflow from {workflow_filename_classname}"
        )

        # Find the workflow class.
        workflow_finder = WorkflowFinder()
        class_object = workflow_finder.find_class_object(workflow_filename_classname)

        # Put everything from the form into the workflow constructor's kwargs.
        workflow_constructor_kwargs = require(
            "request json", request_dict, Keywords.PAYLOAD
        )

        # Construct the workflow instance.
        logger.debug(
            f"[WORKSTART] constructing {workflow_filename_classname} with kwargs"
            f"\n{json.dumps(workflow_constructor_kwargs)}"
        )
        bx_workflow = class_object(**workflow_constructor_kwargs)

        # Register the workflow.
        logger.debug(f"[WORKSTART] building {workflow_filename_classname}")
        # Let the builder build the workflow.
        if inspect.iscoroutinefunction(bx_workflow.build):
            await bx_workflow.build()
        else:
            bx_workflow.build()

        # Start the workflow.
        logger.debug(f"[WORKSTART] starting {workflow_filename_classname}")
        await bx_workflow.start()

        # Return the uuid of the job we started.
        response = {
            "bx_job_uuid": bx_workflow.bx_job.uuid(),
        }

        return response

    # ----------------------------------------------------------------------------------------
    async def _get_workflow_constructor_kwargs(self, opaque, request_dict):
        """
        Implement API call get a workflow's constructor kwargs.
        """
        workflow_filename_classname = require(
            "request json", request_dict, "workflow_filename_classname"
        )

        logger.debug(
            f"[WORKCONST] instantiating workflow from {workflow_filename_classname}"
        )

        # Find the workflow class.
        workflow_finder = WorkflowFinder()
        class_object = workflow_finder.find_class_object(workflow_filename_classname)

        if not hasattr(class_object, "constructor_kwargs"):
            raise RuntimeError(
                f"{workflow_filename_classname} has no constructor_kwargs attribute"
            )

        response = {Keywords.PAYLOAD: class_object.constructor_kwargs}

        return response

    # ----------------------------------------------------------------------------------------
    async def _query_logstore(self, opaque, request_dict):

        # Remember last posted value for auto_update_enabled.
        auto_update_enabled = request_dict.get("auto_update_enabled")
        # logger.debug(
        #     describe(f"[AUTOUP] request_dict auto_update_enabled", auto_update_enabled)
        # )
        auto_update_enabled = await self.set_or_get_cookie_content(
            opaque,
            Cookies.ERROR_LOGS_UX,
            "auto_update_enabled",
            auto_update_enabled,
            False,
        )
        # logger.debug(
        #     describe(
        #         f"[AUTOUP] request_set_or_get_cookie_content auto_update_enabled",
        #         auto_update_enabled,
        #     )
        # )

        payload = require("ajax request", request_dict, Keywords.PAYLOAD)
        where_ands = require("ajax request payload", payload, "where_ands")

        # Query the records from the logstore.
        records = await bx_logstores_get_default().query(where_ands)

        # Massage the messages.
        massaged = []
        for record in records:
            timestamp = record["timestamp"]
            logline = record["logline"]
            parts = logline.split("\n")
            keepers = []
            for part in parts:
                if "ERROR" in part or "EXCEPTION" in part:
                    keepers.append(part)
            if len(keepers) > 0:
                massaged.append({"timestamp": timestamp, "logline": "\n".join(keepers)})

        # Add the records from the query.
        # Expected to have at least {timestamp, logline} keys per record.
        response = {"records": massaged, "auto_update_enabled": auto_update_enabled}

        return response
