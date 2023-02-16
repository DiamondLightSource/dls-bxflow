import logging

# Class for an aiohttp client.
from dls_bxflow_api.aiohttp_client import AiohttpClient

# BxDataface protocolj things.
from dls_bxflow_api.bx_datafaces.constants import Commands, Keywords

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------
class Aiohttp:
    """
    Object implementing client side API for talking to the bx_dataface server.
    Please see doctopic [A01].
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        self.__specification = specification

        self.__aiohttp_client = AiohttpClient(
            specification["type_specific_tbd"]["aiohttp_specification"],
        )

    # ----------------------------------------------------------------------------------------
    def specification(self):
        return self.__specification

    # ----------------------------------------------------------------------------------------
    async def set_bx_cookie(self, bx_cookie_dict):
        """ """
        return await self.__send_protocolj("set_bx_cookie", bx_cookie_dict)

    # ----------------------------------------------------------------------------------------
    async def set_bx_gates(self, bx_gate_dicts):
        """"""
        return await self.__send_protocolj("set_bx_gates", bx_gate_dicts)

    # ----------------------------------------------------------------------------------------
    async def set_bx_launcher(self, bx_launcher_dict):
        """"""
        return await self.__send_protocolj("set_bx_launcher", bx_launcher_dict)

    # ----------------------------------------------------------------------------------------
    async def set_bx_tasks(self, bx_task_dicts):
        """"""
        return await self.__send_protocolj("set_bx_tasks", bx_task_dicts)

    # ----------------------------------------------------------------------------------------
    async def set_bx_task_dependency_bx_gates(self, bx_task_dependency_bx_gate_dicts):
        """"""
        return await self.__send_protocolj(
            "set_bx_task_dependency_bx_gates", bx_task_dependency_bx_gate_dicts
        )

    # ----------------------------------------------------------------------------------------
    async def set_bx_job_blocked_by_bx_gates(self, bx_job_blocked_by_bx_gate_dicts):
        """"""
        return await self.__send_protocolj(
            "set_bx_job_blocked_by_bx_gates", bx_job_blocked_by_bx_gate_dicts
        )

    # ----------------------------------------------------------------------------------------
    async def set_bx_jobs(self, bx_task_dicts):
        """ """
        return await self.__send_protocolj("set_bx_jobs", bx_task_dicts)

    # ----------------------------------------------------------------------------------------
    async def set_bx_variables(self, variables_dict):
        """ """
        return await self.__send_protocolj("set_bx_variables", variables_dict)

    # ----------------------------------------------------------------------------------------
    async def set_bx_workflows(self, bx_workflows_dicts):
        """ """
        return await self.__send_protocolj("set_bx_workflows", bx_workflows_dicts)

    # ----------------------------------------------------------------------------------------
    async def get_bx_cookie(self, bx_cookie_uuid):
        """
        Get single bx_cookie from its uuid.
        Returns database record format.
        """
        return await self.__send_protocolj("get_bx_cookie", bx_cookie_uuid)

    # ----------------------------------------------------------------------------------------
    async def get_bx_launcher(self, bx_launcher_uuid):
        """
        Get single bx_launcher from its uuid.
        Returns database record format.
        """
        return await self.__send_protocolj("get_bx_launcher", bx_launcher_uuid)

    # ----------------------------------------------------------------------------------------
    async def get_bx_launchers(self, states=None):
        """
        Get bx_launchers with given states.
        Returns database records format.
        """
        return await self.__send_protocolj("get_bx_launchers", states=states)

    # ----------------------------------------------------------------------------------------
    async def get_bx_tasks_launched_by(self, bx_launcher_uuid, states=None):
        """
        Get bx_tasks with given states which where launched by given launcher.
        Returns database records format.
        """
        return await self.__send_protocolj(
            "get_bx_tasks_launched_by", bx_launcher_uuid, states=states
        )

    # ----------------------------------------------------------------------------------------
    async def get_bx_task(self, bx_task_uuid):
        """
        Get single bx_task from its uuid.
        Returns database record format.
        """
        return await self.__send_protocolj("get_bx_task", bx_task_uuid)

    # ----------------------------------------------------------------------------------------
    async def get_bx_tasks(self, bx_job_uuid):
        """
        Get all bx_tasks related to current bx_job.
        Returns database records format.
        """
        return await self.__send_protocolj("get_bx_tasks", bx_job_uuid)

    # ----------------------------------------------------------------------------------------
    async def get_bx_job(self, bx_job_uuid, why=None):
        """
        Get single bx_job from its uuid.
        Returns database record format.
        """
        return await self.__send_protocolj("get_bx_job", bx_job_uuid, why=why)

    # ----------------------------------------------------------------------------------------
    async def get_bx_jobs(
        self,
        states=None,
        labels=None,
        order_by=None,
        limit=None,
        why=None,
    ):
        """
        Get bx_jobs with given states.
        Returns database records format.
        """
        return await self.__send_protocolj(
            "get_bx_jobs",
            states=states,
            labels=labels,
            order_by=order_by,
            limit=limit,
            why=why,
        )

    # ----------------------------------------------------------------------------------------
    async def get_controlled_bx_gates(self, bx_job_uuid):
        """
        Get all bx_gates controlled by bx_tasks related to current bx_job.
        Returns database records format.
        """
        return await self.__send_protocolj("get_controlled_bx_gates", bx_job_uuid)

    # ----------------------------------------------------------------------------------------
    async def get_dependency_bx_gates(self, bx_job_uuid):
        """
        Get all bx_gates dependency by bx_tasks related to current bx_job.
        Returns database records format.
        """
        return await self.__send_protocolj("get_dependency_bx_gates", bx_job_uuid)

    # ----------------------------------------------------------------------------------------
    async def get_blocked_by_bx_gates(self, bx_job_uuid):
        """
        Get all bx_gates which block the current bx_job.
        Returns database records format.
        """
        return await self.__send_protocolj("get_blocked_by_bx_gates", bx_job_uuid)

    # ----------------------------------------------------------------------------------------
    async def get_bx_variables(self, bx_job_uuid, why=None):
        """
        Get the variables related to given bx_job.
        Returns database records format.
        """
        return await self.__send_protocolj("get_bx_variables", bx_job_uuid, why=why)

    # ----------------------------------------------------------------------------------------
    async def get_bx_workflows(self, bx_job_uuids, why=None):
        """
        Get all workflows related to given bx_jobs.
        Returns database records format.
        """
        return await self.__send_protocolj("get_bx_workflows", bx_job_uuids, why=why)

    # ----------------------------------------------------------------------------------------
    async def open_bx_gate(self, bx_task_uuid, controlled_bx_gate_name, why=None):
        """"""
        return await self.__send_protocolj(
            "open_bx_gate", bx_task_uuid, controlled_bx_gate_name, why=why
        )

    # ----------------------------------------------------------------------------------------
    async def close_bx_gate(self, bx_task_uuid, controlled_bx_gate_name, why=None):
        """"""
        return await self.__send_protocolj(
            "close_bx_gate", bx_task_uuid, controlled_bx_gate_name, why=why
        )

    # ----------------------------------------------------------------------------------------
    async def update_bx_cookie(self, row):
        """"""
        return await self.__send_protocolj("update_bx_cookie", row)

    # ----------------------------------------------------------------------------------------
    async def update_bx_launcher(self, row, launcher_callsign=None):
        """"""
        return await self.__send_protocolj(
            "update_bx_launcher", row, launcher_callsign=launcher_callsign
        )

    # ----------------------------------------------------------------------------------------
    async def update_bx_task(self, row):
        """"""
        return await self.__send_protocolj("update_bx_task", row)

    # ----------------------------------------------------------------------------------------
    async def update_bx_job(self, row):
        """"""
        return await self.__send_protocolj("update_bx_job", row)

    # ----------------------------------------------------------------------------------------
    async def cancel_bx_job(self, bx_job_uuid):
        """"""
        return await self.__send_protocolj("cancel_bx_job", bx_job_uuid)

    # ----------------------------------------------------------------------------------------
    async def delete_bx_job(self, bx_job_uuid):
        """ """
        return await self.__send_protocolj("delete_bx_job", bx_job_uuid)

    # ----------------------------------------------------------------------------------------
    async def delete_bx_launcher(self, bx_launcher_uuid):
        """ """
        return await self.__send_protocolj("delete_bx_launcher", bx_launcher_uuid)

    # ----------------------------------------------------------------------------------------
    async def unblock_bx_job(self, bx_job_uuid):
        """"""
        return await self.__send_protocolj("unblock_bx_job", bx_job_uuid)

    # ----------------------------------------------------------------------------------------
    async def enable_bx_job(self, uuid):
        """"""
        return await self.__send_protocolj("enable_bx_job", uuid)

    # ----------------------------------------------------------------------------------------
    async def update_bx_job_if_blocked_by_bx_gates(self, bx_job_uuid):
        return await self.__send_protocolj(
            "update_bx_job_if_blocked_by_bx_gates", bx_job_uuid
        )

    # ----------------------------------------------------------------------------------------
    async def get_bx_jobs_bx_tasks_bx_gates(self, bx_job_uuid=None, why=None):
        """"""
        return await self.__send_protocolj(
            "get_bx_jobs_bx_tasks_bx_gates", bx_job_uuid=bx_job_uuid, why=why
        )

    # ----------------------------------------------------------------------------------------
    async def get_bx_news(self, bx_job_uuid=None, why=None):
        """"""
        return await self.__send_protocolj(
            "get_bx_news", bx_job_uuid=bx_job_uuid, why=why
        )

    # ----------------------------------------------------------------------------------------
    async def query(self, sql, subs=None, why=None):
        """"""
        return await self.__send_protocolj("query", sql, subs=subs, why=why)

    # ----------------------------------------------------------------------------------------
    async def report_health(self):
        """"""
        return await self.__send_protocolj("report_health")

    # ----------------------------------------------------------------------------------------
    async def __send_protocolj(self, function, *args, **kwargs):
        """"""

        return await self.__aiohttp_client.client_protocolj(
            {
                Keywords.COMMAND: Commands.EXECUTE,
                Keywords.PAYLOAD: {
                    "function": function,
                    "args": args,
                    "kwargs": kwargs,
                },
            },
        )

    # ----------------------------------------------------------------------------------------
    async def close_client_session(self):
        """"""

        if self.__aiohttp_client is not None:
            await self.__aiohttp_client.close_client_session()

    # ----------------------------------------------------------------------------------------
    async def client_report_health(self):
        """"""

        return await self.__aiohttp_client.client_report_health()
