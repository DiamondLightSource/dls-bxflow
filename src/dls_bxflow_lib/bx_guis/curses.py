import logging
import time

# Types of bx_dataface.
# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Basic things.
from dls_bxflow_api.thing import Thing

# Object managing bx_composers.
from dls_bxflow_lib.bx_composers.bx_composers import bx_composers_get_default

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_guis.curses"


# ------------------------------------------------------------------------------------------
class Curses(Thing):
    """
    Object implementing remote procedure calls for bx_gui methods.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        Thing.__init__(self, thing_type, specification)

        self.__page_name = specification["type_specific_tbd"]["initial_page_name"]

    # ----------------------------------------------------------------------------------------
    async def activate_coro(self):
        """"""

        while True:
            bx_dataface = bx_datafaces_get_default()
            bx_composer = bx_composers_get_default()

            if self.__page_name == "bx_jobs_bx_tasks_bx_gates":
                records = await bx_dataface.get_bx_jobs_bx_tasks_bx_gates()
                text = bx_composer.compose_bx_jobs_bx_tasks_bx_gates(records)
                logger.info("\n%s" % (text))

            if self.__page_name == "recent_actions":
                records = await bx_dataface.get_bx_news()
                text = bx_composer.compose_bx_news(records)

                logger.info("\n%s" % (text))

            if self.__page_name == "recent_jobs":
                records = await bx_dataface.get_bx_jobs(None)
                text = bx_composer.compose_bx_jobs(records)

                logger.info("\n%s" % (text))

            if self.__page_name == "job_details":
                # For now, just pick the latest created job to get details of it.
                records = await bx_dataface.get_bx_jobs(
                    None, order_by="created_on", limit="1"
                )
                bx_job_uuid = records[0]["uuid"]

                records = await bx_dataface.get_bx_jobs_bx_tasks_bx_gates(
                    bx_job_uuid=bx_job_uuid
                )
                text = bx_composer.compose_bx_jobs_bx_tasks_bx_gates(records)

                logger.info("\n%s" % (text))

            time.sleep(1.0)
