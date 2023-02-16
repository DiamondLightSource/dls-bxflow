import json
import logging

# Database field names.
from dls_bxflow_api.bx_databases.constants import BxJobFieldnames

# BxJobs manager.
# States of things.
from dls_bxflow_lib.bx_jobs.states import States as BxJobStates

# BXNews manager.
from dls_bxflow_lib.bx_news.bx_news import BxNews
from dls_bxflow_lib.bx_news.constants import Topics

# BxTasks manager.
from dls_bxflow_run.bx_tasks.states import States as BxTaskStates

logger = logging.getLogger(__name__)


class NewsProducer:
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, bx_dataface):
        self.__bx_dataface = bx_dataface

        bx_news_key = "bx_news_specification"
        bx_news_specification = bx_dataface.specification().get(bx_news_key)
        if bx_news_specification is not None:
            self.__bx_news = BxNews().build_object(bx_news_specification)
            # self.__bx_news.activate_producer()
            # time.sleep(0.100)
        else:
            self.__bx_news = None
            logger.warning(
                f"{bx_dataface.thing_type()} specification does not contain key {bx_news_key},"
                " so not sending notifications",
            )

    # ----------------------------------------------------------------------------------------
    async def disconnect(self):
        logger.debug("disconnecting news producer")
        await self.__bx_news.close_client_session()

    # ----------------------------------------------------------------------------------------
    async def open_bx_gate(self, bx_task_uuid, controlled_bx_gate_name, why=None):
        """"""

        bx_task_record = await self.__bx_dataface.get_bx_task(bx_task_uuid)
        bx_task_label = bx_task_record["label"]

        headline = f"{bx_task_label} indicated {controlled_bx_gate_name}"
        if why is not None:
            headline = f"{headline}, {why}"

        await self.__produce_news(
            Topics.BXGATE_WAS_OPENED,
            headline,
            {
                "bx_gate": {
                    "bx_task_uuid": bx_task_uuid,
                    "label": controlled_bx_gate_name,
                }
            },
            bx_task_uuid=bx_task_uuid,
        )

    # ----------------------------------------------------------------------------------------
    async def close_bx_gate(self, bx_task_uuid, controlled_bx_gate_name, why=None):
        """"""

        bx_task_record = await self.__bx_dataface.get_bx_task(bx_task_uuid)
        bx_task_label = bx_task_record["label"]

        headline = f"{bx_task_label} reset {controlled_bx_gate_name} to closed"
        if why is not None:
            headline = f"{headline}, {why}"

        await self.__produce_news(
            Topics.BXGATE_WAS_CLOSED,
            headline,
            {
                "bx_gate": {
                    "bx_task_uuid": bx_task_uuid,
                    "label": controlled_bx_gate_name,
                }
            },
            bx_task_uuid=bx_task_uuid,
        )

    # ----------------------------------------------------------------------------------------
    async def update_bx_job(self, row, why=None):
        """"""

        state = row.get("state")

        if state == BxJobStates.READY:
            topic = Topics.BXJOB_WAS_ENABLED
        elif state == BxJobStates.IN_PROGRESS:
            topic = Topics.BXJOB_WAS_STARTED
        elif state == BxJobStates.SUCCEEDED:
            topic = Topics.BXJOB_SUCCEEDED
        elif state == BxJobStates.FAILED:
            topic = Topics.BXJOB_FAILED
        elif state == BxJobStates.CANCELLED:
            topic = Topics.BXJOB_WAS_CANCELLED
        else:
            topic = None

        if topic is not None:

            bx_job_record = await self.__bx_dataface.get_bx_job(row["uuid"])
            bx_job_label = bx_job_record[BxJobFieldnames.LABEL]
            data_label = bx_job_record[BxJobFieldnames.DATA_LABEL]
            headline = f"{bx_job_label} for {data_label} became {state}"
            if why is not None:
                headline = f"{headline}, {why}"

            await self.__produce_news(
                topic,
                headline,
                {
                    "bx_job": {
                        "uuid": row["uuid"],
                        "state": state,
                        BxJobFieldnames.LABEL: bx_job_label,
                        BxJobFieldnames.DATA_LABEL: data_label,
                    }
                },
                bx_job_uuid=row["uuid"],
            )

    # ----------------------------------------------------------------------------------------
    async def delete_bx_job(self, bx_job_record):
        """"""

        topic = Topics.BXJOB_WAS_DELETED

        headline = f"{bx_job_record['label']} was deleted"

        await self.__produce_news(
            topic,
            headline,
            {
                "bx_job": {
                    "uuid": bx_job_record["uuid"],
                    "label": bx_job_record["label"],
                    "data_label": bx_job_record["data_label"],
                }
            },
        )

    # ----------------------------------------------------------------------------------------
    async def update_bx_task(self, row, why=None):
        """"""

        state = row.get("state")

        if state == BxTaskStates.STARTED:
            topic = Topics.BXTASK_WAS_STARTED
        elif state == BxTaskStates.FINISHED:
            topic = Topics.BXTASK_WAS_FINISHED
        else:
            topic = None

        if topic is not None:

            bx_task_record = await self.__bx_dataface.get_bx_task(row["uuid"])
            bx_task_label = bx_task_record["label"]

            headline = f"{bx_task_label} became {state}"
            if why is not None:
                headline = f"{headline}, {why}"

            await self.__produce_news(
                topic,
                headline,
                {"bx_task": {"uuid": row["uuid"]}},
                bx_task_uuid=row["uuid"],
            )

    # ----------------------------------------------------------------------------------------
    async def update_bx_launcher(self, row, launcher_callsign=None):
        """"""

        topic = Topics.BXLAUNCHER_WAS_UPDATED

        if launcher_callsign is None:
            launcher_callsign = "a launcher"

        headline = "%s became %s" % (launcher_callsign, row["state"])

        await self.__produce_news(
            topic,
            headline,
            {
                "bx_launcher": {
                    "uuid": row["uuid"],
                    "state": row["state"],
                }
            },
        )

    # ----------------------------------------------------------------------------------------
    async def __produce_news(
        self, topic, headline, details, bx_job_uuid=None, bx_task_uuid=None
    ):
        """"""

        if self.__bx_news is None:
            return

        # logger.debug("producing %s %s" % (topic, headline))

        await self.__bx_news.produce(topic, headline, details)

        record = {
            "bx_job_uuid": bx_job_uuid,
            "bx_task_uuid": bx_task_uuid,
            "topic": topic,
            "headline": headline,
            "details": json.dumps(details),
        }

        await self.__bx_dataface.set_bx_news([record])
