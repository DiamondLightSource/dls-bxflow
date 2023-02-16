import json
import logging
from queue import Queue

import stomp

# Utilities.
from dls_utilpack.describe import describe
from dls_utilpack.explain import explain

# Object managing bx_datafaces.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Object managing bx_composers.
from dls_bxflow_lib.bx_composers.bx_composers import bx_composers_get_default

# Object managing bx_jobs.
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

# Base class for running workflows.
from tests.base import Base

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class Grokker:
    """
    This is a user-defined class which gets instantiated and called at runtime.
    """

    def __init__(self, message):
        self.__message = message

    def run(self):
        logger.info(
            "MyGrok running on message\n%s" % (json.dumps(self.__message, indent=4))
        )


# ----------------------------------------------------------------------------------------
class MyListener(stomp.ConnectionListener):
    def __init__(self, queue):
        self.__queue = queue

    def on_error(self, headers, message):
        print('received an error "%s"' % message)

    def on_message(self, frame):
        try:
            logger.info(describe("got frame which", frame))
            self.__queue.put_nowait(frame)
            logger.info("put that frame")
        except Exception as exception:
            logger.error(explain(exception, "enqueuing frame"), exc_info=exception)


# ----------------------------------------------------------------------------------------
class GdaWorkflow(Base):
    async def _main_coroutine(
        self, constants, infrastructure_context, output_directory
    ):
        """ """

        async with infrastructure_context:
            try:
                stomp_connection = stomp.Connection([("activemq", 61613)])

                # queue = asyncio.Queue()
                queue = Queue()
                listener = MyListener(queue)

                stomp_connection.set_listener("", listener)
                # stomp_connection.start()
                stomp_connection.connect()

                stomp_connection.subscribe(
                    destination="/topic/gda.messages.scan", id=1, ack="auto"
                )

                message = {"a": "a"}
                stomp_connection.send(
                    body=json.dumps(message), destination="/topic/gda.messages.scan"
                )

                logger.info("waiting for queue...")
                frame = queue.get()
                await self.__grok(frame)

            finally:
                stomp_connection.disconnect()

    # ----------------------------------------------------------------------------------------
    async def __grok(self, frame):
        logger.info(describe("got frame which", frame))

        body = frame.body
        message = json.loads(body)

        # Define the class and its constructor arguments.
        grok_task = BxTasks().build_object(
            {
                "type": "dls_bxflow_run.bx_tasks.pickled_class",
                "label": "grokker",
                "type_specific_tbd": {
                    "class": Grokker,
                    "constructor_args": (message,),
                    "constructor_kwargs": {},
                },
            }
        )

        # Create a bx_job as collection of tasks.
        bx_job = BxJobs().build_object(
            {"type": "dls_bxflow_lib.bx_jobs.standard", "label": "bx_job1"}
        )
        bx_job.bx_tasks.add([grok_task])

        # Tell the bx_job what will block its further execution.
        bx_job.blocked_by_bx_gates.add(
            [
                grok_task.failure_bx_gate,
                grok_task.success_bx_gate,
            ]
        )

        # Schedule the bx_job to run.
        await bx_job.enable()

        # Wait for bx_job to finish.
        await bx_job.wait(timeout=20.0)

        # Print the bx_news that happened.
        news_records = await bx_datafaces_get_default().get_bx_news(
            bx_job_uuid=bx_job.uuid()
        )
        bx_news_pretty = bx_composers_get_default().compose_bx_news(news_records)
        logger.info("bx_news\n%s" % (bx_news_pretty))
