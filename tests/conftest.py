import logging
import os
import shutil

import pytest

# Formatting of testing log messages.
from dls_logformatter.dls_logformatter import DlsLogformatter

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    # TODO: Find a way to test ispyb in unit tests.
    parser.addoption(
        "--ispyb",
        action="store_true",
        dest="ispyb",
        default=False,
        help="enable ispyb tests",
    )

    # TODO: Find a way to test activemq in unit tests.
    parser.addoption(
        "--activemq",
        action="store_true",
        dest="activemq",
        default=False,
        help="enable activemq tests",
    )

    # TODO: Find a way to test gpu in unit tests.
    parser.addoption(
        "--gpu",
        action="store_true",
        dest="gpu",
        default=False,
        help="enable gpu tests",
    )

    # TODO: Find a way to test graylog in unit tests.
    parser.addoption(
        "--graylog",
        action="store_true",
        dest="graylog",
        default=False,
        help="enable graylog tests",
    )

    # TODO: Find a way to revitalize making dataface calls from main_isolated.
    parser.addoption(
        "--isolated_dataface",
        action="store_true",
        dest="isolated_dataface",
        default=False,
        help="enable isolated_dataface tests",
    )


# --------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def constants(request):

    constants = {}

    yield constants


# --------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def logging_setup():
    # print("")

    formatter = DlsLogformatter(type="long")
    logger = logging.StreamHandler()
    logger.setFormatter(formatter)
    logging.getLogger().addHandler(logger)

    # Log level for all modules.
    logging.getLogger().setLevel("DEBUG")

    # Turn off noisy debug.
    logging.getLogger("asyncio").setLevel("WARNING")
    logging.getLogger("pika").setLevel("WARNING")
    logging.getLogger("stomp").setLevel("WARNING")
    logging.getLogger("urllib3.connectionpool").setLevel("INFO")

    logging.getLogger("dls_bxflow_lib.things").setLevel("INFO")

    # Messages about starting and stopping services.
    logging.getLogger("dls_bxflow_lib.base_aiohttp").setLevel("INFO")

    # Messages about starting tasks and isolation.
    logging.getLogger("dls_bxflow_lib.bx_launchers.aiohttp").setLevel("INFO")
    logging.getLogger("dls_bxflow_lib.bx_launchers.base").setLevel("INFO")

    # All bxflow database sql commands.
    logging.getLogger("dls_bxflow_lib.bx_databases.aiosqlite").setLevel("INFO")

    # All ispyb database sql commands.
    # logging.getLogger("dls_bxflow_lib.bx_catalogs.ispyb").setLevel("INFO")

    logging.getLogger("dls_bxflow_lib.bx_contexts.classic").setLevel("INFO")
    logging.getLogger("dls_bxflow_lib.bx_news.context").setLevel("INFO")
    logging.getLogger("dls_bxflow_lib.bx_datafaces.context").setLevel("INFO")
    logging.getLogger("dls_bxflow_lib.bx_datafaces.news_producer").setLevel("INFO")
    logging.getLogger("dls_bxflow_lib.bx_launchers.context").setLevel("INFO")
    logging.getLogger("dls_bxflow_lib.bx_schedulers.context").setLevel("INFO")

    # Producing and consuming headlines.
    logging.getLogger("dls_bxflow_lib.bx_news.zmq_pubsub").setLevel("INFO")

    # Registering signal handler.
    logging.getLogger("dls_utilpack.signal").setLevel("INFO")

    # Sending and receiving zmq.
    logging.getLogger("dls_pairstream_lib.zmq_pubsub.writer").setLevel("INFO")
    logging.getLogger("dls_pairstream_lib.zmq_pubsub.reader").setLevel("INFO")

    # Set filter on the traitlets logger (used by nbclient and nbconvert).
    logging.getLogger("traitlets").addFilter(_traitlets_logging_filter())

    # Set filter on the ispyb logger to ignore the annoying NOTICE.
    logging.getLogger("ispyb").addFilter(_ispyb_logging_filter())

    # Cover the version.
    # logger.info("\n%s", (json.dumps(version_meta(), indent=4)))

    yield None


# --------------------------------------------------------------------------------
class _traitlets_logging_filter:
    """
    Python logging filter to remove annoying traitlets messages.
    These are not super useful to see all the time at the DEBUG level.
    """

    def filter(self, record):

        if record.levelno == 10:
            if "jupyter_client/client.py" in record.pathname:
                return 0
            if "jupyter_client/connect.py" in record.pathname:
                return 0
            if "jupyter_client/manager.py" in record.pathname:
                return 0
            if "jupyter_client/provisioning/factory.py" in record.pathname:
                return 0
            if "nbclient/client.py" in record.pathname:
                return 0
            if "nbconvert/exporters/templateexporter.py" in record.pathname:
                return 0
            if "nbconvert/preprocessors/base.py" in record.pathname:
                return 0
            if "/nbconvert/preprocessors/coalescestreams.py" in record.pathname:
                return 0

            # if "" in record.pathname:
            #     return 0

        return 1


# --------------------------------------------------------------------------------
class _ispyb_logging_filter:
    """
    Python logging filter to remove annoying traitlets messages.
    These are not super useful to see all the time at the DEBUG level.
    """

    def filter(self, record):

        if record.msg.startswith(
            "NOTICE: This code uses __future__ functionality in the ISPyB API."
        ):
            return 0

        return 1


# --------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def output_directory(request):
    # TODO: Better way to get a newline in conftest after pytest emits the test class name.
    print("")

    # Tmp directory which we can write into.
    output_directory = "/tmp/%s/%s/%s" % (
        "/".join(__file__.split("/")[-3:-1]),
        request.cls.__name__,
        request.function.__name__,
    )

    # Tmp directory which we can write into.
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=False, onerror=None)
    os.makedirs(output_directory)

    # logger.debug("output_directory is %s" % (output_directory))

    yield output_directory
