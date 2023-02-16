import logging
import time

# Utilities.
from dls_utilpack.require import require

# Class for an aiohttp client.
from dls_bxflow_api.aiohttp_client import AiohttpClient

# Base class for bx_logstore instances.
from dls_bxflow_lib.bx_logstores.base import Base as BxLogstoreBase

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_logstores.graylogger"


# ------------------------------------------------------------------------------------------
class Graylogger(BxLogstoreBase):
    """
    Object representing a bx_logstore which is based on graylog.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification, predefined_uuid=None):
        BxLogstoreBase.__init__(
            self, thing_type, specification, predefined_uuid=predefined_uuid
        )

        self.__aiohttp_client = AiohttpClient(
            specification["type_specific_tbd"],
        )

    # ----------------------------------------------------------------------------------------
    async def query(
        self,
        where_ands,
    ):

        # All log entries.
        # TODO: Limit graylog search to particular index.
        lucenes = ["_exists_:dls_message_plus_exception"]

        from_date = "2021-12-31T00:00:00.000Z"

        for where_and in where_ands:
            if where_and["field"] == "timestamp" and where_and["operator"] == ">":
                from_date = where_and["operand"]
            elif where_and["operator"] == "IN":
                ors = []
                for operand in where_and["operand"]:
                    ors.append(f'"{operand}"')
                lucenes.append("(%s:(%s))" % (where_and["field"], " OR ".join(ors)))

        to_date = "2030-12-31T00:00:00.000Z"
        limit = 100
        offset = 0

        # Just want this one line in the csv.
        fields = "dls_message_plus_exception"

        # Always sort in timestamp order.
        # This is the same timestamp as is formatted into dls_message_plus_exception.
        sort = "timestamp:desc"

        params = {
            "query": " AND ".join(lucenes),
            "from": from_date,
            "to": to_date,
            "limit": limit,
            "offset": offset,
            "fields": fields,
            "sort": sort,
            "decorate": "false",
        }

        t0 = time.time()
        response = await self.__aiohttp_client.client_get_json(
            "api/search/universal/absolute",
            params=params,
        )
        t1 = time.time()
        logger.debug("query graylog took %0.3f seconds" % (t1 - t0))

        messages = require("graylog query response", response, "messages")
        records = []
        # Reverse the message into ascending timestamp.
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]["message"]
            record = {
                "timestamp": require("log message", message, "timestamp"),
                "logline": require(
                    "log message", message, "dls_message_plus_exception"
                ),
            }
            records.append(record)

        # logger.info(describe("response", response))

        return records

    # ----------------------------------------------------------------------------------------
    async def close_client_session(self):
        """"""

        if self.__aiohttp_client is not None:
            await self.__aiohttp_client.close_client_session()
