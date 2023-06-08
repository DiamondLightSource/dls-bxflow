class ClassTypes:
    POPENER = "dls_bxflow_lib.bx_launchers.popener"
    QSUBBER = "dls_bxflow_lib.bx_launchers.qsubber"
    SBATCHER = "dls_bxflow_lib.bx_launchers.sbatcher"
    AIOHTTP = "dls_bxflow_lib.bx_launchers.aiohttp"
    ISLAND = "dls_bxflow_lib.bx_launchers.island"


class Keywords:
    COMMAND = "bx_launchers::keywords::command"
    PAYLOAD = "bx_launchers::keywords::payload"


class Commands:
    EXECUTE = "bx_launchers::commands::execute"


class Queues:
    SUBMIT_ITASK = "bx_launchers__queues__submit_bx_task"


class Channels:
    COMMAND = "bx_launchers__channels__command"
