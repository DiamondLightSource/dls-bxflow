import asyncio
import logging

# Utilities.
from dls_servbase_lib.datafaces.context import Context as DlsServbaseDatafaceContext
from dls_utilpack.callsign import callsign
from dls_utilpack.explain import explain

from dls_bxflow_lib.bx_catalogs.context import Context as BxCatalogContext
from dls_bxflow_lib.bx_collectors.context import Context as BxCollectorContext

# Base class which maps flask requests to methods.
from dls_bxflow_lib.bx_contexts.base import Base
from dls_bxflow_lib.bx_datafaces.context import Context as BxDatafaceContext

# Contexts.
from dls_bxflow_lib.bx_filestores.context import Context as BxFilestoreContext
from dls_bxflow_lib.bx_guis.context import Context as BxGuiContext
from dls_bxflow_lib.bx_jobs.context import Context as BxJobContext
from dls_bxflow_lib.bx_launchers.context import Context as BxLauncherContext
from dls_bxflow_lib.bx_logstores.context import Context as BxLogstoreContext
from dls_bxflow_lib.bx_news.context import Context as BxNewsContext
from dls_bxflow_lib.bx_schedulers.context import Context as BxSchedulerContext

logger = logging.getLogger(__name__)


thing_type = "dls_bxflow_lib.bx_contexts.classic"


class Classic(Base):
    """
    Object representing an event bx_dataface connection.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification):
        Base.__init__(self, thing_type, specification)

        self.__filestore = None
        self.__news = None
        self.__dls_servbase_dataface = None
        self.__dataface = None
        self.__catalog = None
        self.__launchers = []
        self.__job = None
        self.__scheduler = None
        self.__collector = None
        self.__gui = None
        self.__logstore = None

    # ----------------------------------------------------------------------------------------
    async def __dead_or_alive(self, context, dead, alive):

        if context is not None:
            try:
                # A server was defined for this context?
                if await context.is_process_started():
                    if await context.is_process_alive():
                        alive.append(context)
                    else:
                        dead.append(context)
            except Exception:
                raise RuntimeError(
                    f"unable to determine dead or alive for context {callsign(context)}"
                )

    # ----------------------------------------------------------------------------------------
    async def __dead_or_alive_all(self):
        """
        Return two lists, one for dead and one for alive processes.
        TODO: Parallelize context process alive/dead checking.
        """

        dead = []
        alive = []

        await self.__dead_or_alive(self.__news, dead, alive)
        await self.__dead_or_alive(self.__collector, dead, alive)
        await self.__dead_or_alive(self.__dls_servbase_dataface, dead, alive)
        await self.__dead_or_alive(self.__dataface, dead, alive)
        await self.__dead_or_alive(self.__catalog, dead, alive)
        for launcher in self.__launchers:
            await self.__dead_or_alive(launcher, dead, alive)
        await self.__dead_or_alive(self.__scheduler, dead, alive)
        await self.__dead_or_alive(self.__gui, dead, alive)

        return dead, alive

    # ----------------------------------------------------------------------------------------
    async def is_any_process_alive(self):
        """
        Check all configured processes, return if any alive.
        """
        dead, alive = await self.__dead_or_alive_all()

        # logger.debug(f"[PIDAL] {len(dead)} processes are dead, {len(alive)} are alive")

        return len(alive) > 0

    # ----------------------------------------------------------------------------------------
    async def is_any_process_dead(self):
        """
        Check all configured processes, return if any alive.
        """
        dead, alive = await self.__dead_or_alive_all()

        return len(dead) > 0

    # ----------------------------------------------------------------------------------------
    async def __aenter__(self):
        """ """
        logger.debug(f"entering {callsign(self)} context")

        try:

            try:
                specification = self.specification().get("bx_filestore_specification")
                if specification is not None:
                    logger.debug(f"at entering position {callsign(self)} FILESTORE")
                    self.__filestore = BxFilestoreContext(specification)
                    await self.__filestore.aenter()
            except Exception as exception:
                raise RuntimeError(
                    explain(exception, f"creating {callsign(self)} filestore context")
                )

            try:
                specification = self.specification().get("bx_logstore_specification")
                if specification is not None:
                    logger.debug(f"at entering position {callsign(self)} LOGSTORE")
                    self.__logstore = BxLogstoreContext(specification)
                    await self.__logstore.aenter()
            except Exception as exception:
                raise RuntimeError(
                    explain(exception, f"creating {callsign(self)} logstore context")
                )

            try:
                specification = self.specification().get("bx_news_specification")
                if specification is not None:
                    logger.debug(f"at entering position {callsign(self)} NEWS")
                    self.__news = BxNewsContext(specification)
                    await self.__news.aenter()
            except Exception as exception:
                raise RuntimeError(
                    explain(exception, f"creating {callsign(self)} news context")
                )

            try:
                specification = self.specification().get(
                    "dls_servbase_dataface_specification"
                )
                if specification is not None:
                    logger.debug(
                        f"at entering position {callsign(self)} DLS_SERVBASE DATAFACE"
                    )
                    self.__dls_servbase_dataface = DlsServbaseDatafaceContext(
                        specification
                    )
                    await self.__dls_servbase_dataface.aenter()
                else:
                    logger.debug(
                        f"no specification in {callsign(self)} for DLS_SERVBASE DATAFACE"
                    )
            except Exception as exception:
                raise RuntimeError(
                    explain(
                        exception,
                        f"creating {callsign(self)} dls_servbase_dataface context",
                    )
                )

            try:
                specification = self.specification().get("bx_dataface_specification")
                if specification is not None:
                    logger.debug(f"at entering position {callsign(self)} DATAFACE")
                    self.__dataface = BxDatafaceContext(specification)
                    await self.__dataface.aenter()
            except Exception as exception:
                raise RuntimeError(
                    explain(exception, f"creating {callsign(self)} dataface context")
                )

            try:
                specification = self.specification().get("bx_catalog_specification")
                if specification is not None:
                    logger.debug(f"at entering position {callsign(self)} CATALOG")
                    self.__catalog = BxCatalogContext(specification)
                    await self.__catalog.aenter()
            except Exception as exception:
                raise RuntimeError(
                    explain(exception, f"creating {callsign(self)} catalog context")
                )

            try:
                specification = self.specification().get("bx_launcher_specifications")
                if specification is not None:
                    logger.debug(f"at entering position {callsign(self)} LAUNCHERS")
                    for launcher_specification_name in specification:
                        launcher_specification = self.specification().get(
                            launcher_specification_name
                        )
                        uuid = launcher_specification.get("uuid", None)
                        if uuid is None:
                            uuid = (
                                launcher_specification.get("type_specific_tbd", {})
                                .get("actual_bx_launcher_specification", {})
                                .get("uuid", "unknown uuid")
                            )
                        logger.debug(
                            f"at entering position {callsign(self)} LAUNCHER {uuid}"
                        )
                        launcher = BxLauncherContext(launcher_specification)
                        await launcher.aenter()
                        self.__launchers.append(launcher)
                else:
                    logger.debug(f"no specification for {callsign(self)} LAUNCHERS")
            except Exception as exception:
                raise RuntimeError(
                    explain(exception, f"creating {callsign(self)} launcher contexts")
                )

            try:
                specification = self.specification().get("bx_job_specification")
                if specification is not None:
                    logger.debug(f"at entering position {callsign(self)} JOB")
                    self.__job = BxJobContext(specification)
                    await self.__job.aenter()
            except Exception as exception:
                raise RuntimeError(
                    explain(exception, f"creating {callsign(self)} job context")
                )

            try:
                specification = self.specification().get("bx_scheduler_specification")
                if specification is not None:
                    logger.debug(f"at entering position {callsign(self)} SCHEDULER")
                    self.__scheduler = BxSchedulerContext(specification)
                    await self.__scheduler.aenter()
            except Exception as exception:
                raise RuntimeError(
                    explain(exception, f"creating {callsign(self)} scheduler context")
                )

            try:
                specification = self.specification().get("bx_collector_specification")
                if specification is not None:
                    logger.debug(f"at entering position {callsign(self)} COLLECTOR")
                    self.__collector = BxCollectorContext(specification)
                    await self.__collector.aenter()
            except Exception as exception:
                raise RuntimeError(
                    explain(exception, f"creating {callsign(self)} collector context")
                )

            try:
                specification = self.specification().get("bx_gui_specification")
                if specification is not None:
                    logger.debug(f"at entering position {callsign(self)} GUI")
                    self.__gui = BxGuiContext(specification)
                    await self.__gui.aenter()
            except Exception as exception:
                raise RuntimeError(
                    explain(exception, f"creating {callsign(self)} gui context")
                )

        except Exception as exception:
            await self.aexit()
            raise RuntimeError(explain(exception, f"entering {callsign(self)} context"))

        logger.debug(f"entered {callsign(self)} context")

    # ----------------------------------------------------------------------------------------
    async def __aexit__(self, type, value, traceback):
        """ """

        await self.aexit()

    # ----------------------------------------------------------------------------------------
    async def aexit(self):
        """ """

        logger.debug(f"exiting {callsign(self)} context")

        if self.__gui is not None:
            logger.debug(f"at exiting position {callsign(self)} GUI")
            try:
                await self.__gui.aexit()
            except Exception as exception:
                logger.error(
                    explain(exception, f"exiting {callsign(self.__gui)} context"),
                    exc_info=exception,
                )
            self.__gui = None

        if self.__collector is not None:
            logger.debug(f"at exiting position {callsign(self)} COLLECTOR")
            try:
                await self.__collector.aexit()
            except Exception as exception:
                logger.error(
                    explain(exception, f"exiting {callsign(self.__collector)} context"),
                    exc_info=exception,
                )
            self.__collector = None

        if self.__scheduler is not None:
            logger.debug(f"at exiting position {callsign(self)} SCHEDULER")
            try:
                await self.__scheduler.aexit()
            except Exception as exception:
                logger.error(
                    explain(exception, f"exiting {callsign(self.__scheduler)} context"),
                    exc_info=exception,
                )
            self.__scheduler = None

        if self.__job is not None:
            logger.debug(f"at exiting position {callsign(self)} JOB")
            try:
                await self.__job.aexit()
            except Exception as exception:
                logger.error(
                    explain(exception, f"exiting {callsign(self.__job)} context"),
                    exc_info=exception,
                )
            self.__job = None

        if len(self.__launchers) > 0:
            logger.debug(f"at exiting position {callsign(self)} LAUNCHERS")
            for index, launcher in enumerate(self.__launchers):
                logger.debug(f"at exiting position LAUNCHER {callsign(launcher)}")
                try:
                    await launcher.aexit()
                except Exception as exception:
                    logger.error(
                        explain(
                            exception, f"exiting LAUNCHER {callsign(launcher)} context"
                        ),
                        exc_info=exception,
                    )
                self.__launchers[index] = None
            self.__launchers = []

        if self.__catalog is not None:
            logger.debug(f"at exiting position {callsign(self)} CATALOG")
            try:
                await self.__catalog.aexit()
            except Exception as exception:
                logger.error(
                    explain(exception, f"exiting {callsign(self.__catalog)} context"),
                    exc_info=exception,
                )
            self.__catalog = None

        if self.__dataface is not None:
            logger.debug(f"at exiting position {callsign(self)} DATAFACE")
            try:
                await self.__dataface.aexit()
            except Exception as exception:
                logger.error(
                    explain(exception, f"exiting {callsign(self.__dataface)} context"),
                    exc_info=exception,
                )
            self.__dataface = None

        if self.__dls_servbase_dataface is not None:
            logger.debug(f"at exiting position {callsign(self)} DLS_SERVBASE DATAFACE")
            try:
                await self.__dls_servbase_dataface.aexit()
            except Exception as exception:
                logger.error(
                    explain(
                        exception,
                        f"exiting {callsign(self.__dls_servbase_dataface)} context",
                    ),
                    exc_info=exception,
                )
            self.__datafa__dls_servbase_datafacece = None

        if self.__news is not None:
            logger.debug(f"at exiting position {callsign(self)} NEWS")
            try:
                await self.__news.aexit()
            except Exception as exception:
                logger.error(
                    explain(exception, f"exiting {callsign(self.__news)} context"),
                    exc_info=exception,
                )
            self.__news = None

        if self.__logstore is not None:
            logger.debug(f"at exiting position {callsign(self)} LOGSTORE")
            try:
                await self.__logstore.aexit()
            except Exception as exception:
                logger.error(
                    explain(exception, f"exiting {callsign(self.__logstore)} context"),
                    exc_info=exception,
                )
            self.__logstore = None

        if self.__filestore is not None:
            logger.debug(f"at exiting position {callsign(self)} FILESTORE")
            try:
                await self.__filestore.aexit()
            except Exception as exception:
                logger.error(
                    explain(exception, f"exiting {callsign(self.__filestore)} context"),
                    exc_info=exception,
                )
            self.__filestore = None

        logger.debug(f"exited {callsign(self)} context")

    # ----------------------------------------------------------------------------------------
    async def add_news_consumer(self, consumer_callback):
        """ """

        if self.__news is not None:
            self.__news.add_news_consumer(consumer_callback)

            # TODO: Fix early-send problem when setting up news server/client context.
            await asyncio.sleep(0.5)
