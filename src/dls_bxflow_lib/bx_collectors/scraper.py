import asyncio
import copy
import glob
import logging
import os
import time
from typing import Dict

from dls_utilpack.callsign import callsign
from dls_utilpack.describe import describe
from dls_utilpack.explain import explain2

# Utilities.
from dls_utilpack.require import require

# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Base class for bx_collector instances.
from dls_bxflow_lib.bx_collectors.base import Base as BxCollectorBase

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_collectors.popener"


# ------------------------------------------------------------------------------------------
class Scraper(BxCollectorBase):
    """
    Object representing a bx_collector which launches a task using popen for onboard execution.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification, predefined_uuid=None):
        BxCollectorBase.__init__(
            self, thing_type, specification, predefined_uuid=predefined_uuid
        )

        s = f"{callsign(self)} specification", self.specification()

        type_specific_tbd = require(s, self.specification(), "type_specific_tbd")
        self.__scrape_glob = require(s, type_specific_tbd, "scrape_glob")
        self.__scrape_recursive = require(s, type_specific_tbd, "scrape_recursive")

        self.workflow_filename_classname = require(
            s, type_specific_tbd, "workflow_filename_classname"
        )

        # Just the classname part, for finding the job.
        self.workflow_classname = self.workflow_filename_classname.split("::")[-1]

        self.workflow_constructor_kwargs = require(
            s, type_specific_tbd, "workflow_constructor_kwargs"
        )

        # We will use the dataface to discover previously processed files.
        self.__bx_dataface = bx_datafaces_get_default()

        # This flag will stop the ticking async task.
        self.__keep_ticking = True
        self.__tick_future = None

    # ----------------------------------------------------------------------------------------
    async def activate(self):
        """"""

        # Get all the jobs ever done.
        records = await self.__bx_dataface.get_bx_jobs(labels=[self.workflow_classname])

        # Make an initial list of the data labels associated with any job.
        self.__data_labels = []
        for record in records:
            data_label = record["data_label"]
            if data_label not in self.__data_labels:
                self.__data_labels.append(data_label)

        logger.debug(f"activating with {len(records)} existing job records")

        # Poll periodically.
        self.__tick_future = asyncio.get_event_loop().create_task(self.tick())

    # ----------------------------------------------------------------------------------------
    async def job_was_deleted(self, news_payload):
        """"""

        data_label = news_payload.get("bx_job", {}).get("data_label")

        if data_label is not None:
            try:
                self.__data_labels.remove(data_label)
                logger.debug(f"removing deleted job's data_label {data_label}")
            except ValueError:
                logger.debug(f"could not remove deleted job's data_label {data_label}")
        else:
            logger.debug(describe("no bx_job.data_label in news payload", news_payload))

    # ----------------------------------------------------------------------------------------
    async def deactivate(self):
        """"""

        if self.__tick_future is not None:
            # Set flag to stop the periodic ticking.
            self.__keep_ticking = False
            # Wait for the ticking to stop.
            await self.__tick_future

    # ----------------------------------------------------------------------------------------
    async def tick(self):
        """
        Periodic ticking to check for new work.
        """

        while self.__keep_ticking:
            try:
                await self.scrape()
            except Exception as exception:
                logger.error(explain2(exception, "scraping"), exc_info=exception)
            await asyncio.sleep(1.0)

    # ----------------------------------------------------------------------------------------
    async def scrape(self):
        """
        Scrape the disk looking for new files.
        """

        t0 = time.time()
        data_filenames = glob.glob(
            self.__scrape_glob, recursive=self.__scrape_recursive
        )
        t1 = time.time()

        seconds = "%0.3f" % (t1 - t0)
        logger.debug(
            f"glob {self.__scrape_glob} found {len(data_filenames)} files in {seconds} seconds"
        )

        for data_filename in data_filenames:
            data_label = self.derive_data_label(data_filename)

            if data_label not in self.__data_labels:
                await self.trigger_workflow_for_filename(data_filename)
                self.__data_labels.append(data_label)

    # ----------------------------------------------------------------------------------------
    async def trigger_workflow_for_filename(self, data_filename):
        """
        Submit workflow for the file, if appropriate.
        """

        # The task specification may give some workflow constructor args.
        workflow_constructor_kwargs = copy.deepcopy(self.workflow_constructor_kwargs)

        logger.info(
            f"triggering workflow {self.workflow_filename_classname}"
            f" for data_filename {data_filename}"
        )

        # We dynamically provide the workflow with the data_label.
        data_label = self.derive_data_label(data_filename)
        workflow_constructor_kwargs["data_label"] = data_label
        await self.trigger(
            self.workflow_filename_classname,
            **workflow_constructor_kwargs,
        )

        logger.info("triggering complete")

    # ----------------------------------------------------------------------------------------
    async def fire(self, message: Dict):
        """
        TBD: Deal with bx_collectors.scraper.fire method.
        """

        # If message doesn't contain a filename, then scrape for it.
        if message is None:
            await self.scrape()

    # ----------------------------------------------------------------------------------------
    def derive_data_label(self, data_filename):
        """Scan number derived from specification data_filename."""

        # Get just the scan number as wanted for substitution as data_label.
        try:
            data_label = os.path.basename(data_filename)
            data_label = os.path.splitext(data_label)[0]
            data_label = data_label.split("-")[1]
        except Exception:
            raise ValueError(
                f"data_label pattern bbb-NNNNNN.nxs not matched in data_filename {data_filename}"
            )

        return data_label
