import json
import logging

import prettytable

# Utilities.
from dls_utilpack.require import require

logger = logging.getLogger(__name__)


class PrettyHelper:
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ----------------------------------------------------------------------------------------
    def compose_bx_jobs_bx_tasks_bx_gates(self, records):
        """"""
        table = prettytable.PrettyTable()
        table.field_names = [
            "job",
            "task",
            "task state",
            "controlled gates",
            "dependency gates",
        ]

        rows = {}
        for record in records:
            if rows.get(record["bx_task_uuid"]) is None:
                rows[record["bx_task_uuid"]] = {
                    "job": record["bx_job_label"],
                    "task": record["bx_task_label"],
                    "task_state": record["bx_task_state"],
                    "controlled_bx_gates": [],
                    "dependency_bx_gates": [],
                }
            row = rows[record["bx_task_uuid"]]
            if record.get("controlled_bx_gate_label") is not None:
                bx_gate_state = "%s: %s" % (
                    record["controlled_bx_gate_label"],
                    record["controlled_bx_gate_state"],
                )
                if bx_gate_state not in row["controlled_bx_gates"]:
                    row["controlled_bx_gates"].append(bx_gate_state)
            if record.get("dependency_bx_gate_label") is not None:
                bx_gate_state = "%s: %s" % (
                    record["dependency_bx_gate_label"],
                    record["dependency_bx_gate_state"],
                )
                if bx_gate_state not in row["dependency_bx_gates"]:
                    row["dependency_bx_gates"].append(bx_gate_state)

        for _, row in rows.items():
            row2 = []
            row2.append(row["job"])
            row2.append(row["task"])
            row2.append(row["task_state"])
            row2.append("\n".join(row["controlled_bx_gates"]))
            row2.append("\n".join(row["dependency_bx_gates"]))
            table.add_row(row2)

        table.align = "l"
        table.title = "Job Details"

        return table

    # ----------------------------------------------------------------------------------------
    def compose_bx_news(self, records):
        """"""

        table = prettytable.PrettyTable()
        table.field_names = [
            "created_on",
            "job",
            "task",
            "topic",
            # TODO: Compose news details to open under click.
            # "details",
        ]

        rows = []
        for record in records:
            row = []
            topic = record["topic"]
            headline = record["headline"]
            short_topic = topic.split("::")[-1]
            short_topic = " ".join(short_topic.split("_")[1:])

            # details = json.loads(record["details"])

            # composed_details = None
            # if topic == Topics.BXGATE_WAS_OPENED:
            #     gate_label = details.get("bx_gate", {}).get("label")
            #     if gate_label is not None:
            #         # short_topic = f"{gate_label} gate was opened"
            #         short_topic = headline
            #         composed_details = ""
            # elif topic == Topics.BXGATE_WAS_CLOSED:
            #     gate_label = details.get("bx_gate", {}).get("label")
            #     if gate_label is not None:
            #         # short_topic = f"{gate_label} gate was reset to closed"
            #         short_topic = headline
            #         composed_details = ""

            # if composed_details is None:
            #     composed_details = json.dumps(details, indent=4)

            row.append(record["created_on"])
            row.append(record["job"])
            row.append(record["task"])
            row.append(headline)
            # row.append(composed_details)
            rows.append(row)

        table.add_rows(rows)
        table.align = "l"
        table.title = "News (latest at top)"

        return table

    # ----------------------------------------------------------------------------------------
    def compose_bx_jobs(self, records):
        """"""

        table = prettytable.PrettyTable()
        table.field_names = ["created_on", "label", "state"]

        rows = []
        for record in records:
            row = []
            for field_name in table.field_names:
                row.append(record[field_name])
            rows.append(row)

        table.add_rows(rows)
        table.align = "l"

        table.title = "Jobs"
        return table

    # ----------------------------------------------------------------------------------------
    def compose_bx_jobs_data_grid(
        self,
        records,
        compose_cell,
        prepend_job_labels=None,
        append_job_labels=None,
        specific_job_labels=None,
        exclude_job_labels=None,
    ):
        """"""

        if len(records) == 0:
            records = []

            # Make a fake record to get at least one line on the grid.
            seed_record = {
                "uuid": "seed_record",
                "state": "GOOD",
                "label": "",
                "data_label": "nodata",
            }
            records.append(seed_record)

        discovered_job_labels = {}
        data_rows = {}
        for record in records:
            job_label = record["label"]
            # Keep list of all extant job labels to be columns of the final table.
            if job_label not in discovered_job_labels:
                workflow_filename_classname = record.get("workflow_filename_classname")
                if workflow_filename_classname is not None:
                    discovered_job_labels[job_label] = {
                        "workflow_filename_classname": workflow_filename_classname
                    }

            data_label = record["data_label"]
            data_row = data_rows.get(data_label)
            # A new data row?
            if data_row is None:
                # Make sure the data_row is supplied with data_label,
                # so we know what to give when building a new workflow on the row.
                data_row = {"data_label": data_label}
                data_rows[data_label] = data_row

            data_cell = data_row.get(job_label)
            # New data cell on this row?
            if data_cell is None:
                data_cell = record
                data_row[job_label] = data_cell
            else:
                if record["created_on"] > data_cell["created_on"]:
                    data_cell.update(record)

        job_labels = {}
        # Caller only wants specific fields?
        if specific_job_labels is not None:
            job_labels.update(specific_job_labels)
        # Caller wants the job labels which were discovered?
        else:
            if prepend_job_labels is not None:
                job_labels.update(prepend_job_labels)
            job_labels.update(sorted(discovered_job_labels.items()))
            if append_job_labels is not None:
                job_labels.update(append_job_labels)

        # User wants to exclude some job label columns?
        # List applies to all, whether added specifically or discovered.
        if exclude_job_labels is not None:
            keep_job_labels = {}
            for field_name in job_labels.keys():
                if field_name not in exclude_job_labels:
                    keep_job_labels[field_name] = job_labels[field_name]
            job_labels = keep_job_labels

        table = prettytable.PrettyTable()
        field_names = ["data"]
        field_names.extend(job_labels.keys())
        table.field_names = field_names

        # Make the rows of the pretty table itself.
        grid_rows = []
        for data_label, data_row in data_rows.items():
            grid_row = []
            grid_row.append(data_label)
            for job_label in job_labels.keys():
                # We have data cells where an actual job has been run.
                data_cell = data_row.get(job_label)
                # No data cell means we need to use the filename_classname of the column.
                if data_cell is None:
                    workflow_filename_classname = require(
                        f"job_labels[{job_label}]",
                        job_labels[job_label],
                        "workflow_filename_classname",
                    )
                    data_cell = {
                        "workflow_filename_classname": workflow_filename_classname,
                        "data_label": data_row["data_label"],
                        "job_label": job_label,
                    }
                grid_row.append(compose_cell(data_cell))
            grid_rows.append(grid_row)

        table.add_rows(grid_rows)
        table.align = "c"

        table.title = "Workflows vs Data"

        return table

    # ----------------------------------------------------------------------------------------
    def compose_bx_variables(self, records):
        """"""

        table = prettytable.PrettyTable()
        table.field_names = ["name", "value"]

        rows = []
        for record in records:
            row = []
            for field_name in table.field_names:
                row.append(record[field_name])
            rows.append(row)

        table.add_rows(rows)
        table.align = "l"

        table.title = "Variables"
        return table

    # ----------------------------------------------------------------------------------------
    def compose_health_reports(self, health_reports):
        """"""

        table = prettytable.PrettyTable()
        table.field_names = [
            "service",
            "seconds alive",
            "request count",
            "state",
            "details",
        ]

        rows = []
        for health_report in health_reports:
            row = []
            row.append(health_report["name"])
            row.append(health_report.get("time_alive", "-"))
            row.append(health_report.get("request_count", "-"))
            row.append(health_report["state"])

            details = []
            if "exception" in health_report:
                details.append(health_report["exception"])

            if "details" in health_report:
                details.append(json.dumps(health_report["details"], indent=4))

            details = "\n\n".join(details)
            row.append(f"<xmp>{details}</xmp>")

            rows.append(row)

        table.add_rows(rows)

        table.align = "l"
        table.align["seconds alive"] = "r"
        table.align["request count"] = "r"

        table.title = "Health Reports"
        return table
