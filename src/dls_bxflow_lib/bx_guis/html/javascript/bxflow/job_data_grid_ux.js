// Class backing the actions ux.


class Bxflow__JobDataGridUx extends Bxflow__UxAutoUpdate {
    COOKIE_NAME = "BXFLOW_JOB_DATA_GRID_UX";
    GET_JOB_DATA_GRID = "bx_guis::commands::get_job_data_grid";

    #jquery_objects = {};
    #$cell_clicked = undefined;

    constructor(runtime, plugin_link_name, $interaction_parent) {
        super(runtime);

        this.plugin_link_name = plugin_link_name;
        this.$interaction_parent = $interaction_parent;
    }

    // -------------------------------------------------------------
    // Activate things on the UX.

    activate() {
        var F = "Bxflow__JobDataGridUx::activate";

        super.activate()

        this.#jquery_objects.$div = $(".T_composed", this.$interaction_parent);

        // this.request_update();

    } // end method

    // -------------------------------------------------------------
    // Request update from database.

    request_update() {

        var json_object = {}
        json_object[this.COMMAND] = this.GET_JOB_DATA_GRID;
        json_object[this.ENABLE_COOKIES] = [this.COOKIE_NAME];

        this.send(json_object);

    } // end method

    // -------------------------------------------------------------
    // Handle the response when it comes.

    handle_ajax_success(response, status, jqXHR) {
        var F = "Bxflow__JobDataGridUx::_handle_ajax_success";

        // Let the base class have a look at the response.
        super.handle_ajax_success(response, status, jqXHR);

        var html = response.html;
        this.#jquery_objects.$div.html(html);

        // Attach events to all the individual job links in the grid.
        this.attach_job_links();
    }

    // -------------------------------------------------------------
    // Attach events to all the individual job links in the grid.

    attach_job_links(response, status, jqXHR) {
        var F = "Bxflow__JobDataGridUx::attach_job_links";

        var that = this;

        this.$cells = $(".T_cell", this.$interaction_parent);
        this.$cells.click(function (jquery_event_object) { that._handle_cell_clicked(jquery_event_object); })

    }

    // -------------------------------------------------------------

    _handle_cell_clicked(jquery_event_object) {
        var F = "Bxflow__JobDataGridUx::_handle_cell_clicked";

        var $cell = $(jquery_event_object.target);

        if ($cell.attr("bx_job_uuid") === undefined &&
            $cell.attr("workflow_filename_classname") === undefined) {
            $cell = $cell.parent();
        }

        console.log(F + ": clicked $cell " + $cell.attr("class") + ", updating " + $cell.attr("bx_job_uuid"))

        this.#$cell_clicked = $cell;

        this.$cells.parent().removeClass("T_picked");
        this.#$cell_clicked.parent().addClass("T_picked");

        if ($cell.attr("bx_job_uuid") !== undefined) {
            this._request_job_detail($cell);
        }

        if ($cell.attr("workflow_filename_classname") !== undefined) {
            var workflow_filename_classname = $cell.attr("workflow_filename_classname")
            var data_label = $cell.attr("data_label");
            var job_label = $cell.attr("job_label");
            this._request_new_workflow(workflow_filename_classname, data_label, job_label);
        }

    } // end method

    // -------------------------------------------------------------

    _request_job_detail($cell) {
        var F = "Bxflow__JobDataGridUx::_request_job_detail";

        // Trigger an event that the index.js will use to coordinate tab switching.
        var bx_job_uuid = $cell.attr("bx_job_uuid");
        var custom_event = new CustomEvent(Bxflow__Events_DETAIL_JOB_EVENT,
            {
                detail: { bx_job_uuid: bx_job_uuid }
            });
        this.dispatchEvent(custom_event);

    } // end method

    // -------------------------------------------------------------
    // Send request to index.js to switch to the job_submit_ux panel.

    _request_new_workflow(workflow_filename_classname, data_label, job_label) {
        var F = "Bxflow__JobDataGridUx::_request_new_workflow";

        console.log("requesting new workflow for \"" + workflow_filename_classname + "\", \"" + data_label + "\"")

        // Trigger an event that the index.js will use to coordinate tab switching.
        let custom_event = new CustomEvent(Bxflow__Events_NEW_WORKFLOW_EVENT, {
            detail: {
                "workflow_filename_classname": workflow_filename_classname,
                "data_label": data_label,
                "job_label": job_label
            }
        });
        this.dispatchEvent(custom_event);

    } // end method
}