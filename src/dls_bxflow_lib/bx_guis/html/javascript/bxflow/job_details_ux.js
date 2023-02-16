// Class backing the actions ux.




class Bxflow__JobDetailsUx extends Bxflow__UxAutoUpdate {
    COOKIE_NAME = "BXFLOW_JOB_DETAILS_UX";
    GET_JOB_DETAILS = "bx_guis::commands::get_job_details";

    #jquery_objects = {};
    #bx_job_uuid = null;

    constructor(runtime, plugin_link_name, $interaction_parent) {
        super(runtime);

        this.plugin_link_name = plugin_link_name;
        this.$interaction_parent = $interaction_parent;
    }

    // -------------------------------------------------------------
    // Activate things on the UX.

    activate() {
        super.activate()

        this.#jquery_objects.$div = $(".T_composed.T_details", this.$interaction_parent);
        this.#jquery_objects.$job_summary_div = $(".T_composed.T_summary", this.$interaction_parent);
        this.#jquery_objects.$settings_container = $(".T_settings_container", this.$interaction_parent);
        this.#jquery_objects.$workflow_filename_classname = $(".T_workflow_filename_classname", this.$interaction_parent);
        this.#jquery_objects.$data_label = $(".T_data_label", this.$interaction_parent);
        this.#jquery_objects.$job_label = $(".T_job_label", this.$interaction_parent);

        this.#jquery_objects.$new_workflow_button = $(".T_new_workflow", this.$interaction_parent);
        this.#jquery_objects.$new_workflow_button.button();

        var that = this;
        this.#jquery_objects.$new_workflow_button
            .click(function (jquery_event_object) { that._handle_new_workflow_clicked(jquery_event_object); })

        // this.request_update();
    } // end method

    // -------------------------------------------------------------

    _handle_new_workflow_clicked(jquery_event_object) {
        var F = "Bxflow__JobDetailsUx::_handle_new_workflow_clicked";

        var workflow_filename_classname = this.#jquery_objects.$workflow_filename_classname.text();
        var data_label = this.#jquery_objects.$data_label.text();
        var job_label = this.#jquery_objects.$job_label.text();

        this._request_new_workflow(workflow_filename_classname, data_label, job_label);

    } // end method

    // -------------------------------------------------------------
    // Send request to index.js to switch to the job_submit_ux panel.

    _request_new_workflow(workflow_filename_classname, data_label, job_label) {
        var F = "Bxflow__JobDetailsUx::_request_new_workflow";

        console.log(F + ": requesting new workflow for \"" + workflow_filename_classname + "\", data_label \"" + data_label + "\", job_label \"" + job_label + "\"")

        // Trigger an event that the index.js will use to coordinate tab switching.
        let custom_event = new CustomEvent(Bxflow__Events_NEW_WORKFLOW_EVENT, {
            detail: {
                "workflow_filename_classname": workflow_filename_classname,
                "data_label": data_label,
                "job_label": job_label,
                "bx_job_uuid": this.#bx_job_uuid
            }
        });
        this.dispatchEvent(custom_event);

    } // end method

    // -------------------------------------------------------------
    // Request update from database.

    set_bx_job_uuid(bx_job_uuid) {
        // Clear the display areas to be updated by the new job's content.
        this.#jquery_objects.$div.html("");
        this.#jquery_objects.$job_summary_div.html("");

        this.#bx_job_uuid = bx_job_uuid;
        this.request_update()

    } // end method

    // -------------------------------------------------------------
    // Request update from database.

    request_update() {
        var json_object = {}
        json_object[this.COMMAND] = this.GET_JOB_DETAILS;
        json_object["bx_job_uuid"] = this.#bx_job_uuid;
        json_object[this.ENABLE_COOKIES] = [this.COOKIE_NAME];

        this.send(json_object);

    } // end method

    // -------------------------------------------------------------
    // Handle the response when it comes.

    handle_ajax_success(response, status, jqXHR) {
        var F = "Bxflow__JobDetailsUx::_handle_ajax_success";

        // Let the base class check for and display any error in the response.
        var error_message = super.handle_ajax_success(response, status, jqXHR);

        if (error_message !== null)
            return;

        if (response.has_been_deleted) {

            var should_revert_to_jobs_list_if_job_deleted = false;

            if (should_revert_to_jobs_list_if_job_deleted) {
                // Trigger an event that the index.js will use to coordinate tab switching.
                let custom_event = new CustomEvent(Bxflow__Events_JOB_WAS_DELETED_EVENT, {
                    detail: {}
                });

                console.log(F + ": posting custom event " + Bxflow__Events_JOB_WAS_DELETED_EVENT);
                this.dispatchEvent(custom_event);
                return;
            }

            response.html = "<div>This job has been deleted.  Please choose a different job from one of the other tabs.</div>";
        }

        // Assign the composed html to the DOM element.
        var html = response.html;
        this.#jquery_objects.$div.html(html);

        var job_summary_html = response.job_summary_html;
        if (job_summary_html) {
            this.#jquery_objects.$job_summary_div.html(job_summary_html);

            // Attach events to all the individual job links in the "recent jobs" grid.
            this.attach_job_links();
        }

        // Remember the bx_job_uuid replied to us by the server.
        console.log(F + ": response.bx_job_uuid is " + response.bx_job_uuid);
        if (response.bx_job_uuid) {
            this.#bx_job_uuid = response.bx_job_uuid
            this.#jquery_objects.$new_workflow_button.show();
        }
        else { this.#jquery_objects.$new_workflow_button.hide(); }

        // Insert the settings html composed by the server into the DOM.
        // console.log(F + ": this.#jquery_objects.$settings_container.length is " + this.#jquery_objects.$settings_container.length)
        this.#jquery_objects.$settings_container.html(response.settings_html);

        console.log(F + ": response.workflow_filename_classname is " + response.workflow_filename_classname);

        // Handle the two tidbits that allow us to request a new workflow.
        if (response.workflow_filename_classname) {
            this.#jquery_objects.$workflow_filename_classname.text(response.workflow_filename_classname)
        }
        else {
            this.#jquery_objects.$workflow_filename_classname.text("-")
        }

        if (response.data_label) {
            this.#jquery_objects.$data_label.text(response.data_label)
        }
        else {
            this.#jquery_objects.$data_label.text("")
        }

        if (response.job_label) {
            this.#jquery_objects.$job_label.text(response.job_label)
        }
        else {
            this.#jquery_objects.$job_label.text("")
        }

    }
}