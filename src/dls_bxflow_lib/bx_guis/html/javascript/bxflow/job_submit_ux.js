// Class backing the actions ux.


class Bxflow__JobSubmitUx extends Bxflow__UxBase {
    COOKIE_NAME = "BXFLOW_JOB_SUBMIT_UX";
    SHOW_WORKFLOW_SETTINGS = "bx_guis::commands::show_workflow_settings";
    START_WORKFLOW = "bx_guis::commands::start_workflow";

    #jquery_objects = {};
    #workflow_filename_classname = null;
    #data_label = null;
    #job_label = null;
    #bx_job_uuid = null;

    constructor(runtime, plugin_link_name, $interaction_parent) {
        super(runtime);

        this.plugin_link_name = plugin_link_name;
        this.$interaction_parent = $interaction_parent;
    }

    // -------------------------------------------------------------
    // Activate things on the UX.

    activate() {
        var F = "Bxflow__JobSubmitUx::activate";

        super.activate();

        this.#jquery_objects.$contents = $(".T_contents", this.$interaction_parent);
        this.#jquery_objects.$initial = $(".T_initial", this.$interaction_parent);
        this.#jquery_objects.$form = $("FORM", this.$interaction_parent);
        this.#jquery_objects.$workflow_filename_classname = $(".T_workflow_filename_classname", this.$interaction_parent);
        this.#jquery_objects.$data_label = $(".T_data_label", this.$interaction_parent);
        this.#jquery_objects.$settings_container = $(".T_settings_container", this.$interaction_parent);

        this.#jquery_objects.$start_button = $(".T_start", this.$interaction_parent);
        this.#jquery_objects.$start_button.button();

        this.#jquery_objects.$refresh_settings_button = $(".T_refresh_settings", this.$interaction_parent);
        this.#jquery_objects.$refresh_settings_button.button();

        var that = this;
        this.#jquery_objects.$start_button
            .click(function (jquery_event_object) { that._handle_start_clicked(jquery_event_object); })

        this.#jquery_objects.$refresh_settings_button
            .click(function (jquery_event_object) { that._handle_refresh_settings_clicked(jquery_event_object); })

        // Initially hide the panel contents until called with workfow to start from.
        this.#jquery_objects.$contents.hide();
        this.#jquery_objects.$initial.show();

        // Kick off a content request, get initial arguments from the cookie.
        // this.show_workflow_settings(this.#workflow_filename_classname, this.#data_label);

    } // end method

    // -------------------------------------------------------------

    _handle_start_clicked(jquery_event_object) {
        var F = "Bxflow__JobSubmitUx::_handle_start_clicked";

        var json_object = {}
        json_object[this.COMMAND] = this.START_WORKFLOW;
        json_object[this.ENABLE_COOKIES] = [this.COOKIE_NAME];
        json_object["workflow_filename_classname"] = this.#workflow_filename_classname;
        json_object["data_label"] = this.#data_label;

        // Build a payload to pick up all the values in the form.
        var payload = {};
        var elements = this.#jquery_objects.$form.get(0).elements;
        for (var i = 0; i < elements.length; i++) {
            var name = elements[i].name;
            var $element = $(elements[i]);
            console.log(F + ": payload[" + name + "] is " + $element.val());
            payload[name] = $element.val();
        }
        json_object[this.PAYLOAD] = payload;

        this.send(json_object);

    } // end method

    // -------------------------------------------------------------

    _handle_refresh_settings_clicked(jquery_event_object) {
        var F = "Bxflow__JobSubmitUx::_handle_refresh_settings_clicked";

        var json_object = {}
        json_object[this.COMMAND] = this.SHOW_WORKFLOW_SETTINGS;
        json_object[this.ENABLE_COOKIES] = [this.COOKIE_NAME];
        json_object["workflow_filename_classname"] = this.#workflow_filename_classname;
        json_object["data_label"] = this.#data_label;
        json_object["job_label"] = this.#job_label;
        json_object["bx_job_uuid"] = this.#bx_job_uuid;

        // Build a payload to pick up all the values in the form.
        var elements = this.#jquery_objects.$form.get(0).elements;
        for (var i = 0; i < elements.length; i++) {
            var name = elements[i].name;
            var $element = $(elements[i]);
            json_object[name] = $element.val();
        }

        var notebook = json_object["notebook"];
        if (notebook !== undefined) {
            json_object["job_label"] = notebook;
        }

        this.send(json_object);

    } // end method

    // -------------------------------------------------------------
    // Request update from database.

    request_update() {

    } // end method

    // -------------------------------------------------------------
    // Called from index.js to prepare input fields for this workflow.
    show_workflow_settings(
        workflow_filename_classname,
        data_label,
        job_label,
        bx_job_uuid) {
        var F = "Bxflow__JobSubmitUx::show_workflow_settings";

        console.log(F + ": requesting settings for new workflow for" +
            " \"" + workflow_filename_classname + "\"," +
            " data label \"" + data_label + "\"," +
            " job label \"" + job_label + "\", " +
            " bx_job_uuid \"" + bx_job_uuid + "\"");

        // Remember the values which have been requested.
        this.#workflow_filename_classname = workflow_filename_classname;
        this.#data_label = data_label;
        this.#job_label = job_label;
        this.#bx_job_uuid = bx_job_uuid;

        var json_object = {}
        json_object[this.COMMAND] = this.SHOW_WORKFLOW_SETTINGS;
        json_object[this.ENABLE_COOKIES] = [this.COOKIE_NAME];
        json_object["workflow_filename_classname"] = workflow_filename_classname
        json_object["data_label"] = data_label
        json_object["job_label"] = job_label
        json_object["bx_job_uuid"] = bx_job_uuid

        this.#jquery_objects.$workflow_filename_classname.html("-")
        this.#jquery_objects.$settings_container.html("-")

        this.send(json_object);

    } // end method

    // -------------------------------------------------------------
    // Handle the response when it comes.

    handle_ajax_success(response, status, jqXHR) {
        var F = "Bxflow__JobSubmitUx::_handle_ajax_success";

        // Show the panel contents (which may have been initially hidden).
        this.#jquery_objects.$contents.show();
        this.#jquery_objects.$initial.hide();

        // Let the base class check for and display any error in the response.
        var error_message = super.handle_ajax_success(response, status, jqXHR);

        if (error_message !== null)
            return;

        // Check to see if a job got launched.
        var bx_job_uuid = response.bx_job_uuid;

        // A job has been submitted?
        if (bx_job_uuid !== undefined) {
            // Trigger an event that the index.js will use to coordinate tab switching.
            var custom_event = new CustomEvent(Bxflow__Events_DETAIL_JOB_EVENT,
                {
                    detail: { bx_job_uuid: bx_job_uuid }
                });
            console.log(F + ": dispatching " + Bxflow__Events_DETAIL_JOB_EVENT + " for " + bx_job_uuid)
            this.dispatchEvent(custom_event);
        }
        else {
            // Remember the values which may have been pulled from the cookie,
            // or otherwise adjusted by the server.
            if (response.workflow_filename_classname !== undefined) {
                this.#workflow_filename_classname = response.workflow_filename_classname;
                this.#jquery_objects.$workflow_filename_classname.html(this.#workflow_filename_classname)
            }
            if (response.data_label !== undefined) {
                this.#data_label = response.data_label;
                this.#jquery_objects.$data_label.html(this.#data_label)
                this.#jquery_objects.$settings_container.html(response.settings_html);
            }
            if (response.settings_html !== undefined) {
                this.#jquery_objects.$settings_container.html(response.settings_html);
            }
        }
    }

}
