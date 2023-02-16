class Index extends Bxflow__Page {
    #tabs_manager = null;
    #job_news_ux = null;
    #recent_jobs_ux = null;
    #job_details_ux = null;
    #job_variables_ux = null;
    #job_submit_ux = null;
    #job_data_grid_ux = null;
    #system_health_ux = null;

    #tab_id_last_opened = null;

    constructor(runtime) {
        super(runtime);
    } // end constructor


    // -------------------------------------------------------------------------------
    _stretch_height(thing, height) {
        var $thing = $(thing)
        var available = height - $thing.offset().top;
        $thing.innerHeight(available);
        return thing +
            " from top " + $thing.offset().top.toFixed(0) +
            " to height " + height.toFixed(0) +
            " has " + available.toFixed(0) + " available\n"
    }
    // -------------------------------------------------------------------------------
    _sizewatch_thing(thing) {
        return thing + " top " + $(thing).offset().top.toFixed(0) + ", height " + $(thing).innerHeight().toFixed(0) + "\n";
    }

    // -------------------------------------------------------------------------------
    _sizewatch() {
        var text = "";

        var window_height = $("BODY").innerHeight();
        text += this._stretch_height("#tabs", window_height - 2);

        text += this._sizewatch_thing("BODY");
        text += this._sizewatch_thing("#tabs");
        $("#sizewatch").text(text);

        // console.log(text);
    }

    // -------------------------------------------------------------------------------
    // Called after page is loaded and all DOM elements are available.
    activate() {
        super.activate();

        var that = this;

        // -------------------------------------------------------------------

        this.#tabs_manager = new Bxflow__TabsManager(
            self.runtime,
            "tabs_manager",
            $("#tabs_manager_interaction_parent"));


        // -------------------------------------------------------------------

        this.#recent_jobs_ux = new Bxflow__RecentJobsUx(
            self.runtime,
            "recent_jobs",
            $("#recent_jobs_ux_interaction_parent"));


        // -------------------------------------------------------------------

        this.#job_details_ux = new Bxflow__JobDetailsUx(
            self.runtime,
            "job_details",
            $("#job_details_ux_interaction_parent"));


        // -------------------------------------------------------------------

        this.#job_news_ux = new Bxflow__JobNewsUx(
            self.runtime,
            "job_news",
            $("#job_news_ux_interaction_parent"));


        // -------------------------------------------------------------------

        this.#job_variables_ux = new Bxflow__JobVariablesUx(
            self.runtime,
            "job_variables",
            $("#job_variables_ux_interaction_parent"));


        // -------------------------------------------------------------------

        this.#job_submit_ux = new Bxflow__JobSubmitUx(
            self.runtime,
            "job_submit",
            $("#job_submit_ux_interaction_parent"));


        // -------------------------------------------------------------------

        this.#job_data_grid_ux = new Bxflow__JobDataGridUx(
            self.runtime,
            "job_data_grid",
            $("#job_data_grid_ux_interaction_parent"));

        // -------------------------------------------------------------------

        this.#system_health_ux = new Bxflow__SystemHealthUx(
            self.runtime,
            "system_health",
            $("#system_health_ux_interaction_parent"));
        this.#system_health_ux.set_auto_update_sleep_ms(5000);


        // -------------------------------------------------------------------

        var that = this;

        // Tabs have created, which may need some tweaking inside the tab.
        this.#tabs_manager.addEventListener(
            Bxflow__Events_TABS_CREATED_EVENT,
            function (event) { that.handle_tabs_created(event); });

        // User picks a job from the list of recent jobs.
        this.#recent_jobs_ux.addEventListener(
            Bxflow__Events_DETAIL_JOB_EVENT,
            function (event) { that.handle_job_detail_request(event); });

        // User picks a job from a column of the job data grid.
        this.#job_data_grid_ux.addEventListener(
            Bxflow__Events_DETAIL_JOB_EVENT,
            function (event) { that.handle_job_detail_request(event); });

        // User has started a new job.
        this.#job_submit_ux.addEventListener(
            Bxflow__Events_DETAIL_JOB_EVENT,
            function (event) { that.handle_job_detail_request(event); });

        // User wants to go to the jobs list.
        this.#job_details_ux.addEventListener(
            Bxflow__Events_JOB_WAS_DELETED_EVENT,
            function (event) { that.handle_job_was_deleted_request(event); });

        // User wants to start a new workflow.
        this.#job_details_ux.addEventListener(
            Bxflow__Events_NEW_WORKFLOW_EVENT,
            function (event) { that.handle_new_workflow_request(event); });
        this.#job_data_grid_ux.addEventListener(
            Bxflow__Events_NEW_WORKFLOW_EVENT,
            function (event) { that.handle_new_workflow_request(event); });

        // Auto update has happened on the job details panel.
        this.#job_details_ux.addEventListener(
            Bxflow__UxAutoUpdate__AUTO_UPDATE_EVENT,
            function (event) { that.handle_job_details_auto_update(event); });

        // -------------------------------------------------------------------

        this.#tabs_manager.activate()
        this.#recent_jobs_ux.activate()
        this.#job_details_ux.activate()
        this.#job_news_ux.activate()
        this.#job_variables_ux.activate()
        this.#job_data_grid_ux.activate()
        this.#job_submit_ux.activate()
        this.#system_health_ux.activate()

        // Tab has been opened (made current).
        this.#tabs_manager.addEventListener(
            Bxflow__Events_TAB_OPENED,
            function (event) { that.handle_tab_opened(event); });

        // -------------------------------------------------------------------
        $(window).resize(function (jquery_event_object) { that._sizewatch(); });

        // // TODO: Remove index.js sizewatch debug later.
        // setTimeout(function () { that._sizewatch(); }, 1000);

    } // end method

    // -----------------------------------------------------------------------
    // Propagate event where user clicks a particular job.
    handle_tabs_created(event) {
        var F = "Index::handle_tabs_created";
        console.log(F + ": tabs created");
        this._sizewatch();
    } // end method

    // -----------------------------------------------------------------------
    // Propagate event where user clicks a particular job.
    handle_tab_opened(event) {
        var F = "Index::handle_tab_opened";

        // Have to hide the top-level error or the sizewatch doesn't compute right.
        this.#tabs_manager.display_ajax_error(null);
        // Adjust the sizing for the new tab.
        // TODO: Move sizewatch inside particular ux that needs it.
        this._sizewatch();

        var tab_id = event.detail.tab_id;

        if (tab_id == this.#tab_id_last_opened)
            return;

        this.#tab_id_last_opened = tab_id;

        console.log(F + ": tab_id is \"" + tab_id + "\" dispatched by " + event.detail.dispatched_by);

        var $interaction_parent = $("#" + tab_id).children().first();

        var interaction_parent_id = $interaction_parent.attr("id");

        console.log(F + ": $interaction_parent_id is \"" + interaction_parent_id + "\"");

        if (interaction_parent_id == "recent_jobs_ux_interaction_parent") {
            this.#recent_jobs_ux.request_update()
        }
        else if (interaction_parent_id == "job_details_ux_interaction_parent") {
            this.#job_details_ux.request_update()
        }
        else if (interaction_parent_id == "job_news_ux_interaction_parent") {
            this.#job_news_ux.request_update()
        }
        else if (interaction_parent_id == "job_data_grid_ux_interaction_parent") {
            this.#job_data_grid_ux.request_update()
        }
        else if (interaction_parent_id == "job_submit_ux_interaction_parent") {
            this.#job_submit_ux.request_update()
        }
        else if (interaction_parent_id == "system_health_ux_interaction_parent") {
            this.#system_health_ux.request_update()
        }

    } // end method



    // -----------------------------------------------------------------------
    // Handle event where user wants to switch to recent jobs list.
    handle_job_was_deleted_request(event) {
        var F = "Index::handle_job_was_deleted_request";

        var tab_id = "tab-jobs-list";
        console.log(F + ": switching to " + tab_id);

        this.#tabs_manager.switch_to_tab(tab_id)
    } // end method

    // -----------------------------------------------------------------------
    // Propagate event where user clicks a particular job.
    handle_job_detail_request(event) {
        var bx_job_uuid = event.detail.bx_job_uuid;

        this.#job_details_ux.set_bx_job_uuid(bx_job_uuid);
        this.#job_news_ux.set_bx_job_uuid(bx_job_uuid);
        this.#job_variables_ux.set_bx_job_uuid(bx_job_uuid);

        this.#tabs_manager.switch_to_tab("tab-job-details")
    } // end method

    // -----------------------------------------------------------------------
    // Propagate event where user clicks to create a new workflow.
    handle_new_workflow_request(event) {
        var F = "Index::handle_tab_opened";

        var workflow_filename_classname = event.detail.workflow_filename_classname;
        var data_label = event.detail.data_label;
        var job_label = event.detail.job_label;
        var bx_job_uuid = event.detail.bx_job_uuid;

        console.log(F + ": handling request for new workflow for" +
            " \"" + workflow_filename_classname + "\"," +
            " data label \"" + data_label + "\"," +
            " job label \"" + job_label + "\", " +
            " bx_job_uuid \"" + bx_job_uuid + "\"");

        // Tell the job_submit_ux panel to prepare input fields for this workflow.
        this.#job_submit_ux.show_workflow_settings(
            workflow_filename_classname,
            data_label,
            job_label,
            bx_job_uuid);

        // Switch to show the panel.
        this.#tabs_manager.switch_to_tab("tab-job-submit")
    } // end method

    // -----------------------------------------------------------------------
    // Propagate event where the job details has auto updated.
    handle_job_details_auto_update(event) {

        this.#job_news_ux.request_update();
        this.#job_variables_ux.request_update();

    } // end method


} // end class

// -------------------------------------------------------------
// All elements on the page ready.
$(document).ready(function () {
    // Make an object to handle the page logic.
    var page = new Index(global_runtime);
    page.activate();
});

