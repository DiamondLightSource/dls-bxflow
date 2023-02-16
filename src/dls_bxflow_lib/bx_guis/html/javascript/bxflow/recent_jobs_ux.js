// Class backing the actions ux.

class Bxflow__RecentJobsUx extends Bxflow__UxAutoUpdate {
    COOKIE_NAME = "BXFLOW_RECENT_JOBS_UX";
    GET_RECENT_JOBS = "bx_guis::commands::get_recent_jobs";

    #jquery_objects = {};

    constructor(runtime, plugin_link_name, $interaction_parent) {
        super(runtime);

        this.plugin_link_name = plugin_link_name;
        this.$interaction_parent = $interaction_parent;
    }

    // -------------------------------------------------------------
    // Activate things on the UX.

    activate() {
        super.activate()

        this.#jquery_objects.$div = $(".T_composed", this.$interaction_parent);

        // this.request_update();

    } // end method

    // -------------------------------------------------------------
    // Request update from database.

    request_update() {

        var json_object = {}
        json_object[this.COMMAND] = this.GET_RECENT_JOBS;
        json_object[this.ENABLE_COOKIES] = [this.COOKIE_NAME];

        this.send(json_object);

    } // end method

    // -------------------------------------------------------------
    // Handle the response when it comes.

    handle_ajax_success(response, status, jqXHR) {
        var F = "Bxflow__RecentJobsUx::_handle_ajax_success";

        // Let the base class have a look at the response.
        super.handle_ajax_success(response, status, jqXHR);

        var html = response.html;

        if (html !== undefined) {
            this.#jquery_objects.$div.html(html);
            // Attach events to all the individual job links in the "recent jobs" grid.
            this.attach_job_links();
        }

    }

    // -------------------------------------------------------------

    _handle_detail_job_clicked(jquery_event_object) {

        var $detail_job = $(jquery_event_object.target);

        this._request_detail_job($detail_job);

        this.set_and_render_auto_update(false);

    } // end method

    // -------------------------------------------------------------

    _request_detail_job($detail_job) {
        var F = "Bxflow__RecentJobsUx::_pick";

        console.log(F + ": updating " + $detail_job.attr("bx_job_uuid"))

        this.$detail_job.removeClass("T_picked");
        $detail_job.addClass("T_picked");

        // Trigger an event that the index.js will use to coordinate tab switching.
        var bx_job_uuid = $detail_job.attr("bx_job_uuid");
        var custom_event = new CustomEvent(Bxflow__Events_DETAIL_JOB_EVENT,
            {
                detail: { bx_job_uuid: bx_job_uuid }
            });
        this.dispatchEvent(custom_event);

    } // end method


}
