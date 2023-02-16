// Class backing the actions ux.

class Bxflow__JobNewsUx extends Bxflow__UxBase {
    COOKIE_NAME = "BXFLOW_JOB_NEWS_UX";
    GET_JOB_NEWS = "bx_guis::commands::get_job_news";

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

        this.#jquery_objects.$div = $(".T_composed", this.$interaction_parent);

        // this.request_update();
    } // end method

    // -------------------------------------------------------------
    // Request update from database.

    set_bx_job_uuid(bx_job_uuid) {
        // Clear the display areas to be updated by the new job's content.
        this.#jquery_objects.$div.html("waiting for update on job " + bx_job_uuid + "...");

        this.#bx_job_uuid = bx_job_uuid;

        this.request_update()

    } // end method

    // -------------------------------------------------------------
    // Request update from database.

    request_update() {
        var json_object = {}
        json_object[this.COMMAND] = this.GET_JOB_NEWS;
        json_object["bx_job_uuid"] = this.#bx_job_uuid;
        json_object[this.ENABLE_COOKIES] = [this.COOKIE_NAME];

        this.send(json_object);

    } // end method

    // -------------------------------------------------------------
    // Handle the response when it comes.

    handle_ajax_success(response, status, jqXHR) {

        // Let the base class have a look at the response.
        super.handle_ajax_success(response, status, jqXHR);

        var html = response.html;
        this.#jquery_objects.$div.html(html);

        // Remember the bx_job_uuid replied to us by the server.
        if (response.bx_job_uuid !== undefined) {
            this.#bx_job_uuid = response.bx_job_uuid
        }

        // this.render();
    }



}
