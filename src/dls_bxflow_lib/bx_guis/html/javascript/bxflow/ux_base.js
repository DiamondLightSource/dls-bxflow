// Class backing the actions ux.


class Bxflow__UxBase extends DlsCommon__Base {
    COMMAND = "bx_guis::keywords::command";
    PAYLOAD = "bx_guis::keywords::payload";
    ENABLE_COOKIES = "dls_servbase_api::keywords::enable_cookies";
    UPDATE_JOB = "bx_guis::commands::update_job";
    CANCEL_JOB = "bx_guis::commands::cancel_job";
    DELETE_JOB = "bx_guis::commands::delete_job";
    UNBLOCK_JOB = "bx_guis::commands::unblock_job";
    JOB_COMMENT = "comment"
    JOB_RATING = "rating"

    #jquery_objects = {};

    constructor(runtime, plugin_link_name, $interaction_parent) {
        super(runtime);

        this.plugin_link_name = plugin_link_name;
        this.$interaction_parent = $interaction_parent;
    }

    // -------------------------------------------------------------
    // Activate things on the UX.

    activate() {

        var em_class = "T_ajax_error_message";
        var $em_container = $("." + em_class + "_container", this.$interaction_parent);
        var $em = $("." + em_class, this.$interaction_parent);

        if ($em_container.length == 0) {
            this.$interaction_parent.prepend("<div class='" + em_class + "_container'><xmp class='" + em_class + "'></xmp></div>");
            $em_container = $("." + em_class + "_container", this.$interaction_parent);
            $em = $("." + em_class, this.$interaction_parent);
        }

        $em_container.hide();
        this.#jquery_objects.$error_message_container = $em_container;
        this.#jquery_objects.$error_message = $em;

    } // end method

    // -------------------------------------------------------------
    // Handle the response when it comes.

    handle_ajax_success(response, status, jqXHR) {
        var F = "Bxflow__UxBase::handle_ajax_success[" + this.plugin_link_name + "]";

        var error_message = null;
        var http_code = jqXHR.status;

        // If not 200, then we don't likely have response as json.
        if (http_code !== 200) {
            error_message = "error " + http_code + " (" + status + ")" + "\n" + response;
        }
        // Presumably we got a cogent json response, so check if the content has an error field.
        else {
            if (response.exception || response.error) {
                error_message = "exception in server response\n" + JSON.stringify(response);
            }
        }

        this.display_ajax_error(error_message);

        return error_message;
    }

    // -------------------------------------------------------------
    // Handle the response failure if it comes.
    handle_ajax_failure(jqXHR, status, error_thrown) {
        var F = "Bxflow__UxBase::handle_ajax_failure[" + this.plugin_link_name + "]";


        var http_code = jqXHR.status;
        var error_message = null;

        if (error_thrown.name != undefined)
            error_message = error_thrown.name + ": " + error_thrown.message;
        else
            error_message = error_thrown;

        error_message = "error " + http_code + " " + error_message + "\n" + jqXHR.responseText;

        console.log(F + ": " + error_message);

        this.display_ajax_error(error_message);

        return error_message;
    }

    // -------------------------------------------------------------
    // Display any error from ajax response.
    display_ajax_error(error_message) {
        var F = "Bxflow__UxBase::display_ajax_error[" + this.plugin_link_name + "]";

        if (error_message === null) {
            this.#jquery_objects.$error_message_container.hide();
        }
        else {
            console.log(F + ": displaying " + error_message);

            this.#jquery_objects.$error_message.text(error_message);
            this.#jquery_objects.$error_message_container.show();
        }
    }

    // -------------------------------------------------------------
    send(json_object) {
        var F = "Bxflow__RecentJobsUx::send[" + this.plugin_link_name + "]";

        var json_string = JSON.stringify(json_object);

        // console.log(F + ": sending " + json_string);

        var url = window.location.protocol +
            "//" + window.location.hostname +
            ":" + window.location.port +
            "/protocolj";

        var that = this;

        var $request = $.ajax(
            {
                url: url,
                cache: false,
                data: json_string,
                method: "POST",
                processData: false,
                contentType: "application/json",
                success: function (response, status, jqXHR) { that.handle_ajax_success(response, status, jqXHR); },
                error: function (jqXHR, status, error_thrown) { that.handle_ajax_failure(jqXHR, status, error_thrown); }
            }
        );

    } // end method

    // -------------------------------------------------------------
    // Attach events to all the individual job links in the grid.

    attach_job_links(response, status, jqXHR) {
        var F = "Bxflow__RecentJobsUx::attach_job_links[" + this.plugin_link_name + "]";

        var that = this;

        this.$detail_job = $(".T_bx_job_detail_job", this.$interaction_parent);
        this.$detail_job.click(function (jquery_event_object) { that._handle_detail_job_clicked(jquery_event_object); })

        var $cancel_jobs = $(".T_bx_job_cancel_job", this.$interaction_parent);
        $cancel_jobs.click(function (jquery_event_object) { that._handle_cancel_job_clicked(jquery_event_object); })

        var $delete_jobs = $(".T_bx_job_delete_job", this.$interaction_parent);
        $delete_jobs.click(function (jquery_event_object) { that._handle_delete_job_clicked(jquery_event_object); })

        var $unblock_jobs = $(".T_bx_job_unblock_job", this.$interaction_parent);
        $unblock_jobs.click(function (jquery_event_object) { that._handle_unblock_job_clicked(jquery_event_object); })

        var $job_comments = $("TEXTAREA.T_job_comment", this.$interaction_parent);
        $job_comments.change(function (jquery_event_object) { that._handle_job_comment_changed(jquery_event_object); })

        var $job_ratings = $(".T_job_ratings", this.$interaction_parent).children();
        $job_ratings.click(function (jquery_event_object) { that._handle_job_rating_changed(jquery_event_object); })


    }


    // -------------------------------------------------------------

    _handle_cancel_job_clicked(jquery_event_object) {

        var $cancel_job = $(jquery_event_object.target);

        this._request_cancel_job($cancel_job.attr("bx_job_uuid"));

    } // end method

    // -------------------------------------------------------------
    // Request cancel job in the database.

    _request_cancel_job(bx_job_uuid) {

        var json_object = {}
        json_object[this.COMMAND] = this.CANCEL_JOB;
        json_object["bx_job_uuid"] = bx_job_uuid;
        json_object[this.ENABLE_COOKIES] = [this.COOKIE_NAME];

        this.send(json_object);

    } // end method

    // -------------------------------------------------------------

    _handle_delete_job_clicked(jquery_event_object) {

        var $delete_job = $(jquery_event_object.target);

        this._request_delete_job($delete_job.attr("bx_job_uuid"));

    } // end method

    // -------------------------------------------------------------
    // Request delete job in the database.

    _request_delete_job(bx_job_uuid) {

        var json_object = {}
        json_object[this.COMMAND] = this.DELETE_JOB;
        json_object["bx_job_uuid"] = bx_job_uuid;
        json_object[this.ENABLE_COOKIES] = [this.COOKIE_NAME];

        this.send(json_object);

    } // end method

    // -------------------------------------------------------------

    _handle_unblock_job_clicked(jquery_event_object) {

        var $unblock_job = $(jquery_event_object.target);

        this._request_unblock_job($unblock_job.attr("bx_job_uuid"));

    } // end method

    // -------------------------------------------------------------
    // Request unblock job in the database.

    _request_unblock_job(bx_job_uuid) {

        var json_object = {}
        json_object[this.COMMAND] = this.UNBLOCK_JOB;
        json_object["bx_job_uuid"] = bx_job_uuid;
        json_object[this.ENABLE_COOKIES] = [this.COOKIE_NAME];

        this.send(json_object);

    } // end method



    // -------------------------------------------------------------

    _handle_job_comment_changed(jquery_event_object) {
        var F = "Bxflow__RecentJobsUx::_handle_job_comment_changed[" + this.plugin_link_name + "]";

        var $textarea = $(jquery_event_object.target);

        // The bx_job_uuid is an attribute of the containing row.
        var $row = $textarea.closest("TR")
        var bx_job_uuid = $row.attr("bx_job_uuid")

        console.log(F +
            ": $textarea " + $textarea.length +
            ", $row " + $row.length +
            ", bx_job_uuid " + bx_job_uuid)

        var json_object = {}
        json_object[this.COMMAND] = this.UPDATE_JOB;
        json_object["bx_job_uuid"] = bx_job_uuid;
        json_object[this.JOB_COMMENT] = $textarea.val()

        this.send(json_object);

    } // end method


    // -------------------------------------------------------------

    _handle_job_rating_changed(jquery_event_object) {
        var F = "Bxflow__RecentJobsUx::_handle_job_rating_changed[" + this.plugin_link_name + "]";

        var $rating = $(jquery_event_object.target);
        var new_rating = $rating.attr("rating")
        var was_selected = $rating.hasClass("T_selected");
        if (was_selected) {
            new_rating = 0;
        }

        // The bx_job_uuid is an attribute of the containing row.
        var $row = $rating.closest("TR")
        var bx_job_uuid = $row.attr("bx_job_uuid")

        console.log(F +
            ": $rating " + $rating.length +
            ", $row " + $row.length +
            ", bx_job_uuid " + bx_job_uuid +
            ", new_rating " + new_rating);

        var $parent = $rating.parent()
        $parent.children().removeClass("T_selected");
        if (!was_selected)
            $rating.addClass("T_selected");

        var json_object = {}
        json_object[this.COMMAND] = this.UPDATE_JOB;
        json_object["bx_job_uuid"] = bx_job_uuid;
        json_object[this.JOB_RATING] = new_rating;

        this.send(json_object);

    } // end method    
}
