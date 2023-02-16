// Class backing the actions ux.


class Bxflow__SystemHealthUx extends Bxflow__UxAutoUpdate {
    COOKIE_NAME = "BXFLOW_SYSTEM_HEALTH_UX";
    GET_SYSTEM_HEALTH = "bx_guis::commands::get_system_health";

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
        var F = "Bxflow__SystemHealthUx::activate";

        super.activate()

        this.#jquery_objects.$div = $(".T_composed", this.$interaction_parent);

        // this.request_update();

    } // end method

    // -------------------------------------------------------------
    // Request update from database.

    request_update() {

        var json_object = {}
        json_object[this.COMMAND] = this.GET_SYSTEM_HEALTH;
        json_object[this.ENABLE_COOKIES] = [this.COOKIE_NAME];

        this.send(json_object);

    } // end method

    // -------------------------------------------------------------
    // Handle the response when it comes.

    handle_ajax_success(response, status, jqXHR) {
        var F = "Bxflow__SystemHealthUx::_handle_ajax_success";

        // Let the base class check for and display any error in the response.
        var error_message = super.handle_ajax_success(response, status, jqXHR);

        if (error_message !== null)
            return;

        // Assign the composed html to the DOM element.
        var html = response.html;
        this.#jquery_objects.$div.html(html);
    }

}