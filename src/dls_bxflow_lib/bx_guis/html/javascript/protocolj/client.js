// --------------------------------------------------------------------

// inherit the base methods and variables
dls__bxview__ProtocoljClient.prototype = new DlsCommon__Base();

// provide an explicit name for the base class
dls__bxview__ProtocoljClient.prototype.base = DlsCommon__Base.prototype;

// override the constructor
dls__bxview__ProtocoljClient.prototype.constructor = dls__bxview__ProtocoljClient;

// -------------------------------------------------------------------------------
// constructor (functioning as a prototype, this constructor cannot take arguments)

function dls__bxview__ProtocoljClient(runtime, classname) {
    var F = "dls__bxview__ProtocoljClient";

    // we are not doing a prototype construction?
    if (arguments.length > 0) {
        var F = "dls__bxview__ProtocoljClient";

        // call the base class constructor helper 
        dls__bxview__ProtocoljClient.prototype.base.constructor.call(
            this,
            runtime,
            classname != undefined ? classname : F);


        this.url = window.location;
    }

} // end constructor

// -------------------------------------------------------------
dls__bxview__ProtocoljClient.prototype.send = function (data) {
    var F = "dls__bxview__ProtocoljClient::send";

    json = JSON.stringify(data);

    console.log("sending " + this.url + " " + json)
    var that = this;

    var $request = $.ajax(
        {
            url: this.url,
            cache: false,
            data: json,
            method: "POST",
            processData: false,
            contentType: "application/json",
            success: function (response, status, jqXHR) { that.handle_success(response, status, jqXHR); },
            error: function (jqXHR, status, error_thrown) { that.handle_error(jqXHR, status, error_thrown); }
        }
    );

} // end method


// -------------------------------------------------------------
// Handle the response when it comes.

dls__bxview__ProtocoljClient.prototype.handle_success = function (response, status, jqXHR) {
    var F = "dls__bxview__ProtocoljClient::handle_success";

    // First, any error obviates anything else.
    var message = response[dls__common__dispatcher__Keywords.ERROR];

    // Some dispatchers return a confirmation string.
    if (message === undefined)
        message = response[dls__common__dispatcher__Keywords.CONFIRMATION];

    if (message === undefined)
        message = "request finished but no error and no confirmation"

    this.message = message;
    this.render();
}


// -------------------------------------------------------------
// Handle the response failure if it comes.
dls__bxview__ProtocoljClient.prototype.handle_error = function (jqXHR, status, error_thrown) {
    var F = "dls__bxview__ProtocoljClient::handle_error";

    this.message = error_thrown;
    this.render();
}


// -------------------------------------------------------------
dls__bxview__ProtocoljClient.prototype.send_arm = function (arguments) {
    var F = "dls__bxview__ProtocoljClient::send_arm";

    arguments[dls__common__dispatcher__Keywords.COMMAND] = dls__common__dispatcher__Commands.ARM;

    this.send(arguments);
} // end method


// -------------------------------------------------------------
dls__bxview__ProtocoljClient.prototype.send_trigger = function (arguments) {
    var F = "dls__bxview__ProtocoljClient::send_trigger";

    json = {}
    json[dls__common__dispatcher__Keywords.COMMAND] = dls__common__dispatcher__Commands.TRIGGER;

    this.send(json);
} // end method


// -------------------------------------------------------------
dls__bxview__ProtocoljClient.prototype.send_disarm = function (arguments) {
    var F = "dls__bxview__ProtocoljClient::send_disarm";

    json = {}
    json[dls__common__dispatcher__Keywords.COMMAND] = dls__common__dispatcher__Commands.DISARM;

    this.send(json);
} // end method


// -------------------------------------------------------------
dls__bxview__ProtocoljClient.prototype.send_stop = function () {
    var F = "dls__bxview__ProtocoljClient::send_stop";

    json = {}
    json[dls__common__dispatcher__Keywords.COMMAND] = dls__common__dispatcher__Commands.STOP;
    this.send(json);
} // end method

// -------------------------------------------------------------
dls__bxview__ProtocoljClient.prototype.send_clear = function () {
    var F = "dls__bxview__ProtocoljClient::send_clear";

    json = {}
    json[dls__common__dispatcher__Keywords.COMMAND] = dls__common__dispatcher__Commands.CLEAR_INCIDENTS;
    this.send(json);
} // end method

// -------------------------------------------------------------
dls__bxview__ProtocoljClient.prototype.send_exit = function () {
    var F = "dls__bxview__ProtocoljClient::send_exit";

    json = {}
    json[dls__common__dispatcher__Keywords.COMMAND] = dls__common__dispatcher__Commands.EXIT;
    this.send(json);
} // end method

// -------------------------------------------------------------
dls__bxview__ProtocoljClient.prototype.render = function () {
    var F = "dls__bxview__ProtocoljClient::render_response";

    $("#response").parent().css("visibility", "visible");
    $("#response").text(this.message);
} // end method
