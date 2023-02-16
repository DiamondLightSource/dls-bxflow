// Implement the logic common to all classes.

class DlsCommon__Base extends EventTarget {
  constructor(runtime) {
    super();
    this.runtime = runtime;
  }
}
