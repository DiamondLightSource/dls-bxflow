import argparse
import json

import dls_mainiac_lib.version

import dls_bxflow_cli
import dls_bxflow_lib.version


# ----------------------------------------------------------
def version():
    """
    Current version.
    DO NOT CHANGE THE NEXT LINE, since bumpversion expects it exactly.
    """

    return dls_bxflow_cli.__version__


# ----------------------------------------------------------
def meta(given_meta=None):
    """
    Returns version information as a dict.
    Adds version information to given meta, if any.
    """
    s = {}
    s["dls_bxflow_cli"] = version()
    s.update(dls_bxflow_lib.version.meta())
    s.update(dls_mainiac_lib.version.meta())

    if given_meta is not None:
        given_meta.update(s)
    else:
        given_meta = s
    return given_meta


# ----------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print version stack in json.",
    )

    # -------------------------------------------------------------------------
    given_args, remaining_args = parser.parse_known_args()

    if given_args.json:
        print(json.dumps(meta(), indent=4))
    else:
        print(version())


# ----------------------------------------------------------
if __name__ == "__main__":
    main()
