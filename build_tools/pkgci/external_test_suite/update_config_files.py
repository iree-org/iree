# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script to help with updating config.json files used with conftest.py.
#
# Usage:
#   1. Run tests with --report-log=logs.json
#   2. Run the script: `python update_config_xfails.py --config-file=config.json --log-file=logs.json`
#   3. Commit the modified config.json

import argparse
import json
import logging
import pyjson5

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file updater.")
    parser.add_argument(
        "--config-file",
        default="",
        required=True,
        help="Path to a config JSON file (`pytest --config-files=config.json`) to update",
    )
    parser.add_argument(
        "--log-file",
        default="",
        required=True,
        help="Path to a log JSON file (`pytest --report-log=logs.json`) to read results from",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Set the logging level",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    config_file = args.config_file

    logger.info(f"Reading config from '{config_file}'")
    with open(config_file, "r") as f:
        config = pyjson5.load(f)

        # Sanity check the config file structure before going any further.
        def check_field(field_name):
            if field_name not in config:
                raise ValueError(
                    f"config file '{config_file}' is missing a '{field_name}' field"
                )

        check_field("config_name")
        check_field("iree_compile_flags")
        check_field("iree_run_module_flags")

    with open(args.log_file, "r") as f:
        log_lines = f.readlines()

    compile_failures = []
    run_failures = []

    for log_line in log_lines:
        log_json = pyjson5.decode(log_line)

        # Only look at test "call" events in the log.
        if "when" not in log_json or log_json["when"] != "call":
            continue

        if log_json["outcome"] == "passed":
            continue

        # conftest.py adds extra info with `self.user_properties.append` for us
        # to use. If that info is missing skip this log line.
        if "user_properties" not in log_json or not log_json["user_properties"]:
            logger.warning("Missing 'user_properties', ignoring log line")
            continue
        user_properties = log_json["user_properties"]

        # TODO(scotttodd): handle multiple config files writing to one log file?

        # Find the test directory path relative to the iree_tests root, since
        # that is what our config.json uses to label tests for skipping or XFAIL.
        relative_test_directory_name = ""
        for user_property in user_properties:
            if user_property[0] == "relative_test_directory_name":
                relative_test_directory_name = user_property[1]
        if not relative_test_directory_name:
            logger.warning(
                "Missing 'relative_test_directory_name' property, ignoring log line"
            )
            continue

        # If the test failed, it should have a "repr" from repr_failure().
        # Parse the text of that to determine the test result.
        repr = log_json["longrepr"] if "longrepr" in log_json else ""
        if not repr:
            continue

        if "Error invoking iree-compile" in repr:
            logger.debug(f"test {relative_test_directory_name} failed to compile")
            compile_failures.append(relative_test_directory_name)
        elif "Error invoking iree-run-module" in repr:
            logger.debug(f"test {relative_test_directory_name} failed to run")
            run_failures.append(relative_test_directory_name)
        elif "remove from 'expected_compile_failures'" in repr:
            logger.debug(
                f"test {relative_test_directory_name} compiled and ran successfully"
            )
        elif "remove from 'expected_run_failures'" in repr:
            logger.debug(f"test {relative_test_directory_name} ran successfully")
        elif "move to 'expected_run_failures'" in repr:
            logger.debug(
                f"test {relative_test_directory_name} compile xfail -> run fail"
            )
            run_failures.append(relative_test_directory_name)
        else:
            logger.warning(
                f"Unhandled error for {relative_test_directory_name}: '{repr}'"
            )

    logger.info(f"Updating config")
    # Remove duplicates and sort.
    config["expected_compile_failures"] = sorted(list(set(compile_failures)))
    config["expected_run_failures"] = sorted(list(set(run_failures)))

    logger.info(f"Writing updated config to '{config_file}'")
    with open(config_file, "w") as f:
        print(json.dumps(config, indent=2), file=f)
