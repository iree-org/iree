# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest

_TOOL_PATHS = argparse.Namespace(fixture_generator=None, iree_profile=None)


def _runfile_path(relative_path):
    workspace = os.environ.get("TEST_WORKSPACE")
    roots = [
        os.environ.get("RUNFILES_DIR"),
        os.environ.get("TEST_SRCDIR"),
        os.getcwd(),
    ]
    suffixes = ["", ".exe"] if os.name == "nt" else [""]
    for root in roots:
        if not root:
            continue
        prefixes = [root]
        if workspace:
            prefixes.insert(0, os.path.join(root, workspace))
        for prefix in prefixes:
            for suffix in suffixes:
                path = os.path.join(prefix, relative_path) + suffix
                if os.path.exists(path):
                    return path
    raise FileNotFoundError(relative_path)


def _run_checked(argv):
    completed = subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        command = " ".join(argv)
        raise AssertionError(
            f"command failed: {command}\n"
            f"exit_code={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _run_jsonl(argv):
    completed = _run_checked(argv)
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def _find_row(rows, row_type):
    for row in rows:
        if row.get("type") == row_type or row.get("record_type") == row_type:
            return row
    raise AssertionError(f"missing row type {row_type!r}: {rows!r}")


def _row_type_set(rows):
    return {row.get("type", row.get("record_type")) for row in rows}


def _parse_tool_paths(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fixture-generator")
    parser.add_argument("--iree-profile")
    return parser.parse_known_args(argv)


class IreeProfileCliTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture_generator = (
            _TOOL_PATHS.fixture_generator
            if _TOOL_PATHS.fixture_generator
            else _runfile_path(
                "runtime/src/iree/tooling/profile/test_fixture_generator"
            )
        )
        cls.iree_profile = (
            _TOOL_PATHS.iree_profile
            if _TOOL_PATHS.iree_profile
            else _runfile_path("tools/iree-profile")
        )

    def _create_profile(self, directory, *fixture_flags):
        profile_path = os.path.join(directory, "smoke.ireeprof")
        _run_checked(
            [self.fixture_generator, *fixture_flags, profile_path],
        )
        return profile_path

    def _profile_jsonl(self, profile_path, command, *flags):
        return _run_jsonl(
            [self.iree_profile, "--format=jsonl", *flags, command, profile_path]
        )

    def test_report_commands_load_production_sink_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = self._create_profile(directory)
            summary_rows = self._profile_jsonl(profile_path, "summary")
            queue_rows = self._profile_jsonl(profile_path, "queue")
            command_rows = self._profile_jsonl(profile_path, "command")
            executable_rows = self._profile_jsonl(profile_path, "executable")
            dispatch_rows = self._profile_jsonl(profile_path, "dispatch")
            statistics_rows = self._profile_jsonl(profile_path, "statistics")

        self.assertIn("summary", _row_type_set(summary_rows))
        self.assertIn("device_summary", _row_type_set(summary_rows))
        summary = _find_row(summary_rows, "summary")
        self.assertGreater(summary["device_metric_sample_records"], 0)
        self.assertIn("queue_event", _row_type_set(queue_rows))
        self.assertIn("queue_device_event", _row_type_set(queue_rows))
        self.assertIn("host_execution_event", _row_type_set(queue_rows))
        self.assertIn("command_operation", _row_type_set(command_rows))
        self.assertIn("command_execution", _row_type_set(command_rows))
        self.assertIn("command_host_execution", _row_type_set(command_rows))
        self.assertIn(
            "executable_export_host_dispatch_group", _row_type_set(executable_rows)
        )
        self.assertIn("host_dispatch_group", _row_type_set(dispatch_rows))
        self.assertIn("statistics_summary", _row_type_set(statistics_rows))
        self.assertIn("statistics_row", _row_type_set(statistics_rows))

    def test_statistics_reports_aggregate_row_families(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self._profile_jsonl(self._create_profile(directory), "statistics")

        summary = _find_row(rows, "statistics_summary")
        self.assertGreater(summary["row_count"], 0)
        self.assertEqual(summary["dropped_record_count"], 0)

        row_types = {
            row["row_type"] for row in rows if row.get("type") == "statistics_row"
        }
        self.assertIn("dispatch_export", row_types)
        self.assertIn("queue_device_operation", row_types)
        self.assertIn("queue_host_operation", row_types)
        self.assertIn("host_execution_export", row_types)
        self.assertIn("memory_lifecycle", row_types)

        export_row = next(
            row
            for row in rows
            if row.get("type") == "statistics_row"
            and row.get("row_type") == "dispatch_export"
        )
        self.assertEqual(export_row["export_name"], "smoke_export")
        self.assertGreater(export_row["total_duration_ns"], 0)

    def test_queue_projection_separates_host_and_device_time(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self._profile_jsonl(self._create_profile(directory), "queue")

        queue_event = _find_row(rows, "queue_event")
        self.assertEqual(queue_event["host_time_domain"], "iree_host_time_ns")
        self.assertLessEqual(
            queue_event["host_time_ns"], queue_event["ready_host_time_ns"]
        )

        device_event = _find_row(rows, "queue_device_event")
        self.assertIn("start_tick", device_event)
        self.assertIn("end_tick", device_event)
        self.assertIn("duration_ns", device_event)

        host_event = _find_row(rows, "host_execution_event")
        self.assertEqual(host_event["host_time_domain"], "iree_host_time_ns")
        self.assertEqual(queue_event["submission_id"], device_event["submission_id"])
        self.assertEqual(queue_event["submission_id"], host_event["submission_id"])
        self.assertEqual(queue_event["op"], device_event["op"])
        self.assertEqual(queue_event["op"], host_event["op"])

    def test_command_and_dispatch_projections_join_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = self._create_profile(directory)
            command_rows = self._profile_jsonl(profile_path, "command")
            dispatch_rows = self._profile_jsonl(
                profile_path, "dispatch", "--dispatch_events"
            )

        operation = _find_row(command_rows, "command_operation")
        self.assertEqual(operation["op"], "dispatch")

        execution = _find_row(command_rows, "command_execution")
        self.assertEqual(operation["command_buffer_id"], execution["command_buffer_id"])

        dispatch_event = _find_row(dispatch_rows, "dispatch_event")
        self.assertEqual(
            operation["command_buffer_id"], dispatch_event["command_buffer_id"]
        )
        self.assertEqual(operation["command_index"], dispatch_event["command_index"])
        self.assertEqual(dispatch_event["device_tick_domain"], "device_tick")
        self.assertEqual(
            dispatch_event["duration_time_domain"], "device_tick_duration_ns"
        )

        host_dispatch_event = _find_row(dispatch_rows, "host_dispatch_event")
        self.assertEqual(
            operation["command_buffer_id"], host_dispatch_event["command_buffer_id"]
        )
        self.assertEqual(
            operation["command_index"], host_dispatch_event["command_index"]
        )
        self.assertEqual(host_dispatch_event["timing_source"], "host_execution_event")
        self.assertEqual(host_dispatch_event["time_domain"], "iree_host_time_ns")
        self.assertEqual(
            host_dispatch_event["duration_time_domain"], "iree_host_duration_ns"
        )

    def test_explain_uses_host_dispatch_spans_without_device_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = self._create_profile(directory, "--omit-dispatch-events")
            rows = self._profile_jsonl(profile_path, "explain")

        summary = _find_row(rows, "explain_summary")
        self.assertEqual(summary["valid_dispatches"], 0)
        self.assertGreater(summary["valid_host_dispatches"], 0)

        top_export = _find_row(rows, "explain_top_export")
        self.assertEqual(top_export["timing_source"], "host_execution_event")
        self.assertGreater(top_export["total_ns"], 0)
        self.assertGreater(top_export["total_tile_count"], 0)

        top_dispatch = _find_row(rows, "explain_top_dispatch")
        self.assertEqual(top_dispatch["timing_source"], "host_execution_event")
        self.assertGreater(top_dispatch["duration_ns"], 0)
        self.assertGreater(top_dispatch["tile_count"], 0)

    def test_export_ireeperf_jsonl_decodes_record_families(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = self._create_profile(directory)
            output_path = os.path.join(directory, "smoke.ireeperf.jsonl")
            _run_checked(
                [
                    self.iree_profile,
                    "--format=ireeperf-jsonl",
                    f"--output={output_path}",
                    "export",
                    profile_path,
                ]
            )
            with open(output_path, "r", encoding="utf-8") as file:
                rows = [json.loads(line) for line in file if line.strip()]

        schema = _find_row(rows, "schema")
        self.assertEqual(schema["format"], "ireeperf-jsonl")
        self.assertEqual(schema["row_key"], "record_type")

        queue_event = _find_row(rows, "queue_event")
        self.assertEqual(queue_event["host_time_domain"], "iree_host_time_ns")
        self.assertLessEqual(
            queue_event["host_time_ns"], queue_event["ready_host_time_ns"]
        )

        device_event = _find_row(rows, "queue_device_event")
        self.assertIn("derived_time_available", device_event)
        self.assertIn("duration_ns", device_event)

        relationship = _find_row(rows, "event_relationship")
        self.assertEqual(relationship["kind"], "queue_event_host_execution_event")
        self.assertEqual(relationship["source_type"], "queue_event")
        self.assertEqual(relationship["target_type"], "host_execution_event")

        metric_source = _find_row(rows, "device_metric_source")
        self.assertEqual(metric_source["device_class"], "gpu")
        self.assertEqual(metric_source["name"], "smoke.metrics")

        metric_descriptor = _find_row(rows, "device_metric_descriptor")
        self.assertIn(metric_descriptor["unit"], {"count", "millipercent"})

        metric_sample = _find_row(rows, "device_metric_sample")
        self.assertEqual(metric_sample["host_time_domain"], "iree_host_time_ns")
        self.assertEqual(len(metric_sample["values"]), 2)
        value_names = {value["name"] for value in metric_sample["values"]}
        self.assertIn("activity.compute", value_names)
        self.assertIn("smoke.source_specific", value_names)
        self.assertTrue(all("value" in value for value in metric_sample["values"]))


if __name__ == "__main__":
    _TOOL_PATHS, remaining_argv = _parse_tool_paths(sys.argv[1:])
    unittest.main(argv=[sys.argv[0], *remaining_argv])
