#!/usr/bin/env python3
"""Unit tests for iree-profile-render shared helpers."""

from __future__ import annotations

import unittest

from iree_profile_render import backends
from iree_profile_render import common
from iree_profile_render import perfetto


class _FakeRepeated(list):

    def __init__(self, item_type):
        super().__init__()
        self._item_type = item_type

    def add(self):
        item = self._item_type()
        self.append(item)
        return item


class _FakeAnnotation:
    pass


class _FakeCounter:

    def SetInParent(self):
        pass


class _FakeTrackDescriptor:
    EXPLICIT = 1

    def __init__(self):
        self.counter = _FakeCounter()


class _FakeTrackEvent:
    TYPE_SLICE_BEGIN = 1
    TYPE_SLICE_END = 2
    TYPE_INSTANT = 3
    TYPE_COUNTER = 4

    def __init__(self):
        self.flow_ids = []
        self.debug_annotations = _FakeRepeated(_FakeAnnotation)


class _FakePacket:

    def __init__(self):
        self.track_descriptor = _FakeTrackDescriptor()
        self.track_event = _FakeTrackEvent()


class _FakeTraceProtoBuilder:

    def __init__(self):
        self.packets = []

    def add_packet(self):
        packet = _FakePacket()
        self.packets.append(packet)
        return packet

    def serialize(self):
        return f"packets={len(self.packets)}".encode("utf-8")


class CommonTest(unittest.TestCase):

    def test_validate_schema_rejects_old_version(self):
        with self.assertRaisesRegex(SystemExit, "schema version 2; expected 5"):
            common.validate_schema(
                [
                    {
                        "schema_version": 2,
                        "record_type": "schema",
                        "format": "ireeperf-jsonl",
                    }
                ]
            )

    def test_clock_mapper_interpolates_device_ticks(self):
        mappers = common.build_device_clock_mappers(
            [
                {
                    "record_type": "clock_correlation",
                    "physical_device_ordinal": 0,
                    "device_tick": 100,
                    "host_time_begin_ns": 1000,
                    "host_time_end_ns": 1000,
                },
                {
                    "record_type": "clock_correlation",
                    "physical_device_ordinal": 0,
                    "device_tick": 200,
                    "host_time_begin_ns": 2000,
                    "host_time_end_ns": 2000,
                },
            ]
        )

        self.assertEqual(1500, mappers[0].host_time_from_device_tick(150))

    def test_device_event_uses_clock_fit_before_derived_host_time(self):
        mappers = common.build_device_clock_mappers(
            [
                {
                    "record_type": "clock_correlation",
                    "physical_device_ordinal": 0,
                    "device_tick": 100,
                    "host_time_begin_ns": 1000,
                    "host_time_end_ns": 1000,
                },
                {
                    "record_type": "clock_correlation",
                    "physical_device_ordinal": 0,
                    "device_tick": 200,
                    "host_time_begin_ns": 2000,
                    "host_time_end_ns": 2000,
                },
            ]
        )

        self.assertEqual(
            (1200, 1300, "iree_host_time_from_device_clock_fit"),
            common.device_event_host_time_range(
                {
                    "valid": True,
                    "physical_device_ordinal": 0,
                    "start_tick": 120,
                    "end_tick": 130,
                    "derived_time_available": True,
                    "start_driver_host_cpu_time_ns": 999999,
                    "end_driver_host_cpu_time_ns": 1000000,
                },
                mappers,
            ),
        )


class PerfettoTest(unittest.TestCase):

    def test_perfetto_backend_is_registered(self):
        self.assertIn("perfetto", backends.BACKENDS)
        self.assertEqual(
            "Native Perfetto TrackEvent .pftrace.",
            backends.BACKENDS["perfetto"].description,
        )

    def test_lane_allocator_uses_lowest_free_lane(self):
        lanes = perfetto.LaneAllocator()

        self.assertEqual(0, lanes.allocate(0, 10))
        self.assertEqual(1, lanes.allocate(1, 9))
        self.assertEqual(0, lanes.allocate(10, 11))
        self.assertEqual(1, lanes.allocate(10, 12))

    def test_build_trace_with_fake_perfetto_backend(self):
        trace_bytes, stats = perfetto.build_trace(
            [
                {"record_type": "device", "physical_device_ordinal": 0},
                {
                    "record_type": "queue",
                    "physical_device_ordinal": 0,
                    "queue_ordinal": 0,
                    "stream_id": 0,
                },
                {
                    "record_type": "clock_correlation",
                    "physical_device_ordinal": 0,
                    "device_tick": 100,
                    "host_time_begin_ns": 1000,
                    "host_time_end_ns": 1000,
                },
                {
                    "record_type": "clock_correlation",
                    "physical_device_ordinal": 0,
                    "device_tick": 200,
                    "host_time_begin_ns": 2000,
                    "host_time_end_ns": 2000,
                },
                {
                    "record_type": "event_relationship",
                    "relationship_id": 1,
                    "kind": "queue_submission_dispatch",
                    "source_type": "queue_submission",
                    "target_type": "dispatch_event",
                    "physical_device_ordinal": 0,
                    "queue_ordinal": 0,
                    "stream_id": 0,
                    "source_id": 1,
                    "target_id": 2,
                },
                {
                    "record_type": "queue_event",
                    "event_id": 1,
                    "submission_id": 1,
                    "op": "execute",
                    "physical_device_ordinal": 0,
                    "queue_ordinal": 0,
                    "stream_id": 0,
                    "host_time_ns": 1050,
                },
                {
                    "record_type": "dispatch_event",
                    "event_id": 2,
                    "submission_id": 1,
                    "key": "dispatch_0",
                    "valid": True,
                    "physical_device_ordinal": 0,
                    "queue_ordinal": 0,
                    "stream_id": 0,
                    "start_tick": 110,
                    "end_tick": 120,
                },
            ],
            perfetto.PerfettoImports(
                trace_proto_builder=_FakeTraceProtoBuilder,
                track_descriptor=_FakeTrackDescriptor,
                track_event=_FakeTrackEvent,
            ),
        )

        self.assertTrue(trace_bytes.startswith(b"packets="))
        self.assertEqual(1, stats.dispatch_slices)
        self.assertEqual(1, stats.queue_instants)
        self.assertEqual(2, stats.clock_instants)
        self.assertEqual(1, stats.relationship_flows)


if __name__ == "__main__":
    unittest.main()
