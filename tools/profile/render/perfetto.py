"""Perfetto renderer for ireeperf-jsonl streams."""

from __future__ import annotations

import collections
import dataclasses
import decimal
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable

from .common import (
    INT64_MAX,
    INT64_MIN,
    UINT64_MASK,
    build_device_clock_mappers,
    device_event_host_time_range,
    event_annotations,
    event_endpoint_key,
    normalized_time_range,
    parse_integer,
    parse_ordinal,
    queue_key,
    relationship_endpoint_key,
    submission_key,
    timestamp_midpoint,
)


TRUSTED_PACKET_SEQUENCE_ID = 1001
FORMAT_NAME = "perfetto"
FORMAT_DESCRIPTION = "Native Perfetto TrackEvent .pftrace."

DEPENDENCY_HINT = """\
Missing optional Perfetto Python dependency.

Run this script in a one-shot environment:

  uvx --with perfetto --with protobuf python tools/profile/iree-profile-render \\
      --format=perfetto INPUT.ireeperf.jsonl -o OUTPUT.pftrace

or install the optional packages into your active environment:

  python -m pip install perfetto protobuf
"""


@dataclasses.dataclass
class PerfettoImports:
    trace_proto_builder: type
    track_descriptor: type
    track_event: type


@dataclasses.dataclass
class TimelineEvent:
    timestamp_ns: int
    sort_key: tuple[Any, ...]
    callback: Any


@dataclasses.dataclass
class PendingSlice:
    lane_family: str
    record_family: str
    sequence_index: int
    start_time_ns: int
    end_time_ns: int
    physical_device_ordinal: int
    queue_ordinal: int
    name: str
    annotations: dict[str, Any]
    flow_ids: tuple[int, ...]


@dataclasses.dataclass
class ConversionStats:
    records: int = 0
    output_byte_count: int = 0
    dispatch_slices: int = 0
    queue_device_slices: int = 0
    host_execution_slices: int = 0
    queue_instants: int = 0
    memory_instants: int = 0
    clock_instants: int = 0
    diagnostic_instants: int = 0
    counter_samples: int = 0
    relationship_flows: int = 0
    skipped_dispatches: int = 0
    skipped_queue_device_events: int = 0
    skipped_host_execution_events: int = 0


class TrackRegistry:
    """Creates deterministic Perfetto custom tracks."""

    def __init__(self, builder: Any, track_descriptor: type):
        self._builder = builder
        self._track_descriptor = track_descriptor
        self._defined: set[int] = set()

    def uuid(self, *parts: Any) -> int:
        digest = hashlib.blake2b(digest_size=8)
        for part in parts:
            digest.update(str(part).encode("utf-8"))
            digest.update(b"\0")
        value = int.from_bytes(digest.digest(), "little") & UINT64_MASK
        return value or 1

    def define(
        self,
        track_uuid: int,
        name: str,
        *,
        parent_uuid: int | None = None,
        sibling_order_rank: int | None = None,
        is_counter: bool = False,
        explicit_child_order: bool = False,
    ) -> int:
        if track_uuid in self._defined:
            return track_uuid
        packet = self._builder.add_packet()
        descriptor = packet.track_descriptor
        descriptor.uuid = track_uuid
        descriptor.name = name
        if parent_uuid is not None:
            descriptor.parent_uuid = parent_uuid
        if sibling_order_rank is not None:
            descriptor.sibling_order_rank = sibling_order_rank
        if is_counter:
            descriptor.counter.SetInParent()
        if explicit_child_order:
            descriptor.child_ordering = self._track_descriptor.EXPLICIT
        self._defined.add(track_uuid)
        return track_uuid


class LaneAllocator:
    """Assigns start-time-sorted overlapping slices to sibling tracks."""

    def __init__(self):
        self._active_lanes: list[tuple[int, int]] = []
        self._free_lanes: list[int] = []
        self._lane_count = 0

    def allocate(self, start_time_ns: int, end_time_ns: int) -> int:
        while self._active_lanes and start_time_ns >= self._active_lanes[0][0]:
            _, lane_index = heapq.heappop(self._active_lanes)
            heapq.heappush(self._free_lanes, lane_index)
        if self._free_lanes:
            lane_index = heapq.heappop(self._free_lanes)
        else:
            lane_index = self._lane_count
            self._lane_count += 1
        heapq.heappush(self._active_lanes, (end_time_ns, lane_index))
        return lane_index


def import_perfetto() -> PerfettoImports:
    try:
        from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import (
            TrackDescriptor,
            TrackEvent,
        )
        from perfetto.trace_builder.proto_builder import TraceProtoBuilder
    except ImportError as error:
        raise SystemExit(f"{DEPENDENCY_HINT}\nImport error: {error}") from error
    return PerfettoImports(
        trace_proto_builder=TraceProtoBuilder,
        track_descriptor=TrackDescriptor,
        track_event=TrackEvent,
    )


def add_debug_annotations(track_event: Any, values: dict[str, Any]) -> None:
    for name, value in values.items():
        if value is None:
            continue
        annotation = track_event.debug_annotations.add()
        annotation.name = name
        if isinstance(value, bool):
            annotation.bool_value = value
        elif isinstance(value, decimal.Decimal):
            annotation.double_value = float(value)
        elif isinstance(value, int):
            if 0 <= value <= UINT64_MASK:
                annotation.uint_value = value
            elif INT64_MIN <= value <= INT64_MAX:
                annotation.int_value = value
            else:
                annotation.string_value = str(value)
        elif isinstance(value, float):
            annotation.double_value = value
        elif isinstance(value, (list, tuple, dict)):
            annotation.string_value = json.dumps(value, sort_keys=True)
        else:
            annotation.string_value = str(value)


def add_slice_begin(
    builder: Any,
    track_event_type: type,
    timestamp_ns: int,
    track_uuid: int,
    name: str,
    annotations: dict[str, Any],
    flow_ids: Iterable[int] = (),
) -> None:
    packet = builder.add_packet()
    packet.timestamp = timestamp_ns
    packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID
    event = packet.track_event
    event.type = track_event_type.TYPE_SLICE_BEGIN
    event.track_uuid = track_uuid
    event.name = name
    for flow_id in flow_ids:
        event.flow_ids.append(flow_id)
    add_debug_annotations(event, annotations)


def add_slice_end(
    builder: Any,
    track_event_type: type,
    timestamp_ns: int,
    track_uuid: int,
) -> None:
    packet = builder.add_packet()
    packet.timestamp = timestamp_ns
    packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID
    event = packet.track_event
    event.type = track_event_type.TYPE_SLICE_END
    event.track_uuid = track_uuid


def add_instant(
    builder: Any,
    track_event_type: type,
    timestamp_ns: int,
    track_uuid: int,
    name: str,
    annotations: dict[str, Any],
    flow_ids: Iterable[int] = (),
) -> None:
    packet = builder.add_packet()
    packet.timestamp = timestamp_ns
    packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID
    event = packet.track_event
    event.type = track_event_type.TYPE_INSTANT
    event.track_uuid = track_uuid
    event.name = name
    for flow_id in flow_ids:
        event.flow_ids.append(flow_id)
    add_debug_annotations(event, annotations)


def add_counter(
    builder: Any,
    track_event_type: type,
    timestamp_ns: int,
    track_uuid: int,
    value: int,
) -> None:
    packet = builder.add_packet()
    packet.timestamp = timestamp_ns
    packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID
    event = packet.track_event
    event.type = track_event_type.TYPE_COUNTER
    event.track_uuid = track_uuid
    event.counter_value = value


class PerfettoTraceConverter:
    """Converts ireeperf-jsonl records into a Perfetto trace."""

    def __init__(self, records: list[dict[str, Any]], perfetto: PerfettoImports):
        self.records = records
        self.perfetto = perfetto
        self.builder = perfetto.trace_proto_builder()
        self.tracks = TrackRegistry(self.builder, perfetto.track_descriptor)
        self.stats = ConversionStats(records=len(records))
        self.clock_mappers = build_device_clock_mappers(records)
        self.root_uuid = self.tracks.uuid("iree", "root")
        self.diagnostics_uuid = self.tracks.uuid("iree", "diagnostics")
        self.dispatch_lane_allocators: dict[tuple[int, int], LaneAllocator] = (
            collections.defaultdict(LaneAllocator)
        )
        self.queue_device_lane_allocators: dict[tuple[int, int], LaneAllocator] = (
            collections.defaultdict(LaneAllocator)
        )
        self.host_execution_lane_allocators: dict[tuple[int, int], LaneAllocator] = (
            collections.defaultdict(LaneAllocator)
        )
        self.timeline_events: list[TimelineEvent] = []
        self.pending_slices: list[PendingSlice] = []
        self.all_timestamp_ns: list[int] = []
        self.flow_ids_by_endpoint: dict[tuple[str, int, int, int, int], list[int]] = (
            collections.defaultdict(list)
        )
        self.flow_ids_by_submission: dict[tuple[int, int, int, int], list[int]] = (
            collections.defaultdict(list)
        )

    def build(self) -> tuple[bytes, ConversionStats]:
        self.define_root_tracks()
        self.define_recorded_tracks()
        self.collect_relationships()
        for record in self.records:
            self.collect_record(record)
        self.emit_pending_slices()
        self.emit_queue_allocation_counters()
        self.emit_timeline_events()
        return self.builder.serialize(), self.stats

    def define_root_tracks(self) -> None:
        self.tracks.define(
            self.root_uuid,
            "IREE HAL profile",
            sibling_order_rank=0,
            explicit_child_order=True,
        )
        self.tracks.define(
            self.diagnostics_uuid,
            "diagnostics",
            parent_uuid=self.root_uuid,
            sibling_order_rank=100000,
        )

    def define_recorded_tracks(self) -> None:
        for record in self.records:
            record_type = record.get("record_type")
            if record_type == "device":
                self.ensure_device_track(
                    parse_integer(record["physical_device_ordinal"]), record
                )
            elif record_type == "queue":
                physical_device_ordinal, queue_ordinal = queue_key(record)
                self.ensure_queue_tracks(
                    physical_device_ordinal, queue_ordinal, record.get("stream_id")
                )

    def ensure_device_track(
        self, physical_device_ordinal: int, record: dict[str, Any] | None = None
    ) -> int:
        device_uuid = self.tracks.uuid("iree", "device", physical_device_ordinal)
        device_name = (
            "device[unknown]"
            if physical_device_ordinal < 0
            else f"device[{physical_device_ordinal}]"
        )
        if record is not None and record.get("physical_device_uuid_present"):
            device_name += f" {record.get('physical_device_uuid')}"
        sibling_order_rank = (
            999999 if physical_device_ordinal < 0 else physical_device_ordinal
        )
        self.tracks.define(
            device_uuid,
            device_name,
            parent_uuid=self.root_uuid,
            sibling_order_rank=sibling_order_rank,
            explicit_child_order=True,
        )
        return device_uuid

    def ensure_queue_tracks(
        self,
        physical_device_ordinal: int,
        queue_ordinal: int,
        stream_id: Any | None = None,
    ) -> int:
        device_uuid = self.ensure_device_track(physical_device_ordinal)
        queue_uuid = self.tracks.uuid(
            "iree", "queue", physical_device_ordinal, queue_ordinal
        )
        queue_name = (
            "device-scope events" if queue_ordinal < 0 else f"queue[{queue_ordinal}]"
        )
        if queue_ordinal >= 0 and stream_id is not None:
            queue_name += f" stream={stream_id}"
        sibling_order_rank = 999999 if queue_ordinal < 0 else queue_ordinal
        self.tracks.define(
            queue_uuid,
            queue_name,
            parent_uuid=device_uuid,
            sibling_order_rank=sibling_order_rank,
            explicit_child_order=True,
        )
        self.tracks.define(
            self.tracks.uuid(
                "iree", "queue-events", physical_device_ordinal, queue_ordinal
            ),
            "host queue events",
            parent_uuid=queue_uuid,
            sibling_order_rank=0,
        )
        self.tracks.define(
            self.tracks.uuid("iree", "memory", physical_device_ordinal, queue_ordinal),
            "host memory events",
            parent_uuid=queue_uuid,
            sibling_order_rank=4000,
        )
        self.tracks.define(
            self.tracks.uuid(
                "iree", "queue-allocation-bytes", physical_device_ordinal, queue_ordinal
            ),
            "queue allocation bytes",
            parent_uuid=queue_uuid,
            sibling_order_rank=5000,
            is_counter=True,
        )
        return queue_uuid

    def collect_relationships(self) -> None:
        slice_endpoint_types = {
            "dispatch_event",
            "queue_device_event",
            "host_execution_event",
        }
        for record in self.records:
            if record.get("record_type") != "event_relationship":
                continue
            source_endpoint = relationship_endpoint_key(record, "source")
            target_endpoint = relationship_endpoint_key(record, "target")
            if (
                source_endpoint is None
                or target_endpoint is None
                or target_endpoint[0] not in slice_endpoint_types
            ):
                continue
            flow_id = self.tracks.uuid(
                "iree",
                "relationship-flow",
                parse_integer(record.get("relationship_id", 0)),
                record.get("kind"),
                source_endpoint,
                target_endpoint,
            )
            self.flow_ids_by_endpoint[target_endpoint].append(flow_id)
            self.flow_ids_by_endpoint[source_endpoint].append(flow_id)
            if source_endpoint[0] == "queue_submission":
                self.flow_ids_by_submission[source_endpoint[1:]].append(flow_id)
            self.stats.relationship_flows += 1

    def endpoint_flow_ids(
        self, record: dict[str, Any], endpoint_type: str
    ) -> tuple[int, ...]:
        endpoint_key = event_endpoint_key(record, endpoint_type)
        if endpoint_key is None:
            return ()
        return tuple(self.flow_ids_by_endpoint.get(endpoint_key, ()))

    def define_dispatch_lane(
        self, physical_device_ordinal: int, queue_ordinal: int, lane_index: int
    ) -> int:
        queue_uuid = self.ensure_queue_tracks(physical_device_ordinal, queue_ordinal)
        track_uuid = self.tracks.uuid(
            "iree", "dispatch", physical_device_ordinal, queue_ordinal, lane_index
        )
        self.tracks.define(
            track_uuid,
            "dispatch" if lane_index == 0 else f"dispatch lane {lane_index}",
            parent_uuid=queue_uuid,
            sibling_order_rank=2000 + lane_index,
        )
        return track_uuid

    def define_queue_device_lane(
        self, physical_device_ordinal: int, queue_ordinal: int, lane_index: int
    ) -> int:
        queue_uuid = self.ensure_queue_tracks(physical_device_ordinal, queue_ordinal)
        track_uuid = self.tracks.uuid(
            "iree", "queue-device", physical_device_ordinal, queue_ordinal, lane_index
        )
        self.tracks.define(
            track_uuid,
            (
                "device queue ops"
                if lane_index == 0
                else f"device queue ops lane {lane_index}"
            ),
            parent_uuid=queue_uuid,
            sibling_order_rank=1000 + lane_index,
        )
        return track_uuid

    def define_host_execution_lane(
        self, physical_device_ordinal: int, queue_ordinal: int, lane_index: int
    ) -> int:
        queue_uuid = self.ensure_queue_tracks(physical_device_ordinal, queue_ordinal)
        track_uuid = self.tracks.uuid(
            "iree",
            "host-execution",
            physical_device_ordinal,
            queue_ordinal,
            lane_index,
        )
        self.tracks.define(
            track_uuid,
            (
                "host execution"
                if lane_index == 0
                else f"host execution lane {lane_index}"
            ),
            parent_uuid=queue_uuid,
            sibling_order_rank=3000 + lane_index,
        )
        return track_uuid

    def append_slice_events(
        self,
        *,
        record_family: str,
        start_time_ns: int,
        end_time_ns: int,
        physical_device_ordinal: int,
        queue_ordinal: int,
        track_uuid: int,
        name: str,
        annotations: dict[str, Any],
        flow_ids: Iterable[int],
    ) -> None:
        def emit_begin(
            timestamp_ns: int = start_time_ns,
            event_track_uuid: int = track_uuid,
            event_name: str = name,
            event_annotations: dict[str, Any] = annotations,
            event_flow_ids: Iterable[int] = tuple(flow_ids),
        ) -> None:
            add_slice_begin(
                self.builder,
                self.perfetto.track_event,
                timestamp_ns,
                event_track_uuid,
                event_name,
                event_annotations,
                event_flow_ids,
            )

        def emit_end(
            timestamp_ns: int = end_time_ns,
            event_track_uuid: int = track_uuid,
        ) -> None:
            add_slice_end(
                self.builder,
                self.perfetto.track_event,
                timestamp_ns,
                event_track_uuid,
            )

        self.timeline_events.append(
            TimelineEvent(
                start_time_ns,
                (
                    1,
                    f"{record_family}_begin",
                    physical_device_ordinal,
                    queue_ordinal,
                    start_time_ns,
                ),
                emit_begin,
            )
        )
        self.timeline_events.append(
            TimelineEvent(
                end_time_ns,
                (
                    0,
                    f"{record_family}_end",
                    physical_device_ordinal,
                    queue_ordinal,
                    end_time_ns,
                ),
                emit_end,
            )
        )
        self.all_timestamp_ns.extend([start_time_ns, end_time_ns])

    def append_pending_slice(
        self,
        *,
        lane_family: str,
        record_family: str,
        start_time_ns: int,
        end_time_ns: int,
        physical_device_ordinal: int,
        queue_ordinal: int,
        name: str,
        annotations: dict[str, Any],
        flow_ids: Iterable[int],
    ) -> None:
        self.pending_slices.append(
            PendingSlice(
                lane_family=lane_family,
                record_family=record_family,
                sequence_index=len(self.pending_slices),
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                physical_device_ordinal=physical_device_ordinal,
                queue_ordinal=queue_ordinal,
                name=name,
                annotations=annotations,
                flow_ids=tuple(flow_ids),
            )
        )

    def collect_record(self, record: dict[str, Any]) -> None:
        record_type = record.get("record_type")
        if record_type == "dispatch_event":
            self.collect_dispatch_event(record)
        elif record_type == "queue_device_event":
            self.collect_queue_device_event(record)
        elif record_type == "host_execution_event":
            self.collect_host_execution_event(record)
        elif record_type == "queue_event":
            self.collect_queue_event(record)
        elif record_type == "memory_event":
            self.collect_memory_event(record)
        elif record_type == "clock_correlation":
            self.collect_clock_correlation(record)
        elif record_type == "diagnostic":
            self.collect_diagnostic(record)

    def collect_dispatch_event(self, record: dict[str, Any]) -> None:
        host_range = device_event_host_time_range(record, self.clock_mappers)
        if host_range is None:
            self.stats.skipped_dispatches += 1
            return
        start_time_ns, end_time_ns, time_domain = host_range
        normalized_range = normalized_time_range(start_time_ns, end_time_ns)
        if normalized_range is None:
            self.stats.skipped_dispatches += 1
            return
        start_time_ns, end_time_ns = normalized_range
        physical_device_ordinal, queue_ordinal = queue_key(record)
        self.ensure_queue_tracks(
            physical_device_ordinal, queue_ordinal, record.get("stream_id")
        )
        name = str(record.get("key") or f"dispatch[{record.get('event_id', 0)}]")
        annotations = event_annotations(record)
        annotations["iree_perfetto_time_domain"] = time_domain
        self.append_pending_slice(
            lane_family="dispatch",
            record_family="dispatch",
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            physical_device_ordinal=physical_device_ordinal,
            queue_ordinal=queue_ordinal,
            name=name,
            annotations=annotations,
            flow_ids=self.endpoint_flow_ids(record, "dispatch_event"),
        )
        self.stats.dispatch_slices += 1

    def collect_queue_device_event(self, record: dict[str, Any]) -> None:
        host_range = device_event_host_time_range(record, self.clock_mappers)
        if host_range is None:
            self.stats.skipped_queue_device_events += 1
            return
        start_time_ns, end_time_ns, time_domain = host_range
        normalized_range = normalized_time_range(start_time_ns, end_time_ns)
        if normalized_range is None:
            self.stats.skipped_queue_device_events += 1
            return
        start_time_ns, end_time_ns = normalized_range
        physical_device_ordinal, queue_ordinal = queue_key(record)
        self.ensure_queue_tracks(
            physical_device_ordinal, queue_ordinal, record.get("stream_id")
        )
        annotations = event_annotations(record)
        annotations["iree_perfetto_time_domain"] = time_domain
        self.append_pending_slice(
            lane_family="queue_device",
            record_family="queue_device",
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            physical_device_ordinal=physical_device_ordinal,
            queue_ordinal=queue_ordinal,
            name=f"queue-device:{record.get('op', 'event')}",
            annotations=annotations,
            flow_ids=self.endpoint_flow_ids(record, "queue_device_event"),
        )
        self.stats.queue_device_slices += 1

    def collect_host_execution_event(self, record: dict[str, Any]) -> None:
        start_time_ns = record.get("start_host_time_ns")
        end_time_ns = record.get("end_host_time_ns")
        if start_time_ns is None or end_time_ns is None:
            self.stats.skipped_host_execution_events += 1
            return
        normalized_range = normalized_time_range(
            parse_integer(start_time_ns), parse_integer(end_time_ns)
        )
        if normalized_range is None:
            self.stats.skipped_host_execution_events += 1
            return
        start_time_ns, end_time_ns = normalized_range
        physical_device_ordinal, queue_ordinal = queue_key(record)
        self.ensure_queue_tracks(
            physical_device_ordinal, queue_ordinal, record.get("stream_id")
        )
        name = f"host:{record.get('op', 'event')}"
        if record.get("key"):
            name = str(record["key"])
        annotations = event_annotations(record)
        annotations["iree_perfetto_time_domain"] = record.get(
            "host_time_domain", "iree_host_time_ns"
        )
        self.append_pending_slice(
            lane_family="host_execution",
            record_family="host_execution",
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            physical_device_ordinal=physical_device_ordinal,
            queue_ordinal=queue_ordinal,
            name=name,
            annotations=annotations,
            flow_ids=self.endpoint_flow_ids(record, "host_execution_event"),
        )
        self.stats.host_execution_slices += 1

    def collect_queue_event(self, record: dict[str, Any]) -> None:
        timestamp_ns = record.get("host_time_ns")
        if timestamp_ns is None:
            return
        physical_device_ordinal, queue_ordinal = queue_key(record)
        self.ensure_queue_tracks(
            physical_device_ordinal, queue_ordinal, record.get("stream_id")
        )
        track_uuid = self.tracks.uuid(
            "iree", "queue-events", physical_device_ordinal, queue_ordinal
        )
        event_time_ns = parse_integer(timestamp_ns)
        name = f"queue:{record.get('op', 'event')}"
        annotations = event_annotations(record)
        annotations["iree_perfetto_time_domain"] = record.get(
            "host_time_domain", "iree_host_time_ns"
        )
        submission = submission_key(record)
        flow_ids = (
            list(self.flow_ids_by_submission.get(submission, ()))
            if submission is not None
            else []
        )
        flow_ids.extend(self.endpoint_flow_ids(record, "queue_event"))
        self.timeline_events.append(
            TimelineEvent(
                event_time_ns,
                (2, "queue", physical_device_ordinal, queue_ordinal, event_time_ns),
                lambda timestamp_ns=event_time_ns, track_uuid=track_uuid, name=name, annotations=annotations, flow_ids=tuple(
                    flow_ids
                ): add_instant(
                    self.builder,
                    self.perfetto.track_event,
                    timestamp_ns,
                    track_uuid,
                    name,
                    annotations,
                    flow_ids,
                ),
            )
        )
        self.all_timestamp_ns.append(event_time_ns)
        self.stats.queue_instants += 1

    def collect_memory_event(self, record: dict[str, Any]) -> None:
        timestamp_ns = record.get("host_time_ns")
        if timestamp_ns is None:
            return
        physical_device_ordinal, queue_ordinal = queue_key(record)
        self.ensure_queue_tracks(physical_device_ordinal, queue_ordinal)
        track_uuid = self.tracks.uuid(
            "iree", "memory", physical_device_ordinal, queue_ordinal
        )
        event_time_ns = parse_integer(timestamp_ns)
        name = f"memory:{record.get('event_type', 'event')}"
        annotations = event_annotations(record)
        annotations["iree_perfetto_time_domain"] = record.get(
            "host_time_domain", "iree_host_time_ns"
        )
        self.timeline_events.append(
            TimelineEvent(
                event_time_ns,
                (
                    2,
                    "memory",
                    physical_device_ordinal,
                    queue_ordinal,
                    event_time_ns,
                ),
                lambda timestamp_ns=event_time_ns, track_uuid=track_uuid, name=name, annotations=annotations: add_instant(
                    self.builder,
                    self.perfetto.track_event,
                    timestamp_ns,
                    track_uuid,
                    name,
                    annotations,
                ),
            )
        )
        self.all_timestamp_ns.append(event_time_ns)
        self.stats.memory_instants += 1

    def collect_clock_correlation(self, record: dict[str, Any]) -> None:
        event_time_ns = timestamp_midpoint(record)
        if event_time_ns is None:
            return
        physical_device_ordinal = parse_ordinal(
            record.get("physical_device_ordinal"), 0
        )
        self.ensure_device_track(physical_device_ordinal)
        track_uuid = self.tracks.uuid("iree", "device", physical_device_ordinal)
        annotations = event_annotations(record)
        self.timeline_events.append(
            TimelineEvent(
                event_time_ns,
                (2, "clock", physical_device_ordinal, event_time_ns),
                lambda timestamp_ns=event_time_ns, track_uuid=track_uuid, annotations=annotations: add_instant(
                    self.builder,
                    self.perfetto.track_event,
                    timestamp_ns,
                    track_uuid,
                    "clock correlation",
                    annotations,
                ),
            )
        )
        self.all_timestamp_ns.append(event_time_ns)
        self.stats.clock_instants += 1

    def collect_diagnostic(self, record: dict[str, Any]) -> None:
        annotations = event_annotations(record)
        self.timeline_events.append(
            TimelineEvent(
                0,
                (2, "diagnostic", record.get("source_record_index", 0)),
                lambda timestamp_ns=0, track_uuid=self.diagnostics_uuid, annotations=annotations: add_instant(
                    self.builder,
                    self.perfetto.track_event,
                    timestamp_ns,
                    track_uuid,
                    str(record.get("code", "diagnostic")),
                    annotations,
                ),
            )
        )
        self.stats.diagnostic_instants += 1

    def emit_pending_slices(self) -> None:
        # Allocate lanes after collecting all slices so large traces avoid the
        # quadratic "scan every lane for every slice" case and remain correct
        # even when records arrive out of timestamp order.
        for pending_slice in sorted(
            self.pending_slices,
            key=lambda pending_slice: (
                pending_slice.lane_family,
                pending_slice.physical_device_ordinal,
                pending_slice.queue_ordinal,
                pending_slice.start_time_ns,
                pending_slice.end_time_ns,
                pending_slice.sequence_index,
            ),
        ):
            track_uuid = self.define_slice_track(pending_slice)
            self.append_slice_events(
                record_family=pending_slice.record_family,
                start_time_ns=pending_slice.start_time_ns,
                end_time_ns=pending_slice.end_time_ns,
                physical_device_ordinal=pending_slice.physical_device_ordinal,
                queue_ordinal=pending_slice.queue_ordinal,
                track_uuid=track_uuid,
                name=pending_slice.name,
                annotations=pending_slice.annotations,
                flow_ids=pending_slice.flow_ids,
            )

    def define_slice_track(self, pending_slice: PendingSlice) -> int:
        allocator_key = (
            pending_slice.physical_device_ordinal,
            pending_slice.queue_ordinal,
        )
        if pending_slice.lane_family == "dispatch":
            lane_index = self.dispatch_lane_allocators[allocator_key].allocate(
                pending_slice.start_time_ns, pending_slice.end_time_ns
            )
            return self.define_dispatch_lane(
                pending_slice.physical_device_ordinal,
                pending_slice.queue_ordinal,
                lane_index,
            )
        if pending_slice.lane_family == "queue_device":
            lane_index = self.queue_device_lane_allocators[allocator_key].allocate(
                pending_slice.start_time_ns, pending_slice.end_time_ns
            )
            return self.define_queue_device_lane(
                pending_slice.physical_device_ordinal,
                pending_slice.queue_ordinal,
                lane_index,
            )
        if pending_slice.lane_family == "host_execution":
            lane_index = self.host_execution_lane_allocators[allocator_key].allocate(
                pending_slice.start_time_ns, pending_slice.end_time_ns
            )
            return self.define_host_execution_lane(
                pending_slice.physical_device_ordinal,
                pending_slice.queue_ordinal,
                lane_index,
            )
        raise ValueError(f"unknown slice lane family: {pending_slice.lane_family}")

    def emit_queue_allocation_counters(self) -> None:
        queue_allocation_bytes: dict[tuple[int, int], int] = collections.defaultdict(
            int
        )
        for record in sorted(
            (
                record
                for record in self.records
                if record.get("record_type") == "memory_event"
            ),
            key=lambda item: parse_integer(item.get("host_time_ns", 0)),
        ):
            event_type = record.get("event_type")
            if event_type not in ("queue_alloca", "queue_dealloca"):
                continue
            if "host_time_ns" not in record:
                continue
            physical_device_ordinal, queue_ordinal = queue_key(record)
            self.ensure_queue_tracks(physical_device_ordinal, queue_ordinal)
            key = (physical_device_ordinal, queue_ordinal)
            length = parse_integer(record.get("length", 0))
            if event_type == "queue_alloca":
                queue_allocation_bytes[key] += length
            else:
                queue_allocation_bytes[key] = max(
                    0, queue_allocation_bytes[key] - length
                )
            event_time_ns = parse_integer(record["host_time_ns"])
            track_uuid = self.tracks.uuid(
                "iree",
                "queue-allocation-bytes",
                physical_device_ordinal,
                queue_ordinal,
            )
            value = queue_allocation_bytes[key]
            self.timeline_events.append(
                TimelineEvent(
                    event_time_ns,
                    (
                        2,
                        "counter",
                        physical_device_ordinal,
                        queue_ordinal,
                        event_time_ns,
                    ),
                    lambda timestamp_ns=event_time_ns, track_uuid=track_uuid, value=value: add_counter(
                        self.builder,
                        self.perfetto.track_event,
                        timestamp_ns,
                        track_uuid,
                        value,
                    ),
                )
            )
            self.all_timestamp_ns.append(event_time_ns)
            self.stats.counter_samples += 1

    def emit_timeline_events(self) -> None:
        trace_epoch_ns = min(self.all_timestamp_ns) if self.all_timestamp_ns else 0
        for event in sorted(
            self.timeline_events, key=lambda item: (item.timestamp_ns, item.sort_key)
        ):
            event.callback(timestamp_ns=max(0, event.timestamp_ns - trace_epoch_ns))


def build_trace(
    records: list[dict[str, Any]], perfetto: PerfettoImports
) -> tuple[bytes, ConversionStats]:
    return PerfettoTraceConverter(records, perfetto).build()


def render_perfetto(
    records: list[dict[str, Any]], output_path: Path, perfetto: PerfettoImports
) -> ConversionStats:
    trace_bytes, stats = build_trace(records, perfetto)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(trace_bytes)
    stats.output_byte_count = len(trace_bytes)
    return stats


def render(records: list[dict[str, Any]], output_path: Path) -> ConversionStats:
    return render_perfetto(records, output_path, import_perfetto())


def summary_fields(stats: ConversionStats) -> list[tuple[str, Any]]:
    return [
        ("records", stats.records),
        ("dispatch_slices", stats.dispatch_slices),
        ("queue_device_slices", stats.queue_device_slices),
        ("host_execution_slices", stats.host_execution_slices),
        ("queue_instants", stats.queue_instants),
        ("memory_instants", stats.memory_instants),
        ("clock_instants", stats.clock_instants),
        ("counter_samples", stats.counter_samples),
        ("relationship_flows", stats.relationship_flows),
        ("skipped_dispatches", stats.skipped_dispatches),
        ("skipped_queue_device_events", stats.skipped_queue_device_events),
        ("skipped_host_execution_events", stats.skipped_host_execution_events),
    ]
