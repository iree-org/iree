"""Shared ireeperf-jsonl loading and timeline helpers."""

from __future__ import annotations

import collections
import dataclasses
import decimal
import json
import sys
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 12
UINT64_MASK = (1 << 64) - 1
UINT32_MAX = (1 << 32) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


@dataclasses.dataclass
class DeviceClockMapper:
    """Maps raw device ticks to IREE host time nanoseconds."""

    first_device_tick: int
    first_host_time_ns: int
    last_device_tick: int
    last_host_time_ns: int

    def host_time_from_device_tick(self, device_tick: int) -> int:
        tick_delta = self.last_device_tick - self.first_device_tick
        if tick_delta == 0:
            return self.first_host_time_ns + (device_tick - self.first_device_tick)
        host_delta = self.last_host_time_ns - self.first_host_time_ns
        relative_tick_delta = device_tick - self.first_device_tick
        return self.first_host_time_ns + round_ratio(
            relative_tick_delta * host_delta, tick_delta
        )


def parse_integer(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, decimal.Decimal):
        return int(value.to_integral_value(rounding=decimal.ROUND_HALF_UP))
    return int(value)


def parse_ordinal(value: Any, default: int = -1) -> int:
    ordinal = parse_integer(value, default)
    return default if ordinal == UINT32_MAX else ordinal


def round_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError(f"invalid ratio denominator: {denominator}")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def timestamp_midpoint(record: dict[str, Any]) -> int | None:
    begin_ns = record.get("host_time_begin_ns")
    end_ns = record.get("host_time_end_ns")
    if begin_ns is not None and end_ns is not None:
        return (parse_integer(begin_ns) + parse_integer(end_ns)) // 2
    host_time_ns = record.get("host_time_ns")
    if host_time_ns is not None:
        return parse_integer(host_time_ns)
    return None


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path == "-":
        lines: Iterable[str] = sys.stdin
        source_name = "<stdin>"
    else:
        lines = Path(path).open("r", encoding="utf-8")
        source_name = path
    try:
        for line_number, line in enumerate(lines, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                records.append(
                    json.loads(
                        stripped_line,
                        parse_float=decimal.Decimal,
                        parse_int=int,
                    )
                )
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"{source_name}:{line_number}: invalid JSONL record: {error}"
                ) from error
    finally:
        if path != "-":
            lines.close()
    return records


def validate_schema(records: list[dict[str, Any]]) -> None:
    schema_records = [
        record for record in records if record.get("record_type") == "schema"
    ]
    if not schema_records:
        raise SystemExit("input does not contain an ireeperf-jsonl schema record")
    schema_record = schema_records[0]
    if schema_record.get("format") != "ireeperf-jsonl":
        raise SystemExit(f"unsupported input format: {schema_record.get('format')!r}")
    schema_version = parse_integer(schema_record.get("schema_version", 0))
    if schema_version != SCHEMA_VERSION:
        raise SystemExit(
            f"unsupported ireeperf-jsonl schema version {schema_version}; "
            f"expected {SCHEMA_VERSION}"
        )


def build_device_clock_mappers(
    records: list[dict[str, Any]],
) -> dict[int, DeviceClockMapper]:
    samples_by_device: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        if record.get("record_type") != "clock_correlation":
            continue
        if "physical_device_ordinal" not in record or "device_tick" not in record:
            continue
        host_time_ns = timestamp_midpoint(record)
        if host_time_ns is None:
            continue
        samples_by_device[parse_integer(record["physical_device_ordinal"])].append(
            record
        )

    mappers: dict[int, DeviceClockMapper] = {}
    for physical_device_ordinal, samples in samples_by_device.items():
        samples.sort(key=lambda sample: parse_integer(sample["device_tick"]))
        first = samples[0]
        last = samples[-1]
        first_host_time_ns = timestamp_midpoint(first)
        last_host_time_ns = timestamp_midpoint(last)
        if first_host_time_ns is None or last_host_time_ns is None:
            continue
        mappers[physical_device_ordinal] = DeviceClockMapper(
            first_device_tick=parse_integer(first["device_tick"]),
            first_host_time_ns=first_host_time_ns,
            last_device_tick=parse_integer(last["device_tick"]),
            last_host_time_ns=last_host_time_ns,
        )
    return mappers


def device_event_host_time_range(
    record: dict[str, Any], clock_mappers: dict[int, DeviceClockMapper]
) -> tuple[int, int, str] | None:
    if not record.get("valid", True):
        return None
    if "start_tick" not in record or "end_tick" not in record:
        return None
    physical_device_ordinal = parse_ordinal(record.get("physical_device_ordinal"), 0)
    start_tick = parse_integer(record["start_tick"])
    end_tick = parse_integer(record["end_tick"])
    mapper = clock_mappers.get(physical_device_ordinal)
    if mapper is not None:
        return (
            mapper.host_time_from_device_tick(start_tick),
            mapper.host_time_from_device_tick(end_tick),
            "iree_host_time_from_device_clock_fit",
        )
    if record.get("derived_time_available", False):
        return (
            parse_integer(record.get("start_driver_host_cpu_time_ns")),
            parse_integer(record.get("end_driver_host_cpu_time_ns")),
            str(record.get("derived_time_domain", "driver_host_cpu_timestamp_ns")),
        )
    return None


def normalized_time_range(
    start_time_ns: int, end_time_ns: int
) -> tuple[int, int] | None:
    if end_time_ns < start_time_ns:
        return None
    if end_time_ns == start_time_ns:
        end_time_ns += 1
    return start_time_ns, end_time_ns


def event_annotations(record: dict[str, Any]) -> dict[str, Any]:
    skipped_keys = {"schema_version", "record_type", "key", "name"}
    return {
        key: value
        for key, value in record.items()
        if key not in skipped_keys
        and isinstance(value, (bool, int, float, decimal.Decimal, str, list))
    }


def queue_key(record: dict[str, Any]) -> tuple[int, int]:
    return (
        parse_ordinal(record.get("physical_device_ordinal"), 0),
        parse_ordinal(record.get("queue_ordinal"), -1),
    )


def submission_key(record: dict[str, Any]) -> tuple[int, int, int, int] | None:
    submission_id = parse_integer(record.get("submission_id", 0))
    if submission_id == 0:
        return None
    physical_device_ordinal, queue_ordinal = queue_key(record)
    stream_id = parse_integer(record.get("stream_id", 0))
    return (physical_device_ordinal, queue_ordinal, stream_id, submission_id)


def event_endpoint_key(
    record: dict[str, Any], endpoint_type: str
) -> tuple[str, int, int, int, int] | None:
    event_id = parse_integer(record.get("event_id", 0))
    if event_id == 0:
        return None
    physical_device_ordinal, queue_ordinal = queue_key(record)
    stream_id = parse_integer(record.get("stream_id", 0))
    return (endpoint_type, physical_device_ordinal, queue_ordinal, stream_id, event_id)


def relationship_endpoint_key(
    record: dict[str, Any], prefix: str
) -> tuple[str, int, int, int, int] | None:
    endpoint_type = str(record.get(f"{prefix}_type", ""))
    endpoint_id = parse_integer(record.get(f"{prefix}_id", 0))
    if not endpoint_type or endpoint_id == 0:
        return None
    physical_device_ordinal, queue_ordinal = queue_key(record)
    stream_id = parse_integer(record.get("stream_id", 0))
    return (
        endpoint_type,
        physical_device_ordinal,
        queue_ordinal,
        stream_id,
        endpoint_id,
    )
