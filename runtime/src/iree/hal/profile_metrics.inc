// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Built-in HAL profile device metric descriptors.
//
// IREE_HAL_PROFILE_BUILTIN_METRIC(enum_name, metric_id, name, unit, value_kind,
//                                 semantic, plot_hint, description)

IREE_HAL_PROFILE_BUILTIN_METRIC(CLOCK_COMPUTE_CURRENT, 1u,
                                "clock.compute.current",
                                IREE_HAL_PROFILE_METRIC_UNIT_HERTZ,
                                IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
                                IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT,
                                IREE_HAL_PROFILE_METRIC_PLOT_HINT_FREQUENCY,
                                "Current compute clock frequency.")

IREE_HAL_PROFILE_BUILTIN_METRIC(CLOCK_MEMORY_CURRENT, 2u,
                                "clock.memory.current",
                                IREE_HAL_PROFILE_METRIC_UNIT_HERTZ,
                                IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
                                IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT,
                                IREE_HAL_PROFILE_METRIC_PLOT_HINT_FREQUENCY,
                                "Current memory clock frequency.")

IREE_HAL_PROFILE_BUILTIN_METRIC(
    TEMPERATURE_EDGE, 3u, "temperature.edge",
    IREE_HAL_PROFILE_METRIC_UNIT_MILLIDEGREES_CELSIUS,
    IREE_HAL_PROFILE_METRIC_VALUE_KIND_I64,
    IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT,
    IREE_HAL_PROFILE_METRIC_PLOT_HINT_TEMPERATURE, "Edge temperature.")

IREE_HAL_PROFILE_BUILTIN_METRIC(
    TEMPERATURE_HOTSPOT, 4u, "temperature.hotspot",
    IREE_HAL_PROFILE_METRIC_UNIT_MILLIDEGREES_CELSIUS,
    IREE_HAL_PROFILE_METRIC_VALUE_KIND_I64,
    IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT,
    IREE_HAL_PROFILE_METRIC_PLOT_HINT_TEMPERATURE, "Hotspot temperature.")

IREE_HAL_PROFILE_BUILTIN_METRIC(
    TEMPERATURE_MEMORY, 5u, "temperature.memory",
    IREE_HAL_PROFILE_METRIC_UNIT_MILLIDEGREES_CELSIUS,
    IREE_HAL_PROFILE_METRIC_VALUE_KIND_I64,
    IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT,
    IREE_HAL_PROFILE_METRIC_PLOT_HINT_TEMPERATURE, "Memory temperature.")

IREE_HAL_PROFILE_BUILTIN_METRIC(POWER_SOCKET, 6u, "power.socket",
                                IREE_HAL_PROFILE_METRIC_UNIT_MICROWATTS,
                                IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
                                IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT,
                                IREE_HAL_PROFILE_METRIC_PLOT_HINT_POWER,
                                "Socket power draw.")

IREE_HAL_PROFILE_BUILTIN_METRIC(ENERGY_SOCKET, 7u, "energy.socket",
                                IREE_HAL_PROFILE_METRIC_UNIT_MICROJOULES,
                                IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
                                IREE_HAL_PROFILE_METRIC_SEMANTIC_CUMULATIVE,
                                IREE_HAL_PROFILE_METRIC_PLOT_HINT_ENERGY,
                                "Socket energy counter.")

IREE_HAL_PROFILE_BUILTIN_METRIC(ACTIVITY_COMPUTE, 8u, "activity.compute",
                                IREE_HAL_PROFILE_METRIC_UNIT_MILLIPERCENT,
                                IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
                                IREE_HAL_PROFILE_METRIC_SEMANTIC_AVERAGE,
                                IREE_HAL_PROFILE_METRIC_PLOT_HINT_PERCENTAGE,
                                "Compute activity over the sampling window.")

IREE_HAL_PROFILE_BUILTIN_METRIC(ACTIVITY_MEMORY, 9u, "activity.memory",
                                IREE_HAL_PROFILE_METRIC_UNIT_MILLIPERCENT,
                                IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
                                IREE_HAL_PROFILE_METRIC_SEMANTIC_AVERAGE,
                                IREE_HAL_PROFILE_METRIC_PLOT_HINT_PERCENTAGE,
                                "Memory activity over the sampling window.")

IREE_HAL_PROFILE_BUILTIN_METRIC(MEMORY_LOCAL_USED, 10u, "memory.local.used",
                                IREE_HAL_PROFILE_METRIC_UNIT_BYTES,
                                IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
                                IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT,
                                IREE_HAL_PROFILE_METRIC_PLOT_HINT_MEMORY,
                                "Local device memory currently used.")

IREE_HAL_PROFILE_BUILTIN_METRIC(MEMORY_LOCAL_TOTAL, 11u, "memory.local.total",
                                IREE_HAL_PROFILE_METRIC_UNIT_BYTES,
                                IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
                                IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT,
                                IREE_HAL_PROFILE_METRIC_PLOT_HINT_MEMORY,
                                "Local device memory capacity.")

IREE_HAL_PROFILE_BUILTIN_METRIC(
    MEMORY_TRAFFIC_READ, 12u, "memory.traffic.read",
    IREE_HAL_PROFILE_METRIC_UNIT_BYTES_PER_SECOND,
    IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
    IREE_HAL_PROFILE_METRIC_SEMANTIC_AVERAGE,
    IREE_HAL_PROFILE_METRIC_PLOT_HINT_BANDWIDTH,
    "Memory read bandwidth over the sampling window.")

IREE_HAL_PROFILE_BUILTIN_METRIC(
    MEMORY_TRAFFIC_WRITE, 13u, "memory.traffic.write",
    IREE_HAL_PROFILE_METRIC_UNIT_BYTES_PER_SECOND,
    IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
    IREE_HAL_PROFILE_METRIC_SEMANTIC_AVERAGE,
    IREE_HAL_PROFILE_METRIC_PLOT_HINT_BANDWIDTH,
    "Memory write bandwidth over the sampling window.")

IREE_HAL_PROFILE_BUILTIN_METRIC(THROTTLE_STATUS, 14u, "throttle.status",
                                IREE_HAL_PROFILE_METRIC_UNIT_BITFIELD,
                                IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
                                IREE_HAL_PROFILE_METRIC_SEMANTIC_STATE,
                                IREE_HAL_PROFILE_METRIC_PLOT_HINT_NUMBER,
                                "Producer-defined throttling status bitfield.")

IREE_HAL_PROFILE_BUILTIN_METRIC(PERFORMANCE_TIER, 15u, "performance.tier",
                                IREE_HAL_PROFILE_METRIC_UNIT_ENUM,
                                IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64,
                                IREE_HAL_PROFILE_METRIC_SEMANTIC_STATE,
                                IREE_HAL_PROFILE_METRIC_PLOT_HINT_NUMBER,
                                "Producer-defined performance tier enumerant.")
