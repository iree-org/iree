// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_TOPOLOGY_BUILDER_H_
#define IREE_HAL_TOPOLOGY_BUILDER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Scheduling word (lo) layout constants
//===----------------------------------------------------------------------===//

#define IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_SHIFT 0
#define IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_MASK 0x3ull

#define IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_SHIFT 2
#define IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_MASK 0x3ull

#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_SHIFT 4
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_MASK 0x3ull

#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_SHIFT 6
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_MASK 0x3ull

#define IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_SHIFT 8
#define IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_MASK 0xFFFFull

#define IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_SHIFT 24
#define IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_MASK 0xFull

#define IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_SHIFT 28
#define IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_MASK 0xFull

#define IREE_HAL_TOPOLOGY_EDGE_COPY_COST_SHIFT 32
#define IREE_HAL_TOPOLOGY_EDGE_COPY_COST_MASK 0xFull

#define IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_SHIFT 36
#define IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_MASK 0xFull

#define IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_SHIFT 40
#define IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_MASK 0xFull

#define IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_SHIFT 44
#define IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_MASK 0x7ull

//===----------------------------------------------------------------------===//
// Interop word (hi) layout constants
//===----------------------------------------------------------------------===//

#define IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TYPES_SHIFT 0
#define IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TYPES_MASK 0xFFull

#define IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TYPES_SHIFT 8
#define IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TYPES_MASK 0xFFull

#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_SHIFT 16
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_MASK 0xFFull

#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_SHIFT 24
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_MASK 0xFFull

//===----------------------------------------------------------------------===//
// Scheduling word (lo) setters
//===----------------------------------------------------------------------===//

// Sets the wait interop mode in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_wait_mode(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_interop_mode_t mode) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_MASK
            << IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_SHIFT);
  word |= ((uint64_t)mode & IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_SHIFT;
  return word;
}

// Sets the signal interop mode in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_signal_mode(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_interop_mode_t mode) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_MASK
            << IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_SHIFT);
  word |= ((uint64_t)mode & IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_SHIFT;
  return word;
}

// Sets the buffer read interop mode in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_buffer_read_mode(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_interop_mode_t mode) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_MASK
            << IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_SHIFT);
  word |= ((uint64_t)mode & IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_SHIFT;
  return word;
}

// Sets the buffer write interop mode in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_buffer_write_mode(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_interop_mode_t mode) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_MASK
            << IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_SHIFT);
  word |= ((uint64_t)mode & IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_SHIFT;
  return word;
}

// Sets capability flags in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_capability_flags(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_capability_t flags) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_MASK
            << IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_SHIFT);
  word |= ((uint64_t)flags & IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_SHIFT;
  return word;
}

// Sets wait cost in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_wait_cost(
    iree_hal_topology_edge_scheduling_word_t word, uint8_t cost) {
  cost = iree_min(cost, 15);  // Clamp to 4 bits.
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_MASK
            << IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_SHIFT);
  word |= ((uint64_t)cost & IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_SHIFT;
  return word;
}

// Sets signal cost in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_signal_cost(
    iree_hal_topology_edge_scheduling_word_t word, uint8_t cost) {
  cost = iree_min(cost, 15);  // Clamp to 4 bits.
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_MASK
            << IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_SHIFT);
  word |= ((uint64_t)cost & IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_SHIFT;
  return word;
}

// Sets copy/transfer cost in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_copy_cost(
    iree_hal_topology_edge_scheduling_word_t word, uint8_t cost) {
  cost = iree_min(cost, 15);  // Clamp to 4 bits.
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_COPY_COST_MASK
            << IREE_HAL_TOPOLOGY_EDGE_COPY_COST_SHIFT);
  word |= ((uint64_t)cost & IREE_HAL_TOPOLOGY_EDGE_COPY_COST_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_COPY_COST_SHIFT;
  return word;
}

// Sets latency class in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_latency_class(
    iree_hal_topology_edge_scheduling_word_t word, uint8_t latency_class) {
  latency_class = iree_min(latency_class, 15);  // Clamp to 4 bits.
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_MASK
            << IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_SHIFT);
  word |= ((uint64_t)latency_class & IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_SHIFT;
  return word;
}

// Sets NUMA distance in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_numa_distance(
    iree_hal_topology_edge_scheduling_word_t word, uint8_t distance) {
  distance = iree_min(distance, 15);  // Clamp to 4 bits.
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_MASK
            << IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_SHIFT);
  word |= ((uint64_t)distance & IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_SHIFT;
  return word;
}

// Sets the link class in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_link_class(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_link_class_t link_class) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_MASK
            << IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_SHIFT);
  word |= ((uint64_t)link_class & IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_SHIFT;
  return word;
}

//===----------------------------------------------------------------------===//
// Interop word (hi) setters
//===----------------------------------------------------------------------===//

// Sets semaphore import handle types in an interop word.
static inline iree_hal_topology_edge_interop_word_t
iree_hal_topology_edge_set_semaphore_import_types(
    iree_hal_topology_edge_interop_word_t word,
    iree_hal_topology_handle_type_t types) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TYPES_MASK
            << IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TYPES_SHIFT);
  word |= ((uint64_t)types & IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TYPES_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TYPES_SHIFT;
  return word;
}

// Sets semaphore export handle types in an interop word.
static inline iree_hal_topology_edge_interop_word_t
iree_hal_topology_edge_set_semaphore_export_types(
    iree_hal_topology_edge_interop_word_t word,
    iree_hal_topology_handle_type_t types) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TYPES_MASK
            << IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TYPES_SHIFT);
  word |= ((uint64_t)types & IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TYPES_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TYPES_SHIFT;
  return word;
}

// Sets buffer import handle types in an interop word.
static inline iree_hal_topology_edge_interop_word_t
iree_hal_topology_edge_set_buffer_import_types(
    iree_hal_topology_edge_interop_word_t word,
    iree_hal_topology_handle_type_t types) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_MASK
            << IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_SHIFT);
  word |= ((uint64_t)types & IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_SHIFT;
  return word;
}

// Sets buffer export handle types in an interop word.
static inline iree_hal_topology_edge_interop_word_t
iree_hal_topology_edge_set_buffer_export_types(
    iree_hal_topology_edge_interop_word_t word,
    iree_hal_topology_handle_type_t types) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_MASK
            << IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_SHIFT);
  word |= ((uint64_t)types & IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_SHIFT;
  return word;
}

//===----------------------------------------------------------------------===//
// iree_hal_topology_builder_t
//===----------------------------------------------------------------------===//

// Builder for constructing immutable topologies.
//
// The builder provides a safe way to incrementally construct a topology
// with validation. Once built, the resulting topology is immutable.
// The builder embeds a fixed-size topology and can be stack-allocated.
//
// Usage:
//   iree_hal_topology_builder_t builder;
//   iree_hal_topology_builder_initialize(&builder, device_count);
//
//   // Set edges (self-edges are automatically initialized).
//   iree_hal_topology_builder_set_edge(&builder, 0, 1, edge_0_to_1);
//   iree_hal_topology_builder_set_edge(&builder, 1, 0, edge_1_to_0);
//
//   // Build immutable topology.
//   iree_hal_topology_t topology;
//   iree_hal_topology_builder_finalize(&builder, &topology);
//   // No cleanup needed.
//
// Thread safety: Builders are NOT thread-safe during construction.
// The immutable topology they produce supports lock-free concurrent queries.
typedef struct iree_hal_topology_builder_t {
  // Embedded topology being constructed.
  iree_hal_topology_t topology;

  // Tracking which edges have been explicitly set.
  // Used for validation during build.
  bool edges_set[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT *
                 IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
} iree_hal_topology_builder_t;

//===----------------------------------------------------------------------===//
// Edge construction helpers
//===----------------------------------------------------------------------===//

// Returns an optimal self-edge for a device.
// Self-edges represent a device's relationship with itself and should
// have all optimal settings (native access, zero cost, etc.).
IREE_API_EXPORT iree_hal_topology_edge_t iree_hal_topology_edge_make_self(void);

// Returns a default cross-driver edge.
// This represents the baseline capabilities between devices from different
// drivers, typically requiring import/export or host staging.
IREE_API_EXPORT iree_hal_topology_edge_t
iree_hal_topology_edge_make_cross_driver(void);

//===----------------------------------------------------------------------===//
// Topology builder
//===----------------------------------------------------------------------===//

// Initializes a topology builder for the specified number of devices.
// The builder should be stack-allocated and initialized before use.
// Self-edges are automatically initialized to optimal values.
IREE_API_EXPORT void iree_hal_topology_builder_initialize(
    iree_hal_topology_builder_t* builder, uint32_t device_count);

// Sets the edge from src_ordinal to dst_ordinal.
// Self-edges (src == dst) must use iree_hal_topology_edge_make_self().
IREE_API_EXPORT iree_status_t iree_hal_topology_builder_set_edge(
    iree_hal_topology_builder_t* builder, uint32_t src_ordinal,
    uint32_t dst_ordinal, iree_hal_topology_edge_t edge);

// Sets the NUMA node for a device.
IREE_API_EXPORT iree_status_t iree_hal_topology_builder_set_numa_node(
    iree_hal_topology_builder_t* builder, uint32_t device_ordinal,
    uint8_t numa_node);

// Builds the immutable topology into |out_topology|.
// The topology is copied from the builder's embedded topology.
// The builder can be reused or discarded after this call.
// Returns an error if validation fails (missing edges, invalid symmetry, etc.).
IREE_API_EXPORT iree_status_t iree_hal_topology_builder_finalize(
    iree_hal_topology_builder_t* builder, iree_hal_topology_t* out_topology);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_TOPOLOGY_BUILDER_H_
