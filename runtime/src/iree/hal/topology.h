// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_TOPOLOGY_H_
#define IREE_HAL_TOPOLOGY_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Forward declarations.
typedef struct iree_hal_device_t iree_hal_device_t;
typedef struct iree_hal_device_capabilities_t iree_hal_device_capabilities_t;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Maximum number of devices supported in a single topology.
#define IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT 32

// Invalid device ordinal sentinel value.
#define IREE_HAL_TOPOLOGY_DEVICE_ORDINAL_INVALID UINT32_MAX

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Bitmap type for device compatibility masks.
// Sized based on max device count for efficient bitwise operations.
// Could be extended to a cpu_set-like mechanism if needed but at a big storage
// cost and we'd likely want to rework things anyway. 64 logical devices is a
// lot, though, so we're probably fine.
#if IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT <= 32
typedef uint32_t iree_hal_topology_device_bitmap_t;
#else
typedef uint64_t iree_hal_topology_device_bitmap_t;
#endif

// 64-bit packed edge descriptor encoding directional device capabilities.
//
// Each edge in the topology matrix describes the relationship from a source
// device to a destination device. The edge encodes:
//  - Interop modes: How synchronization and buffers can be shared
//  - External handle types: What import/export mechanisms are supported
//  - Capability flags: Hardware features available on this path
//  - Cost metrics: Relative costs for operations (0-15 scale)
//  - Link class: Physical interconnect type
//
// Bitfield layout (64 bits total):
//  Bits 0-1:   wait_mode (2 bits) - how dst can wait on src semaphores
//  Bits 2-3:   signal_mode (2 bits) - how src can signal to dst
//  Bits 4-5:   buffer_read_mode (2 bits) - how dst can read src buffers
//  Bits 6-7:   buffer_write_mode (2 bits) - how dst can write src buffers
//  Bits 8-11:  semaphore_import_types (4 bits) - handle types dst can import
//  Bits 12-15: semaphore_export_types (4 bits) - handle types src can export
//  Bits 16-19: buffer_import_types (4 bits) - buffer types dst can import
//  Bits 20-23: buffer_export_types (4 bits) - buffer types src can export
//  Bits 24-34: capability_flags (11 bits) - hardware capabilities
//  Bits 35-38: wait_cost (4 bits, 0-15) - relative cost to wait
//  Bits 39-42: signal_cost (4 bits, 0-15) - relative cost to signal
//  Bits 43-46: copy_cost (4 bits, 0-15) - relative cost to copy data
//  Bits 47-50: latency_class (4 bits, 0-15) - latency category
//  Bits 51-54: numa_distance (4 bits, 0-15) - NUMA distance
//  Bits 55-57: link_class (3 bits) - physical link type
//  Bits 58-63: reserved (6 bits) - must be zero
typedef uint64_t iree_hal_topology_edge_t;

// Interop modes describing how resources can be shared between devices.
// Lower values indicate more efficient sharing.
enum iree_hal_topology_interop_mode_bits_t {
  // Native access within the same runtime domain.
  // Resources can be directly accessed without any translation.
  IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE = 0,
  // Import/export via external handles (fd, HANDLE, etc.).
  // Requires handle export from source and import to destination.
  IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT = 1,
  // Host staging required for transfer.
  // Data must be copied through host memory.
  IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY = 2,
  // Not supported - no interop possible.
  // Operations will fail if attempted.
  IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE = 3,
};
typedef uint8_t iree_hal_topology_interop_mode_t;

// Physical link classes between devices.
// Used to infer performance characteristics and capabilities.
enum iree_hal_topology_link_class_bits_t {
  // Same die/chip - highest bandwidth, lowest latency.
  // Example: CPU cores, integrated GPU, chiplets.
  IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE = 0,
  // High-bandwidth interconnect (NVLink, Infinity Fabric, etc.).
  // Typically provides cache coherence and high bandwidth.
  IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF = 1,
  // PCIe within same root complex.
  // Standard expansion bus, good bandwidth.
  IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT = 2,
  // PCIe across root complexes (cross-socket).
  // May require QPI/UPI traversal, higher latency.
  IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_CROSS_ROOT = 3,
  // Host memory staging required.
  // No direct device-to-device path.
  IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED = 4,
  // Network fabric (RDMA, RoCE, etc.).
  // For distributed/clustered systems.
  IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC = 5,
  // Other/unknown interconnect.
  IREE_HAL_TOPOLOGY_LINK_CLASS_OTHER = 6,
  // Isolated - no communication possible (MIG, SR-IOV).
  // Devices cannot interact even through host.
  IREE_HAL_TOPOLOGY_LINK_CLASS_ISOLATED = 7,
};
typedef uint8_t iree_hal_topology_link_class_t;

// External handle type bits for import/export operations.
// These map to platform-specific handle types.
enum iree_hal_topology_handle_type_bits_t {
  // No external handle support.
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_NONE = 0,
  // Native handle for same-driver devices.
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE = 1u << 0,
  // POSIX file descriptor (Linux/Android).
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD = 1u << 1,
  // Win32 HANDLE (Windows).
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_WIN32 = 1u << 2,
  // DMA-BUF file descriptor (Linux).
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF = 1u << 3,
};
typedef uint8_t iree_hal_topology_handle_type_t;

// Capability flags for edge features.
enum iree_hal_topology_capability_bits_t {
  // No special capabilities.
  IREE_HAL_TOPOLOGY_CAPABILITY_NONE = 0,
  // Devices share the same runtime domain (driver instance).
  IREE_HAL_TOPOLOGY_CAPABILITY_SAME_RUNTIME_DOMAIN = 1u << 0,
  // Unified memory accessible by both devices.
  IREE_HAL_TOPOLOGY_CAPABILITY_UNIFIED_MEMORY = 1u << 1,
  // Cache coherent between devices (no explicit flush needed).
  IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT = 1u << 2,
  // Host coherent (CPU can access without explicit sync).
  IREE_HAL_TOPOLOGY_CAPABILITY_HOST_COHERENT = 1u << 3,
  // Direct peer-to-peer copy supported.
  IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY = 1u << 4,
  // Concurrent access safe (no mutual exclusion needed).
  IREE_HAL_TOPOLOGY_CAPABILITY_CONCURRENT_SAFE = 1u << 5,
  // Device-scope atomics work across link.
  IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE = 1u << 6,
  // System-scope atomics work across link.
  IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM = 1u << 7,
  // Timeline semaphores supported.
  IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE = 1u << 8,
  // Binary semaphore emulation required.
  IREE_HAL_TOPOLOGY_CAPABILITY_BINARY_SEMAPHORE_ONLY = 1u << 9,
  // Reserved for future use.
  IREE_HAL_TOPOLOGY_CAPABILITY_RESERVED = 1u << 10,
};
typedef uint16_t iree_hal_topology_capability_t;

//===----------------------------------------------------------------------===//
// iree_hal_resource_origin_t
//===----------------------------------------------------------------------===//

// Unified resource origin for fast compatibility checks.
//
// This compact 12-byte structure is embedded in resources (semaphores, buffers)
// to enable ultra-fast (1-3ns) compatibility queries. The self_edge encodes
// the owning device's capabilities, while topology_index identifies the device
// within its topology group.
typedef struct iree_hal_resource_origin_t {
  // Device capabilities encoded as a self-edge.
  // This is the edge[i][i] diagonal entry from the topology matrix.
  iree_hal_topology_edge_t self_edge;  // 8 bytes

  // Index of the device in the topology (0 to device_count-1).
  // IREE_HAL_TOPOLOGY_DEVICE_ORDINAL_INVALID if not in a group.
  uint32_t topology_index;  // 4 bytes
} iree_hal_resource_origin_t;

// Returns an undefined resource origin.
static inline iree_hal_resource_origin_t iree_hal_resource_origin_undefined(
    void) {
  iree_hal_resource_origin_t origin = {
      .self_edge = 0,
      .topology_index = IREE_HAL_TOPOLOGY_DEVICE_ORDINAL_INVALID,
  };
  return origin;
}

//===----------------------------------------------------------------------===//
// iree_hal_topology_t
//===----------------------------------------------------------------------===//

// Immutable device topology describing relationships between devices.
//
// The topology is a pure data structure (POD) that encodes a directed graph
// of device relationships. Each edge in the graph describes how one device
// can interact with another, including synchronization modes, buffer sharing
// capabilities, and relative costs.
//
// The topology is built once during device group creation and remains
// immutable. Devices cache relevant portions of the topology for ultra-fast
// (1-3ns) compatibility queries without pointer chasing or synchronization.
//
// Memory layout is optimized for cache efficiency:
// - Edge matrix is row-major (all edges from device i are contiguous)
// - Self-edges (diagonal) encode device capabilities
// - Symmetric properties (link_class) must match in both directions
//
// Thread safety: The topology is immutable after creation and can be
// queried concurrently from any thread without synchronization.
typedef struct iree_hal_topology_t {
  // Number of devices in this topology (1 to
  // IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT).
  uint32_t device_count;

  // Edge matrix in row-major order.
  // Edge from device i to device j is at edges[i * device_count + j].
  // Self-edges (i == j) encode device capabilities.
  iree_hal_topology_edge_t edges[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT *
                                 IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];

  // NUMA node assignment for each device (0-255).
  // Used for memory placement optimization.
  uint8_t numa_nodes[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
} iree_hal_topology_t;

// Returns an empty topology.
static inline iree_hal_topology_t iree_hal_topology_empty(void) {
  iree_hal_topology_t topology = {0};
  return topology;
}

// Returns true if the topology is empty (no devices).
static inline bool iree_hal_topology_is_empty(
    const iree_hal_topology_t* topology) {
  return topology->device_count == 0;
}

// Returns the number of devices in the topology.
static inline uint32_t iree_hal_topology_device_count(
    const iree_hal_topology_t* topology) {
  return topology->device_count;
}

// Queries the edge from |src_ordinal| to |dst_ordinal|.
// Returns 0 if either ordinal is out of range.
static inline iree_hal_topology_edge_t iree_hal_topology_query_edge(
    const iree_hal_topology_t* topology, uint32_t src_ordinal,
    uint32_t dst_ordinal) {
  IREE_ASSERT_LT(src_ordinal, topology->device_count);
  IREE_ASSERT_LT(dst_ordinal, topology->device_count);
  if (src_ordinal >= topology->device_count ||
      dst_ordinal >= topology->device_count) {
    return 0;
  }
  return topology->edges[src_ordinal * topology->device_count + dst_ordinal];
}

//===----------------------------------------------------------------------===//
// iree_hal_topology_edge_t pack/unpack helpers
//===----------------------------------------------------------------------===//

// Returns the wait interop mode from an edge.
// This describes how a semaphore created by the source device can be waited on
// by the destination device. The mode determines what mechanism is required:
// - NATIVE: Direct hardware wait, optimal performance (same driver/device)
// - IMPORT: Import external handle, wait natively (cross-driver on same HW)
// - COPY: Must poll or stage through host (cross-driver, incompatible HW)
// - NONE: Cannot wait (isolated devices, requires application synchronization)
//
// Implementations should set this based on their driver's semaphore interop
// capabilities. Same-driver edges are typically NATIVE. Cross-driver edges
// depend on whether one can import the other driver's semaphore handles
// (IMPORT) or need staging.
static inline iree_hal_topology_interop_mode_t iree_hal_topology_edge_wait_mode(
    iree_hal_topology_edge_t edge) {
  return (edge >> 0) & 0x3ull;
}

// Returns the signal interop mode from an edge.
// This describes how the destination device can signal a semaphore that will
// be consumed by the source device. Asymmetric from wait mode as signal and
// wait may have different hardware capabilities:
// - NATIVE: Direct hardware signal (same driver/device)
// - IMPORT: Export handle, signal through imported semaphore (cross-driver)
// - COPY: Must stage signal through host updates (incompatible HW)
// - NONE: Cannot signal (isolated devices)
//
// Implementations should consider their driver's ability to signal semaphores
// that other drivers can observe. GPU->GPU may support IMPORT while GPU->CPU
// might require COPY via host-visible memory or callbacks.
static inline iree_hal_topology_interop_mode_t
iree_hal_topology_edge_signal_mode(iree_hal_topology_edge_t edge) {
  return (edge >> 2) & 0x3ull;
}

// Returns the buffer read interop mode from an edge.
// This describes how the destination device can read from a buffer allocated
// by the source device. Critical for understanding data transfer requirements:
// - NATIVE: Direct memory access, peer-to-peer or unified memory (fastest)
// - IMPORT: Import buffer handle, map to destination address space (P2P)
// - COPY: Must copy through staging buffer (no direct access, PCIe/fabric)
// - NONE: Cannot read (isolated memory spaces, requires explicit transfers)
//
// Implementations should set this based on memory visibility. GPUs with
// NVLink/Infinity Fabric typically use NATIVE. GPUs across PCIe roots may need
// COPY. NUMA effects should be considered: reading non-local memory may be
// possible but slow.
static inline iree_hal_topology_interop_mode_t
iree_hal_topology_edge_buffer_read_mode(iree_hal_topology_edge_t edge) {
  return (edge >> 4) & 0x3ull;
}

// Returns the buffer write interop mode from an edge.
// This describes how the destination device can write to a buffer allocated
// by the source device. Often asymmetric from read mode due to coherency:
// - NATIVE: Direct write access with coherency guarantees (unified memory)
// - IMPORT: Can write but may need flushes/invalidates (peer-to-peer)
// - COPY: Must stage writes through intermediate buffer (no direct access)
// - NONE: Cannot write (isolated memory, read-only access)
//
// Implementations should consider cache coherency protocols. CPU->GPU writes
// may require host cache flushes. GPU->GPU writes across NUMA domains may need
// explicit synchronization. Coherency guarantees should be documented.
static inline iree_hal_topology_interop_mode_t
iree_hal_topology_edge_buffer_write_mode(iree_hal_topology_edge_t edge) {
  return (edge >> 6) & 0x3ull;
}

// Returns the link class from an edge.
// This categorizes the physical interconnect between devices, which determines
// bandwidth, latency, and coherency characteristics. Used by the scheduler to
// make data placement and transfer decisions:
// - SAME_DIE: Same silicon die, L3 cache shared (~TB/s, <5ns)
// - NVLINK_IF: High-speed interconnect like NVLink/Infinity Fabric (~600GB/s)
// - PCIE_SAME_ROOT: PCIe under same root complex (~32GB/s Gen4x16)
// - PCIE_CROSS_ROOT: PCIe across root complexes (~16GB/s, NUMA penalties)
// - HOST_STAGED: Must stage through host memory (slow, ~10GB/s)
// - FABRIC: Network fabric like InfiniBand/RoCE (variable, high latency)
// - OTHER: Unknown/custom interconnect (conservative assumptions)
// - ISOLATED: No direct connection (requires host coordination)
//
// Implementations should query platform interconnect topology. On Linux sysfs
// provides PCIe topology, vendor APIs (ROCm/CUDA) expose GPU links. This field
// must be symmetric: link_class(i,j) must equal link_class(j,i).
static inline iree_hal_topology_link_class_t iree_hal_topology_edge_link_class(
    iree_hal_topology_edge_t edge) {
  return (edge >> 55) & 0x7ull;
}

// Returns capability flags from an edge.
// This bitfield describes advanced interop capabilities between devices that
// affect synchronization, memory access patterns, and performance optimization:
// - SAME_RUNTIME_DOMAIN: Shared command submission, can batch operations
// - UNIFIED_MEMORY: Single address space, no translation needed (HMM/UVM)
// - PEER_COHERENT: Hardware cache coherency between devices (no flush/inval)
// - HOST_COHERENT: CPU can observe device writes without explicit sync
// - P2P_COPY: Hardware DMA between devices (bypasses host)
// - CONCURRENT_SAFE: Can safely access same memory concurrently (no races)
// - ATOMIC_DEVICE: Atomic operations visible across devices
// - ATOMIC_SYSTEM: Atomic operations visible system-wide (CPU + all devices)
// - TIMELINE_SEMAPHORE: Supports timeline semaphores for fine-grained sync
//
// Implementations should be conservative - only set flags that hardware truly
// guarantees. ATOMIC_SYSTEM requires platform support (PCIe atomics, vendor
// extensions). Check CUDA unified addressing, ROCm fine-grained memory, etc.
static inline iree_hal_topology_capability_t
iree_hal_topology_edge_capability_flags(iree_hal_topology_edge_t edge) {
  return (edge >> 24) & 0x7FFull;
}

// Returns wait cost from an edge (0-15, lower is better).
// Relative cost metric for waiting on semaphores across this edge. Used by the
// scheduler to estimate synchronization overhead:
// - 0: Zero cost (same device, hardware wait queue)
// - 1-3: Very low (same driver, native semaphore, <100ns)
// - 4-7: Low (imported semaphore, <1us)
// - 8-11: Moderate (polling/callbacks, <10us)
// - 12-14: High (host staging, >10us)
// - 15: Maximum (avoid if possible)
//
// Implementations should measure actual wait latency. Native waits are
// typically 0-1. Cross-driver imports are 3-5. Host polling is 8+. Consider
// CPU cost: polling wastes cycles even if latency is acceptable.
static inline uint8_t iree_hal_topology_edge_wait_cost(
    iree_hal_topology_edge_t edge) {
  return (edge >> 35) & 0xFull;
}

// Returns signal cost from an edge (0-15, lower is better).
// Relative cost metric for signaling semaphores across this edge. Often
// asymmetric from wait cost due to different hardware mechanisms:
// - 0: Zero cost (same device, single instruction)
// - 1-3: Very low (native signal, write-once)
// - 4-7: Low (exported semaphore, some bookkeeping)
// - 8-11: Moderate (host notification callback)
// - 12-14: High (host must poll and signal separately)
// - 15: Maximum (avoid if possible)
//
// Implementations should consider signal overhead. GPU signals are typically
// cheap (0-2), but callbacks to signal other drivers may be expensive (5-8).
// Host signaling GPU semaphores may require kernel transitions (6-10).
static inline uint8_t iree_hal_topology_edge_signal_cost(
    iree_hal_topology_edge_t edge) {
  return (edge >> 39) & 0xFull;
}

// Returns copy/transfer cost from an edge (0-15, lower is better).
// Relative cost metric for transferring data between devices on this edge.
// Combines bandwidth and latency into a single metric for scheduling:
// - 0: Zero cost (same device, pointer passing)
// - 1-3: Very low (same die, direct memory access, >500GB/s)
// - 4-7: Low (NVLink/Infinity Fabric, peer-to-peer DMA, >100GB/s)
// - 8-11: Moderate (PCIe Gen4x16, ~30GB/s)
// - 12-14: High (cross-NUMA, host staging, <10GB/s)
// - 15: Maximum (network fabric, avoid if possible)
//
// Implementations should consider both bandwidth and transfer setup cost. Large
// transfers care about bandwidth (PCIe=8). Small transfers care about latency
// (P2P=4, staged=12). Measure with realistic workload sizes (1MB-1GB).
static inline uint8_t iree_hal_topology_edge_copy_cost(
    iree_hal_topology_edge_t edge) {
  return (edge >> 43) & 0xFull;
}

// Returns latency class from an edge (0-15, lower is better).
// Categorizes round-trip latency for operations across this edge, independent
// of bandwidth. Used for latency-sensitive workloads like real-time inference:
// - 0: Same device (<10ns, cache/register latency)
// - 1-3: Same die/package (<100ns, L3 cache)
// - 4-6: Local NUMA node (<1us, peer-to-peer)
// - 7-9: Cross-NUMA/PCIe (<10us)
// - 10-12: Host staging (10-100us)
// - 13-15: Network fabric (>100us)
//
// Implementations should measure with small ping-pong transfers (<1KB). Latency
// matters for small operations and pipelined kernels. Don't confuse with
// bandwidth: NVLink has great bandwidth but still 4-5us latency.
static inline uint8_t iree_hal_topology_edge_latency_class(
    iree_hal_topology_edge_t edge) {
  return (edge >> 47) & 0xFull;
}

// Returns NUMA distance from an edge (0-15, lower is better).
// NUMA (Non-Uniform Memory Access) distance between devices, affecting memory
// access latency and bandwidth. Corresponds to ACPI SLIT (System Locality
// Information Table) values, normalized to 4 bits:
// - 0: Same NUMA node (local memory, optimal)
// - 1-3: Adjacent NUMA nodes (1 hop, still good)
// - 4-7: Near nodes (2 hops, noticeable penalty)
// - 8-11: Far nodes (3+ hops, significant penalty)
// - 12-15: Remote/cross-socket (avoid if possible)
//
// Implementations should query OS NUMA topology. On Linux check
// /sys/devices/system/node/, on Windows use GetNumaProximityNodeEx(). For GPUs,
// map to CPU NUMA node via PCIe root complex. Crucial for multi-socket systems
// where cross-socket access is 2-3x slower. Self-edges should always be 0.
static inline uint8_t iree_hal_topology_edge_numa_distance(
    iree_hal_topology_edge_t edge) {
  return (edge >> 51) & 0xFull;
}

// Returns semaphore import handle types from an edge.
// Bitfield of iree_hal_topology_handle_type_t values indicating which external
// semaphore handle types can be imported for waiting by the destination device.
// Common values include OPAQUE_FD (Linux), OPAQUE_WIN32_HANDLE (Windows), or
// SYNC_FD for timeline semaphores.
//
// Implementations should query platform capabilities (e.g., Vulkan/CUDA/ROCm
// external semaphore support) and set corresponding bits for supported types.
static inline iree_hal_topology_handle_type_t
iree_hal_topology_edge_semaphore_import_types(iree_hal_topology_edge_t edge) {
  return (edge >> 8) & 0xFull;
}

// Returns semaphore export handle types from an edge.
// Bitfield of iree_hal_topology_handle_type_t values indicating which external
// semaphore handle types can be exported for signaling by the source device.
//
// Implementations should advertise handle types that other drivers can import.
// Asymmetric from import types when devices have different export capabilities.
static inline iree_hal_topology_handle_type_t
iree_hal_topology_edge_semaphore_export_types(iree_hal_topology_edge_t edge) {
  return (edge >> 12) & 0xFull;
}

// Returns buffer import handle types from an edge.
// Bitfield of iree_hal_topology_handle_type_t values indicating which external
// buffer handle types can be imported for access by the destination device.
// Critical for zero-copy buffer sharing across drivers.
//
// Implementations should check for DMA-BUF (Linux), D3D12 resources (Windows),
// or vendor-specific handles (CUDA IPC, HIP IPC). Only set bits for handles
// that provide actual memory access, not just metadata transfer.
static inline iree_hal_topology_handle_type_t
iree_hal_topology_edge_buffer_import_types(iree_hal_topology_edge_t edge) {
  return (edge >> 16) & 0xFull;
}

// Returns buffer export handle types from an edge.
// Bitfield of iree_hal_topology_handle_type_t values indicating which external
// buffer handle types can be exported from the source device for sharing.
//
// Implementations should advertise handle types supported by their allocator.
// May differ from import types if device can consume more formats than produce.
static inline iree_hal_topology_handle_type_t
iree_hal_topology_edge_buffer_export_types(iree_hal_topology_edge_t edge) {
  return (edge >> 20) & 0xFull;
}

//===----------------------------------------------------------------------===//
// iree_hal_topology_edge_t edge construction
//===----------------------------------------------------------------------===//

// Computes a base edge from device capabilities and driver names by:
//  - Detecting same driver (enables NATIVE mode)
//  - Matching physical device UUIDs (cross-driver same-GPU detection)
//  - Intersecting external handle types (semaphore/buffer import/export)
//  - Deriving interop modes from handle type intersections
//  - Computing buffer modes based on P2P capability flags
//  - Propagating capability flags (bitwise AND of device flags)
//  - Calculating NUMA distance
//  - Assigning default link class and costs
// Returns base edge that can be refined by driver-specific logic via
// iree_hal_device_refine_topology_edge().
IREE_API_EXPORT iree_hal_topology_edge_t
iree_hal_topology_edge_from_capabilities(
    const iree_hal_device_capabilities_t* src_caps,
    const iree_hal_device_capabilities_t* dst_caps,
    iree_string_view_t src_driver_name, iree_string_view_t dst_driver_name);

//===----------------------------------------------------------------------===//
// iree_hal_topology_t formatting
//===----------------------------------------------------------------------===//

// Formats a topology edge as a human-readable string for debugging.
// Example: "wait=NATIVE signal=IMPORT read=COPY write=NONE link=PCIE cost=3"
IREE_API_EXPORT iree_status_t iree_hal_topology_edge_format(
    iree_hal_topology_edge_t edge, iree_string_builder_t* builder);

// Dumps the topology matrix to a string builder for debugging.
// Shows a matrix view with simplified edge representations.
IREE_API_EXPORT iree_status_t iree_hal_topology_dump_matrix(
    const iree_hal_topology_t* topology, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_TOPOLOGY_H_
