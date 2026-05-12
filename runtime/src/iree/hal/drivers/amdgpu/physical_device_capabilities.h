// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_CAPABILITIES_H_
#define IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_CAPABILITIES_H_

#include "iree/base/api.h"
#include "iree/hal/device.h"
#include "iree/hal/drivers/amdgpu/aql_prepublished_kernarg_storage.h"
#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/pm4_capabilities.h"
#include "iree/hal/drivers/amdgpu/util/target_id.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_amdgpu_cpu_visible_device_coarse_memory_flag_bits_e {
  IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_NONE = 0u,
  // All CPU agents can access the GPU coarse-grained memory pool and the
  // driver knows how to publish CPU writes before GPU consumption.
  IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE = 1u << 0,
  // CPU writes require an HDP flush before the GPU consumes the memory.
  IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_HDP_FLUSH = 1u << 1,
} iree_hal_amdgpu_cpu_visible_device_coarse_memory_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_cpu_visible_device_coarse_memory_flags_t;

// Physical-device capability for CPU-visible GPU coarse-grained memory.
typedef struct iree_hal_amdgpu_cpu_visible_device_coarse_memory_t {
  // GPU coarse-grained HSA memory pool CPU agents can access.
  hsa_amd_memory_pool_t memory_pool;
  // Agents granted access for allocations that use |memory_pool|.
  hsa_agent_t access_agents[IREE_HAL_AMDGPU_MAX_CPU_AGENT + 1];
  // Number of valid entries in |access_agents|.
  iree_host_size_t access_agent_count;
  // Publication required after CPU writes and before GPU consumption.
  iree_hal_amdgpu_kernarg_ring_publication_t host_write_publication;
  // Capability flags from
  // iree_hal_amdgpu_cpu_visible_device_coarse_memory_flag_bits_t.
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_flags_t flags;
} iree_hal_amdgpu_cpu_visible_device_coarse_memory_t;

typedef enum iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_flag_bits_e {
  IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_NONE = 0u,
  // Host writes can be published for CPU-visible device-coarse memory.
  IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_HOST_WRITE_PUBLICATION_SUPPORTED =
      1u << 0,
} iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_flag_bits_t;

typedef uint32_t
    iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_flags_t;

// Queried facts used to select CPU-visible device-coarse memory capability.
typedef struct iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t {
  // GPU agent that owns |memory_pool|.
  hsa_agent_t device_agent;
  // GPU coarse-grained memory pool being considered.
  hsa_amd_memory_pool_t memory_pool;
  // Parsed gfx IP version for HDP publication eligibility.
  iree_hal_amdgpu_gfxip_version_t gfxip_version;
  // CPU agents and their access to |memory_pool|.
  struct {
    // CPU agents that may write the memory.
    const hsa_agent_t* agents;
    // Per-CPU-agent access mode for |memory_pool|.
    const hsa_amd_memory_pool_access_t* access;
    // Number of entries in |agents| and |access|.
    iree_host_size_t count;
  } cpu;
  // HDP publication registers reported by HSA.
  struct {
    // Raw HSA HDP flush register descriptor.
    hsa_amd_hdp_flush_t registers;
  } hdp;
  // Selection flags from
  // iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_flag_bits_t.
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_flags_t flags;
} iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t;

// Returns true if CPU-visible device-coarse memory is available.
bool iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* memory);

// Returns true if |access| is a known HSA memory-pool access mode.
bool iree_hal_amdgpu_memory_pool_access_is_valid(
    hsa_amd_memory_pool_access_t access);

// Maps an HSA memory-pool access mode to the safe default topology buffer mode.
iree_hal_topology_interop_mode_t
iree_hal_amdgpu_memory_pool_access_topology_mode(
    hsa_amd_memory_pool_access_t access);

// Maps an HSA memory-pool access mode to additional topology capabilities.
iree_hal_topology_capability_t
iree_hal_amdgpu_memory_pool_access_topology_capabilities(
    hsa_amd_memory_pool_access_t access);

typedef enum iree_hal_amdgpu_physical_topology_link_flag_bits_e {
  IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_NONE = 0u,
  // At least one HSA-reported hop uses PCIe.
  IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_PCIE = 1u << 0,
  // At least one HSA-reported hop uses xGMI.
  IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_XGMI = 1u << 1,
  // At least one HSA-reported hop uses HyperTransport.
  IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_HYPERTRANSPORT = 1u << 2,
  // At least one HSA-reported hop uses QPI.
  IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_QPI = 1u << 3,
  // At least one HSA-reported hop uses InfiniBand.
  IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_INFINIBAND = 1u << 4,
  // At least one HSA-reported hop uses an unknown link type.
  IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_OTHER = 1u << 5,
} iree_hal_amdgpu_physical_topology_link_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_physical_topology_link_flags_t;

// Physical source->destination topology edge selected from already-queried HSA
// memory-pool access and link-hop facts.
typedef struct iree_hal_amdgpu_physical_topology_edge_t {
  // Source-agent access to the destination memory pools.
  struct {
    // Source-agent access to the destination coarse-grained memory pool.
    hsa_amd_memory_pool_access_t coarse;
    // Source-agent access to the destination fine-grained memory pool.
    hsa_amd_memory_pool_access_t fine;
    // True when |coarse| permits some direct device access.
    uint32_t coarse_accessible : 1;
    // True when |fine| permits some direct device access.
    uint32_t fine_accessible : 1;
  } memory_access;

  // HSA link-hop facts collapsed into strategy-friendly topology values.
  struct {
    // Worst physical link class across HSA-reported link hops.
    iree_hal_topology_link_class_t link_class;
    // Conservative copy-cost class derived from |link_class|.
    uint8_t copy_cost;
    // Conservative latency class derived from |link_class|.
    uint8_t latency_class;
    // Worst normalized NUMA distance reported by HSA link hops.
    uint8_t numa_distance;
    // Link flags from iree_hal_amdgpu_physical_topology_link_flag_bits_t.
    iree_hal_amdgpu_physical_topology_link_flags_t flags;
  } link;

  // Link coherency facts.
  struct {
    // True when every HSA-reported link hop supports coherent transactions.
    uint32_t all_hops_coherent : 1;
  } coherency;

  // Link atomic-transaction facts.
  struct {
    // True when every HSA-reported link hop supports 32-bit atomics.
    uint32_t all_hops_32bit : 1;
    // True when every HSA-reported link hop supports 64-bit atomics.
    uint32_t all_hops_64bit : 1;
  } atomics;

  // Generic HAL topology capabilities implied by the physical edge.
  struct {
    // Positive capabilities guaranteed by this physical pair.
    iree_hal_topology_capability_t guaranteed;
    // Requirement bits imposed by this physical pair.
    iree_hal_topology_capability_t required;
  } capabilities;

  // Generic HAL buffer interop modes implied by memory-pool access.
  struct {
    // Noncoherent read mode derived from coarse-grained pool access.
    iree_hal_topology_interop_mode_t noncoherent_read;
    // Noncoherent write mode derived from coarse-grained pool access.
    iree_hal_topology_interop_mode_t noncoherent_write;
    // Coherent read mode derived from fine-grained pool access.
    iree_hal_topology_interop_mode_t coherent_read;
    // Coherent write mode derived from fine-grained pool access.
    iree_hal_topology_interop_mode_t coherent_write;
  } modes;
} iree_hal_amdgpu_physical_topology_edge_t;

// Already-queried HSA facts used to select a physical topology edge.
typedef struct iree_hal_amdgpu_physical_topology_edge_selection_t {
  // Source-agent access to the destination memory pools.
  struct {
    // Source-agent access to the destination coarse-grained memory pool.
    hsa_amd_memory_pool_access_t coarse;
    // Source-agent access to the destination fine-grained memory pool.
    hsa_amd_memory_pool_access_t fine;
  } memory_access;

  // HSA link-hop facts for the source->destination memory path.
  struct {
    // HSA-reported link-hop records.
    const hsa_amd_memory_pool_link_info_t* hops;
    // Number of entries in |hops|.
    iree_host_size_t count;
  } link;
} iree_hal_amdgpu_physical_topology_edge_selection_t;

// Selects a physical topology edge from already-queried HSA facts.
iree_status_t iree_hal_amdgpu_select_physical_topology_edge(
    const iree_hal_amdgpu_physical_topology_edge_selection_t* selection,
    iree_hal_amdgpu_physical_topology_edge_t* out_edge);

// Returns true if the gfx IP family permits HDP kernarg publication.
bool iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(
    iree_hal_amdgpu_gfxip_version_t version);

// Selects CPU-visible device-coarse memory from already-queried topology facts.
iree_status_t iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t*
        selection,
    iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* out_memory);

// Selects the queue-local PM4 timestamp packet strategy for |version|.
iree_hal_amdgpu_pm4_timestamp_strategy_t
iree_hal_amdgpu_select_pm4_timestamp_strategy(
    iree_hal_amdgpu_gfxip_version_t version);

// AMDGPU memory-system facts used to derive conservative HAL topology flags.
typedef struct iree_hal_amdgpu_memory_system_capabilities_t {
  // HSA SVM/HMM process and agent facts.
  struct {
    // HSA SVM attribute and prefetch APIs are available.
    uint32_t supported : 1;
    // System allocations are accessible by GPU agents without per-range grants.
    uint32_t accessible_by_default : 1;
    // The process is bound to XNACK-enabled execution.
    uint32_t xnack_enabled : 1;
    // The host can directly access SVM pages resident in this GPU local memory.
    uint32_t direct_host_access : 1;
  } svm;

  // Device-local memory placement facts.
  struct {
    // A host-coherent fine-grained device memory pool is available.
    uint32_t fine_host_visible : 1;
    // A CPU-visible coarse-grained device memory pool is usable by the driver.
    uint32_t coarse_cpu_visible : 1;
  } device_local;
} iree_hal_amdgpu_memory_system_capabilities_t;

// Already-queried facts used to select memory-system capabilities.
typedef struct iree_hal_amdgpu_memory_system_capabilities_selection_t {
  // HSA SVM/HMM process and agent facts.
  struct {
    // Whether HSA SVM APIs are available in this process.
    uint32_t supported : 1;
    // Whether pageable/system memory is GPU-accessible without SVM attributes.
    uint32_t accessible_by_default : 1;
    // Whether the process is bound to XNACK-enabled execution.
    uint32_t xnack_enabled : 1;
    // Whether this GPU reports direct host access to resident SVM pages.
    uint32_t direct_host_access : 1;
  } svm;

  // Device-local memory placement facts.
  struct {
    // Fine-grained global memory pool considered for host-visible device data.
    hsa_amd_memory_pool_t fine_memory_pool;
    // Selected CPU-visible coarse-grained device-memory capability.
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_t*
        coarse_cpu_visible_memory;
  } device_local;
} iree_hal_amdgpu_memory_system_capabilities_selection_t;

// Selects memory-system capabilities from already-queried facts.
void iree_hal_amdgpu_select_memory_system_capabilities(
    const iree_hal_amdgpu_memory_system_capabilities_selection_t* selection,
    iree_hal_amdgpu_memory_system_capabilities_t* out_capabilities);

// Returns HAL device capability flags implied by AMDGPU memory-system facts.
iree_hal_device_capability_bits_t
iree_hal_amdgpu_select_memory_system_device_capability_flags(
    const iree_hal_amdgpu_memory_system_capabilities_t* capabilities);

// Returns true when SVM ranges require explicit HSA access attributes before a
// GPU can safely access them.
bool iree_hal_amdgpu_memory_system_requires_svm_access_attributes(
    const iree_hal_amdgpu_memory_system_capabilities_t* capabilities);

// Selects command-buffer prepublished kernarg storage from queried memory
// pools.
iree_hal_amdgpu_aql_prepublished_kernarg_storage_t
iree_hal_amdgpu_select_prepublished_kernarg_storage(
    hsa_amd_memory_pool_t fine_block_memory_pool);

// Selects AMD vendor AQL packet and PM4 packet-family capabilities from the
// parsed gfx IP version.
iree_hal_amdgpu_vendor_packet_capability_flags_t
iree_hal_amdgpu_select_vendor_packet_capabilities(
    iree_hal_amdgpu_gfxip_version_t version);

// Selects the cross-queue wait strategy from already-selected vendor packet
// capabilities.
iree_hal_amdgpu_wait_barrier_strategy_t
iree_hal_amdgpu_select_wait_barrier_strategy(
    iree_hal_amdgpu_vendor_packet_capability_flags_t
        vendor_packet_capabilities);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_CAPABILITIES_H_
