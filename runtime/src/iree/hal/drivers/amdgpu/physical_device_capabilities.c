// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device_capabilities.h"

#include <stdint.h>
#include <string.h>

// Inclusive unsigned 16-bit range used for gfx IP table matching.
typedef struct iree_hal_amdgpu_uint16_range_t {
  // Inclusive lower bound.
  uint16_t min;
  // Inclusive upper bound.
  uint16_t max;
} iree_hal_amdgpu_uint16_range_t;

// Gfx IP version range matched by a capability table row.
typedef struct iree_hal_amdgpu_gfxip_version_range_t {
  // Accepted major version range.
  iree_hal_amdgpu_uint16_range_t major;
  // Accepted minor version range.
  iree_hal_amdgpu_uint16_range_t minor;
  // Accepted stepping range.
  iree_hal_amdgpu_uint16_range_t stepping;
} iree_hal_amdgpu_gfxip_version_range_t;

// AMD vendor-packet capability table row.
typedef struct iree_hal_amdgpu_vendor_packet_capability_row_t {
  // Gfx IP version range matched by this row.
  iree_hal_amdgpu_gfxip_version_range_t version;
  // Vendor-packet and PM4 packet-family capabilities enabled by this row.
  iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities;
} iree_hal_amdgpu_vendor_packet_capability_row_t;

enum {
  // Packet families validated on the local gfx1100 bring-up system.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_GFX1100_VALIDATED =
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64 |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_WRITE_DATA_MEMORY |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COPY_DATA_MEMORY |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_UCONFIG_REG |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_REGISTER_READBACK |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_PERFCOUNTER_READBACK |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_IMMEDIATE_WRITE,
};

static bool iree_hal_amdgpu_uint16_range_contains(
    iree_hal_amdgpu_uint16_range_t range, uint16_t value) {
  return value >= range.min && value <= range.max;
}

static bool iree_hal_amdgpu_gfxip_version_range_contains(
    iree_hal_amdgpu_gfxip_version_range_t range,
    iree_hal_amdgpu_gfxip_version_t version) {
  return iree_hal_amdgpu_uint16_range_contains(range.major, version.major) &&
         iree_hal_amdgpu_uint16_range_contains(range.minor, version.minor) &&
         iree_hal_amdgpu_uint16_range_contains(range.stepping,
                                               version.stepping);
}

bool iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* memory) {
  return iree_any_bit_set(
      memory->flags,
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE);
}

bool iree_hal_amdgpu_memory_pool_access_is_valid(
    hsa_amd_memory_pool_access_t access) {
  switch (access) {
    case HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED:
    case HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT:
    case HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT:
      return true;
    default:
      return false;
  }
}

iree_hal_topology_interop_mode_t
iree_hal_amdgpu_memory_pool_access_topology_mode(
    hsa_amd_memory_pool_access_t access) {
  IREE_ASSERT(iree_hal_amdgpu_memory_pool_access_is_valid(access),
              "invalid HSA memory-pool access mode");
  switch (access) {
    case HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT:
      return IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
    case HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT:
      return IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
    case HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED:
    default:
      return IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  }
}

iree_hal_topology_capability_t
iree_hal_amdgpu_memory_pool_access_topology_capabilities(
    hsa_amd_memory_pool_access_t access) {
  IREE_ASSERT(iree_hal_amdgpu_memory_pool_access_is_valid(access),
              "invalid HSA memory-pool access mode");
  if (access == HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT) {
    return IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT;
  }
  return IREE_HAL_TOPOLOGY_CAPABILITY_NONE;
}

// Maps an HSA link type to a HAL topology link class.
//
// For multi-hop links, callers should take the worst/highest class.
static iree_hal_topology_link_class_t iree_hal_amdgpu_link_type_to_link_class(
    hsa_amd_link_info_type_t link_type) {
  switch (link_type) {
    case HSA_AMD_LINK_INFO_TYPE_XGMI:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF;
    case HSA_AMD_LINK_INFO_TYPE_PCIE:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT;
    case HSA_AMD_LINK_INFO_TYPE_QPI:
    case HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT:
      // Cross-socket interconnects: treat as cross-root PCIe.
      return IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_CROSS_ROOT;
    case HSA_AMD_LINK_INFO_TYPE_INFINBAND:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC;
    default:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_OTHER;
  }
}

static iree_hal_amdgpu_physical_topology_link_flags_t
iree_hal_amdgpu_link_type_to_physical_topology_link_flags(
    hsa_amd_link_info_type_t link_type) {
  switch (link_type) {
    case HSA_AMD_LINK_INFO_TYPE_PCIE:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_PCIE;
    case HSA_AMD_LINK_INFO_TYPE_XGMI:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_XGMI;
    case HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_HYPERTRANSPORT;
    case HSA_AMD_LINK_INFO_TYPE_QPI:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_QPI;
    case HSA_AMD_LINK_INFO_TYPE_INFINBAND:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_INFINIBAND;
    default:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_OTHER;
  }
}

static void iree_hal_amdgpu_topology_costs_from_link_class(
    iree_hal_topology_link_class_t link_class, uint8_t* out_copy_cost,
    uint8_t* out_latency_class) {
  switch (link_class) {
    case IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE:
      *out_copy_cost = 0;
      *out_latency_class = 0;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF:
      *out_copy_cost = 3;
      *out_latency_class = 3;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT:
      *out_copy_cost = 7;
      *out_latency_class = 7;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_CROSS_ROOT:
      *out_copy_cost = 9;
      *out_latency_class = 9;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED:
      *out_copy_cost = 13;
      *out_latency_class = 11;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC:
      *out_copy_cost = 15;
      *out_latency_class = 14;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_ISOLATED:
      *out_copy_cost = 15;
      *out_latency_class = 15;
      break;
    default:
      *out_copy_cost = 11;
      *out_latency_class = 10;
      break;
  }
}

static uint8_t iree_hal_amdgpu_topology_scale_hsa_numa_distance(
    uint32_t hsa_numa_distance) {
  if (hsa_numa_distance == 0) return 0;
  uint32_t scaled = hsa_numa_distance > 10 ? (hsa_numa_distance - 10) / 2 : 0;
  return (uint8_t)iree_min(scaled, 15u);
}

static iree_status_t iree_hal_amdgpu_validate_physical_topology_edge_access(
    hsa_amd_memory_pool_access_t access, const char* pool_kind) {
  if (IREE_LIKELY(iree_hal_amdgpu_memory_pool_access_is_valid(access))) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "HSA reported unknown %s memory pool access mode %u",
                          pool_kind, (uint32_t)access);
}

static void iree_hal_amdgpu_physical_topology_edge_initialize(
    iree_hal_amdgpu_physical_topology_edge_t* out_edge) {
  memset(out_edge, 0, sizeof(*out_edge));
  out_edge->memory_access.coarse = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  out_edge->memory_access.fine = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  out_edge->coherency.all_hops_coherent = 1;
  out_edge->atomics.all_hops_32bit = 1;
  out_edge->atomics.all_hops_64bit = 1;
  out_edge->link.link_class = IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE;
  out_edge->modes.noncoherent_read = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  out_edge->modes.noncoherent_write = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  out_edge->modes.coherent_read = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  out_edge->modes.coherent_write = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
}

static iree_hal_topology_capability_t
iree_hal_amdgpu_physical_topology_guaranteed_capabilities(
    const iree_hal_amdgpu_physical_topology_edge_t* edge) {
  iree_hal_topology_capability_t capabilities =
      IREE_HAL_TOPOLOGY_CAPABILITY_NONE;
  if (!edge->memory_access.coarse_accessible &&
      !edge->memory_access.fine_accessible) {
    return capabilities;
  }
  capabilities |= IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY;
  if (edge->coherency.all_hops_coherent) {
    capabilities |= IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT;
  }
  if (edge->atomics.all_hops_32bit) {
    capabilities |= IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE;
  }
  if (edge->atomics.all_hops_64bit) {
    capabilities |= IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM;
  }
  return capabilities;
}

static iree_hal_topology_capability_t
iree_hal_amdgpu_physical_topology_required_capabilities(
    const iree_hal_amdgpu_physical_topology_edge_t* edge) {
  iree_hal_topology_capability_t capabilities =
      IREE_HAL_TOPOLOGY_CAPABILITY_NONE;
  capabilities |= iree_hal_amdgpu_memory_pool_access_topology_capabilities(
      edge->memory_access.coarse);
  capabilities |= iree_hal_amdgpu_memory_pool_access_topology_capabilities(
      edge->memory_access.fine);
  return capabilities;
}

iree_status_t iree_hal_amdgpu_select_physical_topology_edge(
    const iree_hal_amdgpu_physical_topology_edge_selection_t* selection,
    iree_hal_amdgpu_physical_topology_edge_t* out_edge) {
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(out_edge);
  iree_hal_amdgpu_physical_topology_edge_initialize(out_edge);

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_validate_physical_topology_edge_access(
      selection->memory_access.coarse, "coarse"));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_validate_physical_topology_edge_access(
      selection->memory_access.fine, "fine"));
  if (IREE_UNLIKELY(selection->link.count && !selection->link.hops)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU physical topology edge selection requires link hops when "
        "link count is nonzero");
  }

  out_edge->memory_access.coarse = selection->memory_access.coarse;
  out_edge->memory_access.fine = selection->memory_access.fine;
  out_edge->memory_access.coarse_accessible =
      selection->memory_access.coarse !=
      HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  out_edge->memory_access.fine_accessible =
      selection->memory_access.fine != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;

  for (iree_host_size_t i = 0; i < selection->link.count; ++i) {
    const hsa_amd_memory_pool_link_info_t* link_hop = &selection->link.hops[i];
    iree_hal_topology_link_class_t link_class =
        iree_hal_amdgpu_link_type_to_link_class(link_hop->link_type);
    if (link_class > out_edge->link.link_class) {
      out_edge->link.link_class = link_class;
    }
    out_edge->link.flags |=
        iree_hal_amdgpu_link_type_to_physical_topology_link_flags(
            link_hop->link_type);
    uint8_t numa_distance = iree_hal_amdgpu_topology_scale_hsa_numa_distance(
        link_hop->numa_distance);
    if (numa_distance > out_edge->link.numa_distance) {
      out_edge->link.numa_distance = numa_distance;
    }
    if (!link_hop->coherent_support) {
      out_edge->coherency.all_hops_coherent = 0;
    }
    if (!link_hop->atomic_support_32bit) {
      out_edge->atomics.all_hops_32bit = 0;
    }
    if (!link_hop->atomic_support_64bit) {
      out_edge->atomics.all_hops_64bit = 0;
    }
  }

  if (!out_edge->memory_access.coarse_accessible &&
      !out_edge->memory_access.fine_accessible) {
    out_edge->link.link_class = IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED;
    out_edge->coherency.all_hops_coherent = 0;
    out_edge->atomics.all_hops_32bit = 0;
    out_edge->atomics.all_hops_64bit = 0;
  }

  iree_hal_amdgpu_topology_costs_from_link_class(out_edge->link.link_class,
                                                 &out_edge->link.copy_cost,
                                                 &out_edge->link.latency_class);
  out_edge->capabilities.guaranteed =
      iree_hal_amdgpu_physical_topology_guaranteed_capabilities(out_edge);
  out_edge->capabilities.required =
      iree_hal_amdgpu_physical_topology_required_capabilities(out_edge);
  out_edge->modes.noncoherent_read =
      iree_hal_amdgpu_memory_pool_access_topology_mode(
          out_edge->memory_access.coarse);
  out_edge->modes.noncoherent_write = out_edge->modes.noncoherent_read;
  out_edge->modes.coherent_read =
      iree_hal_amdgpu_memory_pool_access_topology_mode(
          out_edge->memory_access.fine);
  out_edge->modes.coherent_write = out_edge->modes.coherent_read;
  return iree_ok_status();
}

static bool iree_hal_amdgpu_gfxip_is_pre_gfx908(
    iree_hal_amdgpu_gfxip_version_t version) {
  return version.major < 9 ||
         (version.major == 9 && version.minor == 0 && version.stepping < 8);
}

static bool iree_hal_amdgpu_gfxip_is_gfx101x(
    iree_hal_amdgpu_gfxip_version_t version) {
  return version.major == 10 && (version.minor == 0 || version.minor == 1);
}

bool iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(
    iree_hal_amdgpu_gfxip_version_t version) {
  // Matches the HDP workaround eligibility in CLR's setKernelArgImpl. Devices
  // outside this set stay on host kernarg memory unless we add a first-class
  // readback publication mode.
  return !iree_hal_amdgpu_gfxip_is_pre_gfx908(version) &&
         !iree_hal_amdgpu_gfxip_is_gfx101x(version);
}

iree_status_t iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t*
        selection,
    iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* out_memory) {
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(out_memory);
  memset(out_memory, 0, sizeof(*out_memory));

  if (!selection->memory_pool.handle || selection->cpu.count == 0) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(selection->cpu.count > IREE_HAL_AMDGPU_MAX_CPU_AGENT)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU topology has %" PRIhsz
        " CPU agents but CPU-visible coarse memory tracks at most %d",
        selection->cpu.count, IREE_HAL_AMDGPU_MAX_CPU_AGENT);
  }
  if (!iree_any_bit_set(
          selection->flags,
          IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_HOST_WRITE_PUBLICATION_SUPPORTED)) {
    return iree_ok_status();
  }
  if (!iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(
          selection->gfxip_version)) {
    return iree_ok_status();
  }
  if (!selection->hdp.registers.HDP_MEM_FLUSH_CNTL ||
      !selection->hdp.registers.HDP_REG_FLUSH_CNTL) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!selection->cpu.agents || !selection->cpu.access)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "CPU-visible device-coarse memory selection requires CPU agents and "
        "access modes");
  }

  for (iree_host_size_t i = 0; i < selection->cpu.count; ++i) {
    const hsa_amd_memory_pool_access_t access = selection->cpu.access[i];
    if (IREE_UNLIKELY(!iree_hal_amdgpu_memory_pool_access_is_valid(access))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HSA reported unknown memory pool access mode %u",
                              (uint32_t)access);
    }
    if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      return iree_ok_status();
    }
  }

  iree_host_size_t access_agent_count = 0;
  for (iree_host_size_t i = 0; i < selection->cpu.count; ++i) {
    out_memory->access_agents[access_agent_count++] = selection->cpu.agents[i];
  }
  out_memory->access_agents[access_agent_count++] = selection->device_agent;
  out_memory->memory_pool = selection->memory_pool;
  out_memory->access_agent_count = access_agent_count;
  out_memory->host_write_publication =
      (iree_hal_amdgpu_kernarg_ring_publication_t){
          .mode = IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_HDP_FLUSH,
          .hdp_mem_flush_control = selection->hdp.registers.HDP_MEM_FLUSH_CNTL,
      };
  out_memory->flags =
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE |
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_HDP_FLUSH;
  return iree_ok_status();
}

void iree_hal_amdgpu_select_memory_system_capabilities(
    const iree_hal_amdgpu_memory_system_capabilities_selection_t* selection,
    iree_hal_amdgpu_memory_system_capabilities_t* out_capabilities) {
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(out_capabilities);
  memset(out_capabilities, 0, sizeof(*out_capabilities));

  out_capabilities->svm.supported = selection->svm.supported ? 1u : 0u;
  out_capabilities->svm.accessible_by_default =
      selection->svm.accessible_by_default ? 1u : 0u;
  out_capabilities->svm.xnack_enabled = selection->svm.xnack_enabled ? 1u : 0u;
  out_capabilities->svm.direct_host_access =
      selection->svm.direct_host_access ? 1u : 0u;
  out_capabilities->device_local.fine_host_visible =
      selection->device_local.fine_memory_pool.handle ? 1u : 0u;
  out_capabilities->device_local.coarse_cpu_visible =
      selection->device_local.coarse_cpu_visible_memory &&
              iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
                  selection->device_local.coarse_cpu_visible_memory)
          ? 1u
          : 0u;
}

iree_hal_device_capability_bits_t
iree_hal_amdgpu_select_memory_system_device_capability_flags(
    const iree_hal_amdgpu_memory_system_capabilities_t* capabilities) {
  IREE_ASSERT_ARGUMENT(capabilities);
  iree_hal_device_capability_bits_t flags = IREE_HAL_DEVICE_CAPABILITY_NONE;
  if (capabilities->svm.supported) {
    flags |= IREE_HAL_DEVICE_CAPABILITY_SHARED_VIRTUAL_ADDRESS;
  }
  if (capabilities->svm.accessible_by_default) {
    flags |= IREE_HAL_DEVICE_CAPABILITY_UNIFIED_MEMORY;
  }
  return flags;
}

bool iree_hal_amdgpu_memory_system_requires_svm_access_attributes(
    const iree_hal_amdgpu_memory_system_capabilities_t* capabilities) {
  IREE_ASSERT_ARGUMENT(capabilities);
  return capabilities->svm.supported &&
         !capabilities->svm.accessible_by_default;
}

iree_hal_amdgpu_aql_prepublished_kernarg_storage_t
iree_hal_amdgpu_select_prepublished_kernarg_storage(
    hsa_amd_memory_pool_t fine_block_memory_pool) {
  if (!fine_block_memory_pool.handle) {
    return iree_hal_amdgpu_aql_prepublished_kernarg_storage_disabled();
  }
  return iree_hal_amdgpu_aql_prepublished_kernarg_storage_device_fine_host_coherent();
}

iree_hal_amdgpu_vendor_packet_capability_flags_t
iree_hal_amdgpu_select_vendor_packet_capabilities(
    iree_hal_amdgpu_gfxip_version_t version) {
  // The CDNA BARRIER_VALUE rows match CLR's barrier_value_packet_ gate:
  // gfx9.0.10 or gfx9.[minor >= 4].[stepping 0..2].
  //
  // AQL PM4-IB is selected for known gfx9-gfx12 ISAs so the timestamp strategy
  // can own queue-device profiling support across the packet families mirrored
  // from aqlprofile. Other PM4 packet families stay opt-in until each
  // packet-family contract has hardware evidence or an explicit probe.
  static const iree_hal_amdgpu_vendor_packet_capability_row_t kRows[] = {
      {
          .version =
              {
                  .major = {9, 9},
                  .minor = {0, 0},
                  .stepping = {10, 10},
              },
          .capabilities =
              IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
              IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE,
      },
      {
          .version =
              {
                  .major = {9, 9},
                  .minor = {4, UINT16_MAX},
                  .stepping = {0, 2},
              },
          .capabilities =
              IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
              IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE,
      },
      {
          .version =
              {
                  .major = {11, 11},
                  .minor = {0, 0},
                  .stepping = {0, 0},
              },
          .capabilities =
              IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_GFX1100_VALIDATED,
      },
  };

  const bool known_pm4_ib_family = version.major >= 9 && version.major <= 12;
  iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities =
      known_pm4_ib_family ? IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB
                          : 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kRows); ++i) {
    if (iree_hal_amdgpu_gfxip_version_range_contains(kRows[i].version,
                                                     version)) {
      capabilities |= kRows[i].capabilities;
    }
  }
  return capabilities;
}

iree_hal_amdgpu_pm4_timestamp_strategy_t
iree_hal_amdgpu_select_pm4_timestamp_strategy(
    iree_hal_amdgpu_gfxip_version_t version) {
  // COPY_DATA GPU-clock readback is the queue-device timestamp path selected on
  // PM4-IB queues. The destination and cache-policy fields mirror the
  // aqlprofile command builder families: gfx9 including gfx94/gfx95 uses
  // memory stream, gfx10/gfx11 uses TC_L2 with LRU, and gfx12 uses TC_L2 with
  // last-use temporal policy.
  switch (version.major) {
    case 9:
      return IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM;
    case 10:
    case 11:
      return IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU;
    case 12:
      return IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU;
    default:
      return IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE;
  }
}

iree_hal_amdgpu_wait_barrier_strategy_t
iree_hal_amdgpu_select_wait_barrier_strategy(
    iree_hal_amdgpu_vendor_packet_capability_flags_t
        vendor_packet_capabilities) {
  if (vendor_packet_capabilities &
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE) {
    return IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_AQL_BARRIER_VALUE;
  }
  if (vendor_packet_capabilities &
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64) {
    return IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_PM4_WAIT_REG_MEM64;
  }
  return IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER;
}
