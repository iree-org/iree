// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/profile_counters.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/libaqlprofile.h"

//===----------------------------------------------------------------------===//
// Counter support tables
//===----------------------------------------------------------------------===//

static const iree_string_view_t iree_hal_amdgpu_profile_counter_set_name(void) {
  return IREE_SV("amdgpu.pmc");
}

enum { iree_hal_amdgpu_profile_counter_packets_per_set = 3u };
enum { iree_hal_amdgpu_profile_counter_event_unsupported = UINT32_MAX };

// Static description of one raw hardware counter we can request from
// aqlprofile by name.
typedef struct iree_hal_amdgpu_profile_counter_descriptor_t {
  // User-visible counter name accepted in profiling options and metadata.
  iree_string_view_t name;
  // User-visible hardware block name emitted in metadata.
  iree_string_view_t block_name;
  // User-visible description emitted in metadata.
  iree_string_view_t description;
  // Display unit for values returned by this counter.
  iree_hal_profile_counter_unit_t unit;
  // aqlprofile hardware block identifier for this counter.
  iree_hal_amdgpu_aqlprofile_block_name_t block_name_id;
  // aqlprofile event id for gfx9 devices, or unsupported.
  uint32_t gfx9_event_id;
  // aqlprofile event id for gfx10 devices, or unsupported.
  uint32_t gfx10_event_id;
  // aqlprofile event id for gfx11 devices, or unsupported.
  uint32_t gfx11_event_id;
  // aqlprofile event id for gfx12 devices, or unsupported.
  uint32_t gfx12_event_id;
} iree_hal_amdgpu_profile_counter_descriptor_t;

#define IREE_HAL_AMDGPU_PROFILE_COUNTER_SV(value) {(value), sizeof(value) - 1}

static const iree_hal_amdgpu_profile_counter_descriptor_t
    iree_hal_amdgpu_profile_counter_descriptors[] = {
        {
            .name = IREE_HAL_AMDGPU_PROFILE_COUNTER_SV("SQ_WAVES"),
            .block_name = IREE_HAL_AMDGPU_PROFILE_COUNTER_SV("SQ"),
            .description = IREE_HAL_AMDGPU_PROFILE_COUNTER_SV(
                "Raw SQ_WAVES values returned by aqlprofile."),
            .unit = IREE_HAL_PROFILE_COUNTER_UNIT_COUNT,
            .block_name_id = IREE_HAL_AMDGPU_AQLPROFILE_BLOCK_NAME_SQ,
            .gfx9_event_id = 4,
            .gfx10_event_id = 4,
            .gfx11_event_id = 4,
            .gfx12_event_id = 4,
        },
        {
            .name = IREE_HAL_AMDGPU_PROFILE_COUNTER_SV("SQ_WAVES_32"),
            .block_name = IREE_HAL_AMDGPU_PROFILE_COUNTER_SV("SQ"),
            .description = IREE_HAL_AMDGPU_PROFILE_COUNTER_SV(
                "Raw SQ_WAVES_32 values returned by aqlprofile."),
            .unit = IREE_HAL_PROFILE_COUNTER_UNIT_COUNT,
            .block_name_id = IREE_HAL_AMDGPU_AQLPROFILE_BLOCK_NAME_SQ,
            .gfx9_event_id = iree_hal_amdgpu_profile_counter_event_unsupported,
            .gfx10_event_id = 5,
            .gfx11_event_id = 5,
            .gfx12_event_id = 5,
        },
        {
            .name = IREE_HAL_AMDGPU_PROFILE_COUNTER_SV("SQ_WAVES_64"),
            .block_name = IREE_HAL_AMDGPU_PROFILE_COUNTER_SV("SQ"),
            .description = IREE_HAL_AMDGPU_PROFILE_COUNTER_SV(
                "Raw SQ_WAVES_64 values returned by aqlprofile."),
            .unit = IREE_HAL_PROFILE_COUNTER_UNIT_COUNT,
            .block_name_id = IREE_HAL_AMDGPU_AQLPROFILE_BLOCK_NAME_SQ,
            .gfx9_event_id = iree_hal_amdgpu_profile_counter_event_unsupported,
            .gfx10_event_id = 6,
            .gfx11_event_id = 6,
            .gfx12_event_id = 6,
        },
};

#undef IREE_HAL_AMDGPU_PROFILE_COUNTER_SV

// One resolved raw counter within a selected counter set.
typedef struct iree_hal_amdgpu_profile_counter_t {
  // Static counter descriptor used for metadata and event matching.
  const iree_hal_amdgpu_profile_counter_descriptor_t* descriptor;
  // Resolved aqlprofile hardware event request for the physical device.
  iree_hal_amdgpu_aqlprofile_pmc_event_t event;
  // Counter ordinal within its owning counter set.
  uint32_t counter_ordinal;
  // First uint64_t slot occupied by this counter in emitted sample records.
  uint32_t sample_value_offset;
  // Number of uint64_t slots occupied by this counter in emitted samples.
  uint32_t sample_value_count;
} iree_hal_amdgpu_profile_counter_t;

// One resolved counter set for one physical device.
typedef struct iree_hal_amdgpu_profile_counter_set_t {
  // Session-local counter set id referenced by counter and sample records.
  uint64_t counter_set_id;
  // Session-local physical device ordinal owning this counter set.
  uint32_t physical_device_ordinal;
  // Number of counters in |counters|.
  uint32_t counter_count;
  // Number of uint64_t values in each sample for this counter set.
  uint32_t sample_value_count;
  // Registered aqlprofile agent used when creating per-use sample handles.
  iree_hal_amdgpu_aqlprofile_agent_handle_t agent;
  // Contiguous aqlprofile event requests for all counters in this set.
  iree_hal_amdgpu_aqlprofile_pmc_event_t* events;
  // Resolved counters with emitted sample-value slices.
  iree_hal_amdgpu_profile_counter_t* counters;
  // Session-owned human-readable counter set name.
  iree_string_view_t name;
} iree_hal_amdgpu_profile_counter_set_t;

// Callback context used by aqlprofile memory allocation and copy hooks.
typedef struct iree_hal_amdgpu_profile_counter_memory_context_t {
  // HSA API table used by raw callback-status functions.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // GPU agent that must be able to access allocated aqlprofile memory.
  hsa_agent_t device_agent;
  // Host memory pools nearest to |device_agent|.
  const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools;
} iree_hal_amdgpu_profile_counter_memory_context_t;

// Per-queue/per-event-ring-slot mutable aqlprofile capture state.
struct iree_hal_amdgpu_profile_counter_sample_slot_t {
  // Callback context retained for the lifetime of |handle|.
  iree_hal_amdgpu_profile_counter_memory_context_t memory_context;
  // aqlprofile handle owning PM4 programs and output storage for this slot.
  iree_hal_amdgpu_aqlprofile_handle_t handle;
  // AQL PM4-IB packet templates referencing |handle|'s immutable PM4 programs.
  iree_hal_amdgpu_aqlprofile_pmc_aql_packets_t packets;
  // Producer-local sample id assigned when this slot is reserved for a
  // dispatch.
  uint64_t sample_id;
};

// Logical-device profiling session for selected hardware counters.
struct iree_hal_amdgpu_profile_counter_session_t {
  // Host allocator used for session and queue slot storage.
  iree_allocator_t host_allocator;
  // Borrowed HSA API table from the logical device.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Dynamically loaded aqlprofile SDK.
  iree_hal_amdgpu_libaqlprofile_t libaqlprofile;
  // Number of physical devices in |agent_handles|.
  iree_host_size_t physical_device_count;
  // Number of requested counter sets per physical device.
  uint32_t counter_set_count;
  // Registered aqlprofile agent handles indexed by physical device ordinal.
  iree_hal_amdgpu_aqlprofile_agent_handle_t* agent_handles;
  // Resolved counter sets indexed by physical device then counter-set ordinal.
  iree_hal_amdgpu_profile_counter_set_t* counter_sets;
  // Next nonzero producer-local counter sample id.
  iree_atomic_int64_t next_sample_id;
};

// Callback context used to count decoded aqlprofile values.
typedef struct iree_hal_amdgpu_profile_counter_count_context_t {
  // Counter set whose per-counter sample widths are being discovered.
  iree_hal_amdgpu_profile_counter_set_t* counter_set;
} iree_hal_amdgpu_profile_counter_count_context_t;

// Callback context used to copy decoded aqlprofile values.
typedef struct iree_hal_amdgpu_profile_counter_collect_context_t {
  // Counter set describing the emitted sample-value layout.
  const iree_hal_amdgpu_profile_counter_set_t* counter_set;
  // Destination value vector.
  uint64_t* values;
  // Per-counter value counts already written for the current sample.
  uint32_t* counter_value_counts;
} iree_hal_amdgpu_profile_counter_collect_context_t;

//===----------------------------------------------------------------------===//
// Raw HSA callbacks used by aqlprofile
//===----------------------------------------------------------------------===//

static hsa_status_t iree_hal_amdgpu_profile_hsa_memory_pool_allocate(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_amd_memory_pool_t memory_pool,
    size_t size, uint32_t flags, void** ptr) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  (void)libhsa;
  return hsa_amd_memory_pool_allocate(memory_pool, size, flags, ptr);
#else
  return libhsa->hsa_amd_memory_pool_allocate(memory_pool, size, flags, ptr);
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC
}

static hsa_status_t iree_hal_amdgpu_profile_hsa_memory_pool_free(
    const iree_hal_amdgpu_libhsa_t* libhsa, void* ptr) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  (void)libhsa;
  return hsa_amd_memory_pool_free(ptr);
#else
  return libhsa->hsa_amd_memory_pool_free(ptr);
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC
}

static hsa_status_t iree_hal_amdgpu_profile_hsa_agents_allow_access(
    const iree_hal_amdgpu_libhsa_t* libhsa, uint32_t num_agents,
    const hsa_agent_t* agents, const uint32_t* flags, const void* ptr) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  (void)libhsa;
  return hsa_amd_agents_allow_access(num_agents, agents, flags, ptr);
#else
  return libhsa->hsa_amd_agents_allow_access(num_agents, agents, flags, ptr);
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC
}

static hsa_status_t iree_hal_amdgpu_profile_hsa_memory_copy(
    const iree_hal_amdgpu_libhsa_t* libhsa, void* target, const void* source,
    size_t size) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  (void)libhsa;
  return hsa_memory_copy(target, source, size);
#else
  return libhsa->hsa_memory_copy(target, source, size);
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC
}

static hsa_status_t iree_hal_amdgpu_profile_counter_memory_alloc(
    void** ptr, uint64_t size,
    iree_hal_amdgpu_aqlprofile_buffer_desc_flags_t flags, void* user_data) {
  iree_hal_amdgpu_profile_counter_memory_context_t* context =
      (iree_hal_amdgpu_profile_counter_memory_context_t*)user_data;
  *ptr = NULL;
  if (size == 0) return HSA_STATUS_SUCCESS;
  if (!flags.host_access) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_amd_memory_pool_t memory_pool = context->host_memory_pools->coarse_pool;
  if (flags.memory_hint ==
      IREE_HAL_AMDGPU_AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED) {
    memory_pool = context->host_memory_pools->kernarg_pool;
  }
  if (!memory_pool.handle) return HSA_STATUS_ERROR_INVALID_ALLOCATION;

  hsa_status_t status = iree_hal_amdgpu_profile_hsa_memory_pool_allocate(
      context->libhsa, memory_pool, (size_t)size,
      HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, ptr);
  if (status != HSA_STATUS_SUCCESS) return status;
  memset(*ptr, 0, (size_t)size);

  if (flags.device_access) {
    status = iree_hal_amdgpu_profile_hsa_agents_allow_access(
        context->libhsa, /*num_agents=*/1, &context->device_agent,
        /*flags=*/NULL, *ptr);
    if (status != HSA_STATUS_SUCCESS) {
      iree_hal_amdgpu_profile_hsa_memory_pool_free(context->libhsa, *ptr);
      *ptr = NULL;
    }
  }
  return status;
}

static void iree_hal_amdgpu_profile_counter_memory_dealloc(void* ptr,
                                                           void* user_data) {
  if (!ptr) return;
  iree_hal_amdgpu_profile_counter_memory_context_t* context =
      (iree_hal_amdgpu_profile_counter_memory_context_t*)user_data;
  iree_hal_amdgpu_profile_hsa_memory_pool_free(context->libhsa, ptr);
}

static hsa_status_t iree_hal_amdgpu_profile_counter_memory_copy(
    void* target, const void* source, size_t size, void* user_data) {
  if (size == 0) return HSA_STATUS_SUCCESS;
  iree_hal_amdgpu_profile_counter_memory_context_t* context =
      (iree_hal_amdgpu_profile_counter_memory_context_t*)user_data;
  return iree_hal_amdgpu_profile_hsa_memory_copy(context->libhsa, target,
                                                 source, size);
}

static const iree_hal_amdgpu_profile_counter_descriptor_t*
iree_hal_amdgpu_profile_counter_find_descriptor(iree_string_view_t name) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amdgpu_profile_counter_descriptors); ++i) {
    const iree_hal_amdgpu_profile_counter_descriptor_t* descriptor =
        &iree_hal_amdgpu_profile_counter_descriptors[i];
    if (iree_string_view_equal(name, descriptor->name)) return descriptor;
  }
  return NULL;
}

static iree_status_t iree_hal_amdgpu_profile_counter_resolve_event(
    const iree_hal_amdgpu_profile_counter_descriptor_t* descriptor,
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    iree_hal_amdgpu_aqlprofile_pmc_event_t* out_event) {
  uint32_t event_id = iree_hal_amdgpu_profile_counter_event_unsupported;
  switch (gfxip_version.major) {
    case 9:
      event_id = descriptor->gfx9_event_id;
      break;
    case 10:
      event_id = descriptor->gfx10_event_id;
      break;
    case 11:
      event_id = descriptor->gfx11_event_id;
      break;
    case 12:
      event_id = descriptor->gfx12_event_id;
      break;
    default:
      break;
  }
  if (IREE_UNLIKELY(event_id ==
                    iree_hal_amdgpu_profile_counter_event_unsupported)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU counter '%.*s' is not mapped for gfx%u.%u.%u",
        (int)descriptor->name.size, descriptor->name.data, gfxip_version.major,
        gfxip_version.minor, gfxip_version.stepping);
  }

  *out_event = (iree_hal_amdgpu_aqlprofile_pmc_event_t){
      .block_index = 0,
      .event_id = event_id,
      .block_name = descriptor->block_name_id,
  };
  return iree_ok_status();
}

static bool iree_hal_amdgpu_profile_counter_events_equal(
    iree_hal_amdgpu_aqlprofile_pmc_event_t lhs,
    iree_hal_amdgpu_aqlprofile_pmc_event_t rhs) {
  return lhs.block_index == rhs.block_index && lhs.event_id == rhs.event_id &&
         lhs.flags.raw == rhs.flags.raw && lhs.block_name == rhs.block_name;
}

static bool iree_hal_amdgpu_profile_counter_find_index_by_event(
    const iree_hal_amdgpu_profile_counter_set_t* counter_set,
    iree_hal_amdgpu_aqlprofile_pmc_event_t event,
    uint32_t* out_counter_ordinal) {
  for (uint32_t i = 0; i < counter_set->counter_count; ++i) {
    const iree_hal_amdgpu_profile_counter_t* counter =
        &counter_set->counters[i];
    if (iree_hal_amdgpu_profile_counter_events_equal(counter->event, event)) {
      *out_counter_ordinal = i;
      return true;
    }
  }
  return false;
}

static hsa_status_t iree_hal_amdgpu_profile_counter_count_callback(
    iree_hal_amdgpu_aqlprofile_pmc_event_t event, uint64_t counter_id,
    uint64_t counter_value, void* user_data) {
  (void)counter_id;
  (void)counter_value;
  iree_hal_amdgpu_profile_counter_count_context_t* context =
      (iree_hal_amdgpu_profile_counter_count_context_t*)user_data;
  uint32_t counter_ordinal = 0;
  if (!iree_hal_amdgpu_profile_counter_find_index_by_event(
          context->counter_set, event, &counter_ordinal)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ++context->counter_set->counters[counter_ordinal].sample_value_count;
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t iree_hal_amdgpu_profile_counter_collect_callback(
    iree_hal_amdgpu_aqlprofile_pmc_event_t event, uint64_t counter_id,
    uint64_t counter_value, void* user_data) {
  (void)counter_id;
  iree_hal_amdgpu_profile_counter_collect_context_t* context =
      (iree_hal_amdgpu_profile_counter_collect_context_t*)user_data;
  uint32_t counter_ordinal = 0;
  if (!iree_hal_amdgpu_profile_counter_find_index_by_event(
          context->counter_set, event, &counter_ordinal)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const iree_hal_amdgpu_profile_counter_t* counter =
      &context->counter_set->counters[counter_ordinal];
  uint32_t* counter_value_count =
      &context->counter_value_counts[counter_ordinal];
  if (*counter_value_count >= counter->sample_value_count) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  context->values[counter->sample_value_offset + *counter_value_count] =
      counter_value;
  ++*counter_value_count;
  return HSA_STATUS_SUCCESS;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_profile_counter_session_t
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_profile_counter_initialize_selection(
    const iree_hal_profile_counter_set_selection_t* selection,
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    iree_hal_amdgpu_profile_counter_set_t* counter_set) {
  if (IREE_UNLIKELY(selection->flags !=
                    IREE_HAL_PROFILE_COUNTER_SET_SELECTION_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU counter set flags 0x%x",
                            selection->flags);
  }
  if (IREE_UNLIKELY(selection->counter_name_count == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU counter profiling requires at least one counter per set");
  }
  if (IREE_UNLIKELY(selection->counter_name_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU counter count exceeds uint32_t");
  }

  for (iree_host_size_t i = 0; i < selection->counter_name_count; ++i) {
    const iree_hal_amdgpu_profile_counter_descriptor_t* descriptor =
        iree_hal_amdgpu_profile_counter_find_descriptor(
            selection->counter_names[i]);
    if (IREE_UNLIKELY(!descriptor)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "unsupported AMDGPU counter '%.*s'; supported counters: "
          "SQ_WAVES, SQ_WAVES_32, SQ_WAVES_64",
          (int)selection->counter_names[i].size,
          selection->counter_names[i].data);
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(selection->counter_names[i],
                                 selection->counter_names[j])) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "duplicate AMDGPU counter '%.*s' in one counter set",
            (int)selection->counter_names[i].size,
            selection->counter_names[i].data);
      }
    }

    const uint32_t counter_ordinal = (uint32_t)i;
    iree_hal_amdgpu_aqlprofile_pmc_event_t event = {0};
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_counter_resolve_event(
        descriptor, gfxip_version, &event));
    counter_set->events[i] = event;
    counter_set->counters[i] = (iree_hal_amdgpu_profile_counter_t){
        .descriptor = descriptor,
        .event = event,
        .counter_ordinal = counter_ordinal,
    };
  }
  counter_set->counter_count = (uint32_t)selection->counter_name_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_counter_register_agent(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_libaqlprofile_t* libaqlprofile,
    hsa_agent_t device_agent,
    iree_hal_amdgpu_aqlprofile_agent_handle_t* out_agent_handle) {
  char agent_name[64] = {0};
  iree_hal_amdgpu_aqlprofile_agent_info_v1_t agent_info;
  memset(&agent_info, 0, sizeof(agent_info));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent, HSA_AGENT_INFO_NAME, agent_name));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_XCC, &agent_info.xcc_num));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES,
      &agent_info.se_num));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
      &agent_info.cu_num));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE,
      &agent_info.shader_arrays_per_se));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN, &agent_info.domain));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &agent_info.location_id));
  agent_info.agent_gfxip = agent_name;
  IREE_RETURN_IF_AQLPROFILE_ERROR(
      libaqlprofile,
      libaqlprofile->aqlprofile_register_agent_info(
          out_agent_handle, &agent_info,
          IREE_HAL_AMDGPU_AQLPROFILE_AGENT_VERSION_V1),
      "registering AMDGPU profiling agent");
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_counter_count_values(
    const iree_hal_amdgpu_libaqlprofile_t* libaqlprofile,
    iree_hal_amdgpu_aqlprofile_handle_t handle,
    iree_hal_amdgpu_profile_counter_set_t* counter_set) {
  for (uint32_t i = 0; i < counter_set->counter_count; ++i) {
    counter_set->counters[i].sample_value_count = 0;
    counter_set->counters[i].sample_value_offset = 0;
  }

  iree_hal_amdgpu_profile_counter_count_context_t context = {
      .counter_set = counter_set,
  };
  IREE_RETURN_IF_AQLPROFILE_ERROR(
      libaqlprofile,
      libaqlprofile->aqlprofile_pmc_iterate_data(
          handle, iree_hal_amdgpu_profile_counter_count_callback, &context),
      "iterating AMDGPU counter metadata");

  iree_host_size_t sample_value_count = 0;
  for (uint32_t i = 0; i < counter_set->counter_count; ++i) {
    iree_hal_amdgpu_profile_counter_t* counter = &counter_set->counters[i];
    if (IREE_UNLIKELY(counter->sample_value_count == 0)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "aqlprofile reported zero values for selected AMDGPU counter '%.*s'",
          (int)counter->descriptor->name.size, counter->descriptor->name.data);
    }
    if (IREE_UNLIKELY(sample_value_count > UINT32_MAX)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU counter sample value count overflow");
    }
    counter->sample_value_offset = (uint32_t)sample_value_count;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(sample_value_count,
                                                  counter->sample_value_count,
                                                  &sample_value_count))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU counter sample value count overflow");
    }
  }
  if (IREE_UNLIKELY(sample_value_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU counter sample value count overflow");
  }
  counter_set->sample_value_count = (uint32_t)sample_value_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_counter_create_packets(
    const iree_hal_amdgpu_libaqlprofile_t* libaqlprofile,
    const iree_hal_amdgpu_profile_counter_set_t* counter_set,
    const iree_hal_amdgpu_profile_counter_memory_context_t* memory_context,
    iree_hal_amdgpu_aqlprofile_handle_t* out_handle,
    iree_hal_amdgpu_aqlprofile_pmc_aql_packets_t* out_packets) {
  iree_hal_amdgpu_aqlprofile_pmc_profile_t profile = {
      .agent = counter_set->agent,
      .events = counter_set->events,
      .event_count = counter_set->counter_count,
  };
  IREE_RETURN_IF_AQLPROFILE_ERROR(
      libaqlprofile,
      libaqlprofile->aqlprofile_pmc_create_packets(
          out_handle, out_packets, profile,
          iree_hal_amdgpu_profile_counter_memory_alloc,
          iree_hal_amdgpu_profile_counter_memory_dealloc,
          iree_hal_amdgpu_profile_counter_memory_copy, (void*)memory_context),
      "creating AMDGPU counter PM4 packets");
  return iree_ok_status();
}

static void iree_hal_amdgpu_profile_counter_destroy_packets(
    const iree_hal_amdgpu_libaqlprofile_t* libaqlprofile,
    iree_hal_amdgpu_aqlprofile_handle_t* handle) {
  if (!handle->handle) return;
  libaqlprofile->aqlprofile_pmc_delete_packets(*handle);
  handle->handle = 0;
}

static iree_status_t iree_hal_amdgpu_profile_counter_initialize_set(
    const iree_hal_device_profiling_options_t* options,
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_host_size_t selection_ordinal,
    const iree_hal_amdgpu_libaqlprofile_t* libaqlprofile,
    iree_hal_amdgpu_profile_counter_session_t* session,
    iree_hal_amdgpu_profile_counter_t** inout_counter_storage,
    iree_hal_amdgpu_aqlprofile_pmc_event_t** inout_event_storage,
    char** inout_string_storage,
    iree_hal_amdgpu_profile_counter_set_t* out_counter_set) {
  const iree_hal_profile_counter_set_selection_t* selection =
      &options->counter_sets[selection_ordinal];
  iree_string_view_t set_name = selection->name;
  if (iree_string_view_is_empty(set_name)) {
    set_name = iree_hal_amdgpu_profile_counter_set_name();
  }
  char* string_storage = *inout_string_storage;
  memcpy(string_storage, set_name.data, set_name.size);
  *inout_string_storage = string_storage + set_name.size;

  *out_counter_set = (iree_hal_amdgpu_profile_counter_set_t){
      .counter_set_id =
          ((uint64_t)(uint32_t)physical_device->device_ordinal << 32) |
          (uint64_t)(selection_ordinal + 1),
      .physical_device_ordinal = (uint32_t)physical_device->device_ordinal,
      .counter_count = 0,
      .sample_value_count = 0,
      .agent = session->agent_handles[physical_device->device_ordinal],
      .events = *inout_event_storage,
      .counters = *inout_counter_storage,
      .name = iree_make_string_view(string_storage, set_name.size),
  };
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_counter_initialize_selection(
      selection, physical_device->gfxip_version, out_counter_set));
  *inout_counter_storage += out_counter_set->counter_count;
  *inout_event_storage += out_counter_set->counter_count;

  for (uint32_t i = 0; i < out_counter_set->counter_count; ++i) {
    const iree_hal_amdgpu_profile_counter_t* counter =
        &out_counter_set->counters[i];
    bool valid = false;
    IREE_RETURN_IF_AQLPROFILE_ERROR(
        libaqlprofile,
        libaqlprofile->aqlprofile_validate_pmc_event(
            session->agent_handles[physical_device->device_ordinal],
            &counter->event, &valid),
        "validating AMDGPU counter event");
    if (IREE_UNLIKELY(!valid)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU counter '%.*s' is not valid on physical device %" PRIhsz,
          (int)counter->descriptor->name.size, counter->descriptor->name.data,
          physical_device->device_ordinal);
    }
  }

  iree_hal_amdgpu_profile_counter_memory_context_t memory_context = {
      .libhsa = session->libhsa,
      .device_agent = physical_device->device_agent,
      .host_memory_pools = &physical_device->host_memory_pools,
  };
  iree_hal_amdgpu_aqlprofile_handle_t metadata_handle = {0};
  iree_hal_amdgpu_aqlprofile_pmc_aql_packets_t metadata_packets;
  memset(&metadata_packets, 0, sizeof(metadata_packets));
  iree_status_t status = iree_hal_amdgpu_profile_counter_create_packets(
      libaqlprofile, out_counter_set, &memory_context, &metadata_handle,
      &metadata_packets);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_counter_count_values(
        libaqlprofile, metadata_handle, out_counter_set);
  }
  iree_hal_amdgpu_profile_counter_destroy_packets(libaqlprofile,
                                                  &metadata_handle);
  return status;
}

iree_status_t iree_hal_amdgpu_profile_counter_session_create(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_device_profiling_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_counter_session_t** out_session) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_session = NULL;

  if (!iree_hal_device_profiling_options_have_counter_sets(options)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!options->sink)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU hardware counter profiling requires a profile sink");
  }
  if (IREE_UNLIKELY(options->counter_set_count > UINT32_MAX)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU counter set count exceeds uint32_t");
  }
  if (IREE_UNLIKELY(logical_device->physical_device_count > UINT32_MAX)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU physical device count exceeds uint32_t");
  }

  iree_host_size_t counter_set_total_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          logical_device->physical_device_count, options->counter_set_count,
          &counter_set_total_count))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU counter set count overflow");
  }
  iree_host_size_t counter_total_count = 0;
  iree_host_size_t string_storage_length = 0;
  for (iree_host_size_t i = 0; i < options->counter_set_count; ++i) {
    if (IREE_UNLIKELY(options->counter_sets[i].counter_name_count == 0)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU counter profiling requires at least one counter per set");
    }
    iree_host_size_t replicated_counter_count = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
                          logical_device->physical_device_count,
                          options->counter_sets[i].counter_name_count,
                          &replicated_counter_count) ||
                      !iree_host_size_checked_add(counter_total_count,
                                                  replicated_counter_count,
                                                  &counter_total_count))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU counter count overflow");
    }

    iree_string_view_t set_name = options->counter_sets[i].name;
    if (iree_string_view_is_empty(set_name)) {
      set_name = iree_hal_amdgpu_profile_counter_set_name();
    }
    iree_host_size_t replicated_length = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(logical_device->physical_device_count,
                                        set_name.size, &replicated_length) ||
            !iree_host_size_checked_add(string_storage_length,
                                        replicated_length,
                                        &string_storage_length))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU counter set name storage overflow");
    }
  }

  iree_host_size_t agent_handles_offset = 0;
  iree_host_size_t counter_sets_offset = 0;
  iree_host_size_t counters_offset = 0;
  iree_host_size_t events_offset = 0;
  iree_host_size_t string_storage_offset = 0;
  iree_host_size_t session_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_hal_amdgpu_profile_counter_session_t), &session_size,
              IREE_STRUCT_FIELD(logical_device->physical_device_count,
                                iree_hal_amdgpu_aqlprofile_agent_handle_t,
                                &agent_handles_offset),
              IREE_STRUCT_FIELD(counter_set_total_count,
                                iree_hal_amdgpu_profile_counter_set_t,
                                &counter_sets_offset),
              IREE_STRUCT_FIELD(counter_total_count,
                                iree_hal_amdgpu_profile_counter_t,
                                &counters_offset),
              IREE_STRUCT_FIELD(counter_total_count,
                                iree_hal_amdgpu_aqlprofile_pmc_event_t,
                                &events_offset),
              IREE_STRUCT_FIELD(string_storage_length, char,
                                &string_storage_offset)));

  iree_hal_amdgpu_profile_counter_session_t* session = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, session_size, (void**)&session));
  memset(session, 0, session_size);
  session->host_allocator = host_allocator;
  session->libhsa = &logical_device->system->libhsa;
  session->physical_device_count = logical_device->physical_device_count;
  session->counter_set_count = (uint32_t)options->counter_set_count;
  session->agent_handles =
      (iree_hal_amdgpu_aqlprofile_agent_handle_t*)((uint8_t*)session +
                                                   agent_handles_offset);
  session->counter_sets =
      (iree_hal_amdgpu_profile_counter_set_t*)((uint8_t*)session +
                                               counter_sets_offset);
  iree_hal_amdgpu_profile_counter_t* counter_storage =
      (iree_hal_amdgpu_profile_counter_t*)((uint8_t*)session + counters_offset);
  iree_hal_amdgpu_aqlprofile_pmc_event_t* event_storage =
      (iree_hal_amdgpu_aqlprofile_pmc_event_t*)((uint8_t*)session +
                                                events_offset);
  iree_atomic_store(&session->next_sample_id, 1, iree_memory_order_relaxed);

  iree_status_t status = iree_hal_amdgpu_libaqlprofile_initialize(
      session->libhsa, iree_string_view_list_empty(), host_allocator,
      &session->libaqlprofile);
  for (iree_host_size_t i = 0;
       i < logical_device->physical_device_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    status = iree_hal_amdgpu_profile_counter_register_agent(
        session->libhsa, &session->libaqlprofile, physical_device->device_agent,
        &session->agent_handles[physical_device->device_ordinal]);
  }
  char* string_storage = (char*)session + string_storage_offset;
  for (iree_host_size_t i = 0;
       i < logical_device->physical_device_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[i];
    for (iree_host_size_t j = 0;
         j < options->counter_set_count && iree_status_is_ok(status); ++j) {
      const iree_host_size_t counter_set_index =
          physical_device->device_ordinal * options->counter_set_count + j;
      status = iree_hal_amdgpu_profile_counter_initialize_set(
          options, physical_device, j, &session->libaqlprofile, session,
          &counter_storage, &event_storage, &string_storage,
          &session->counter_sets[counter_set_index]);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_session = session;
  } else {
    iree_hal_amdgpu_libaqlprofile_deinitialize(&session->libaqlprofile);
    iree_allocator_free(host_allocator, session);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_profile_counter_session_destroy(
    iree_hal_amdgpu_profile_counter_session_t* session) {
  if (!session) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = session->host_allocator;
  iree_hal_amdgpu_libaqlprofile_deinitialize(&session->libaqlprofile);
  iree_allocator_free(host_allocator, session);
  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdgpu_profile_counter_session_is_active(
    const iree_hal_amdgpu_profile_counter_session_t* session) {
  return session && session->counter_set_count != 0;
}

static iree_status_t iree_hal_amdgpu_profile_counter_set_record_size(
    const iree_hal_amdgpu_profile_counter_set_t* counter_set,
    iree_host_size_t* out_record_size) {
  return IREE_STRUCT_LAYOUT(
      0, out_record_size,
      IREE_STRUCT_FIELD(1, iree_hal_profile_counter_set_record_t, NULL),
      IREE_STRUCT_FIELD(counter_set->name.size, char, NULL));
}

static iree_status_t iree_hal_amdgpu_profile_counter_record_size(
    const iree_hal_amdgpu_profile_counter_t* counter,
    iree_host_size_t* out_record_size) {
  const iree_hal_amdgpu_profile_counter_descriptor_t* descriptor =
      counter->descriptor;
  return IREE_STRUCT_LAYOUT(
      0, out_record_size,
      IREE_STRUCT_FIELD(1, iree_hal_profile_counter_record_t, NULL),
      IREE_STRUCT_FIELD(descriptor->block_name.size + descriptor->name.size +
                            descriptor->description.size,
                        char, NULL));
}

static iree_status_t iree_hal_amdgpu_profile_counter_metadata_size(
    const iree_hal_amdgpu_profile_counter_session_t* session,
    iree_host_size_t* out_counter_set_size,
    iree_host_size_t* out_counter_size) {
  iree_host_size_t counter_set_size = 0;
  iree_host_size_t counter_size = 0;
  const iree_host_size_t counter_set_total_count =
      session->physical_device_count * session->counter_set_count;
  for (iree_host_size_t i = 0; i < counter_set_total_count; ++i) {
    const iree_hal_amdgpu_profile_counter_set_t* counter_set =
        &session->counter_sets[i];
    iree_host_size_t record_size = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_counter_set_record_size(
        counter_set, &record_size));
    if (IREE_UNLIKELY(!iree_host_size_checked_add(counter_set_size, record_size,
                                                  &counter_set_size))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU counter set metadata overflow");
    }

    for (uint32_t j = 0; j < counter_set->counter_count; ++j) {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_counter_record_size(
          &counter_set->counters[j], &record_size));
      if (IREE_UNLIKELY(!iree_host_size_checked_add(counter_size, record_size,
                                                    &counter_size))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU counter metadata overflow");
      }
    }
  }
  *out_counter_set_size = counter_set_size;
  *out_counter_size = counter_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_counter_pack_counter_set_record(
    const iree_hal_amdgpu_profile_counter_set_t* counter_set,
    uint8_t** inout_storage_ptr) {
  iree_host_size_t record_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_counter_set_record_size(
      counter_set, &record_size));
  if (IREE_UNLIKELY(record_size > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU counter set metadata record exceeds uint32_t");
  }

  iree_hal_profile_counter_set_record_t record =
      iree_hal_profile_counter_set_record_default();
  record.record_length = (uint32_t)record_size;
  record.counter_set_id = counter_set->counter_set_id;
  record.physical_device_ordinal = counter_set->physical_device_ordinal;
  record.counter_count = counter_set->counter_count;
  record.sample_value_count = counter_set->sample_value_count;
  record.name_length = (uint32_t)counter_set->name.size;

  uint8_t* storage_ptr = *inout_storage_ptr;
  memcpy(storage_ptr, &record, sizeof(record));
  memcpy(storage_ptr + sizeof(record), counter_set->name.data,
         counter_set->name.size);
  *inout_storage_ptr = storage_ptr + record_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_counter_pack_counter_record(
    const iree_hal_amdgpu_profile_counter_set_t* counter_set,
    const iree_hal_amdgpu_profile_counter_t* counter,
    uint8_t** inout_storage_ptr) {
  iree_host_size_t record_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_counter_record_size(counter, &record_size));
  if (IREE_UNLIKELY(record_size > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU counter metadata record exceeds uint32_t");
  }

  const iree_hal_amdgpu_profile_counter_descriptor_t* descriptor =
      counter->descriptor;
  iree_hal_profile_counter_record_t record =
      iree_hal_profile_counter_record_default();
  record.record_length = (uint32_t)record_size;
  record.flags = IREE_HAL_PROFILE_COUNTER_FLAG_RAW;
  record.unit = descriptor->unit;
  record.physical_device_ordinal = counter_set->physical_device_ordinal;
  record.counter_set_id = counter_set->counter_set_id;
  record.counter_ordinal = counter->counter_ordinal;
  record.sample_value_offset = counter->sample_value_offset;
  record.sample_value_count = counter->sample_value_count;
  record.block_name_length = (uint32_t)descriptor->block_name.size;
  record.name_length = (uint32_t)descriptor->name.size;
  record.description_length = (uint32_t)descriptor->description.size;

  uint8_t* storage_ptr = *inout_storage_ptr;
  memcpy(storage_ptr, &record, sizeof(record));
  uint8_t* string_ptr = storage_ptr + sizeof(record);
  memcpy(string_ptr, descriptor->block_name.data, descriptor->block_name.size);
  string_ptr += descriptor->block_name.size;
  memcpy(string_ptr, descriptor->name.data, descriptor->name.size);
  string_ptr += descriptor->name.size;
  memcpy(string_ptr, descriptor->description.data,
         descriptor->description.size);
  *inout_storage_ptr = storage_ptr + record_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_counter_write_metadata_chunk(
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name, iree_string_view_t content_type,
    const uint8_t* storage, iree_host_size_t storage_size) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = content_type;
  metadata.name = stream_name;
  metadata.session_id = session_id;
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span(storage, storage_size);
  return iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
}

iree_status_t iree_hal_amdgpu_profile_counter_session_write_metadata(
    const iree_hal_amdgpu_profile_counter_session_t* session,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name) {
  if (!iree_hal_amdgpu_profile_counter_session_is_active(session)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t counter_set_storage_size = 0;
  iree_host_size_t counter_storage_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_profile_counter_metadata_size(
              session, &counter_set_storage_size, &counter_storage_size));

  uint8_t* counter_set_storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(session->host_allocator, counter_set_storage_size,
                            (void**)&counter_set_storage));
  uint8_t* counter_storage = NULL;
  iree_status_t status = iree_allocator_malloc(
      session->host_allocator, counter_storage_size, (void**)&counter_storage);

  const iree_host_size_t counter_set_total_count =
      session->physical_device_count * session->counter_set_count;
  uint8_t* counter_set_ptr = counter_set_storage;
  uint8_t* counter_ptr = counter_storage;
  for (iree_host_size_t i = 0;
       i < counter_set_total_count && iree_status_is_ok(status); ++i) {
    const iree_hal_amdgpu_profile_counter_set_t* counter_set =
        &session->counter_sets[i];
    status = iree_hal_amdgpu_profile_counter_pack_counter_set_record(
        counter_set, &counter_set_ptr);
    for (uint32_t j = 0;
         j < counter_set->counter_count && iree_status_is_ok(status); ++j) {
      status = iree_hal_amdgpu_profile_counter_pack_counter_record(
          counter_set, &counter_set->counters[j], &counter_ptr);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_counter_write_metadata_chunk(
        sink, session_id, stream_name,
        IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS, counter_set_storage,
        counter_set_storage_size);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_counter_write_metadata_chunk(
        sink, session_id, stream_name, IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS,
        counter_storage, counter_storage_size);
  }

  iree_allocator_free(session->host_allocator, counter_storage);
  iree_allocator_free(session->host_allocator, counter_set_storage);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Queue-local counter sample slots
//===----------------------------------------------------------------------===//

static const iree_hal_amdgpu_profile_counter_set_t*
iree_hal_amdgpu_host_queue_profile_counter_set(
    const iree_hal_amdgpu_host_queue_t* queue, uint32_t counter_set_ordinal) {
  const iree_hal_amdgpu_profile_counter_session_t* session =
      queue->profiling.counter_session;
  if (!session || counter_set_ordinal >= session->counter_set_count ||
      queue->device_ordinal >= session->physical_device_count) {
    return NULL;
  }
  return &session->counter_sets[queue->device_ordinal *
                                    session->counter_set_count +
                                counter_set_ordinal];
}

static iree_hal_amdgpu_profile_counter_sample_slot_t*
iree_hal_amdgpu_host_queue_profile_counter_slot(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint32_t counter_set_ordinal) {
  const uint32_t event_index =
      (uint32_t)(event_position & queue->profiling.dispatch_event_mask);
  const iree_host_size_t slot_index =
      (iree_host_size_t)event_index * queue->profiling.counter_set_count +
      counter_set_ordinal;
  return &queue->profiling.counter_sample_slots[slot_index];
}

iree_status_t iree_hal_amdgpu_host_queue_enable_profile_counters(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_counter_session_t* session) {
  if (!iree_hal_amdgpu_profile_counter_session_is_active(session)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  if (IREE_UNLIKELY(!queue->profiling.dispatch_event_capacity)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU counter profiling requires dispatch event storage");
  }
  if (IREE_UNLIKELY(!iree_any_bit_set(
          queue->vendor_packet_capabilities,
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU counter profiling requires AQL PM4-IB packet support");
  }
  if (IREE_UNLIKELY(queue->profiling.counter_sample_slots)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU counter profiling is already enabled");
  }

  iree_host_size_t slot_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          queue->profiling.dispatch_event_capacity, session->counter_set_count,
          &slot_count))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU counter sample slot count overflow");
  }
  iree_host_size_t slot_storage_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &slot_storage_size,
              IREE_STRUCT_FIELD(slot_count,
                                iree_hal_amdgpu_profile_counter_sample_slot_t,
                                NULL)));
  iree_hal_amdgpu_profile_counter_sample_slot_t* slots = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(queue->host_allocator, slot_storage_size,
                                (void**)&slots));
  memset(slots, 0, slot_storage_size);

  queue->profiling.counter_session = session;
  queue->profiling.counter_sample_slots = slots;
  queue->profiling.counter_set_count = session->counter_set_count;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_host_queue_disable_profile_counters(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (!queue->profiling.counter_sample_slots) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_profile_counter_session_t* session =
      queue->profiling.counter_session;
  const iree_host_size_t slot_count =
      (iree_host_size_t)queue->profiling.dispatch_event_capacity *
      queue->profiling.counter_set_count;
  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    iree_hal_amdgpu_profile_counter_destroy_packets(
        &session->libaqlprofile,
        &queue->profiling.counter_sample_slots[i].handle);
  }
  iree_allocator_free(queue->host_allocator,
                      queue->profiling.counter_sample_slots);
  queue->profiling.counter_session = NULL;
  queue->profiling.counter_sample_slots = NULL;
  queue->profiling.counter_set_count = 0;

  IREE_TRACE_ZONE_END(z0);
}

uint32_t iree_hal_amdgpu_host_queue_profile_counter_packet_count(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation) {
  if (!reservation.event_count || !queue->profiling.counter_session) return 0;
  return reservation.event_count * queue->profiling.counter_set_count *
         iree_hal_amdgpu_profile_counter_packets_per_set;
}

uint32_t iree_hal_amdgpu_host_queue_profile_counter_set_count(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation) {
  if (!reservation.event_count || !queue->profiling.counter_session) return 0;
  return queue->profiling.counter_set_count;
}

iree_status_t iree_hal_amdgpu_host_queue_prepare_profile_counter_samples(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation) {
  iree_hal_amdgpu_profile_counter_session_t* session =
      queue->profiling.counter_session;
  if (!reservation.event_count || !session) return iree_ok_status();

  for (uint32_t event_ordinal = 0; event_ordinal < reservation.event_count;
       ++event_ordinal) {
    const uint64_t event_position =
        reservation.first_event_position + event_ordinal;
    for (uint32_t counter_set_ordinal = 0;
         counter_set_ordinal < queue->profiling.counter_set_count;
         ++counter_set_ordinal) {
      const iree_hal_amdgpu_profile_counter_set_t* counter_set =
          iree_hal_amdgpu_host_queue_profile_counter_set(queue,
                                                         counter_set_ordinal);
      if (IREE_UNLIKELY(!counter_set)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "AMDGPU counter set is not available");
      }
      iree_hal_amdgpu_profile_counter_sample_slot_t* slot =
          iree_hal_amdgpu_host_queue_profile_counter_slot(queue, event_position,
                                                          counter_set_ordinal);
      if (!slot->handle.handle) {
        iree_hal_amdgpu_logical_device_t* logical_device =
            (iree_hal_amdgpu_logical_device_t*)queue->logical_device;
        iree_hal_amdgpu_physical_device_t* physical_device =
            logical_device->physical_devices[queue->device_ordinal];
        slot->memory_context =
            (iree_hal_amdgpu_profile_counter_memory_context_t){
                .libhsa = session->libhsa,
                .device_agent = physical_device->device_agent,
                .host_memory_pools = &physical_device->host_memory_pools,
            };
        IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_counter_create_packets(
            &session->libaqlprofile, counter_set, &slot->memory_context,
            &slot->handle, &slot->packets));
      }
      slot->sample_id = (uint64_t)iree_atomic_fetch_add(
          &session->next_sample_id, 1, iree_memory_order_relaxed);
    }
  }
  return iree_ok_status();
}

static void iree_hal_amdgpu_host_queue_emplace_profile_counter_packet(
    const iree_hsa_amd_aql_pm4_ib_packet_t* source_packet,
    iree_hal_amdgpu_aql_packet_t* packet,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint16_t* out_header,
    uint16_t* out_setup) {
  memcpy((uint8_t*)&packet->pm4_ib + sizeof(uint32_t),
         (const uint8_t*)source_packet + sizeof(uint32_t),
         sizeof(*source_packet) - sizeof(uint32_t));
  packet->pm4_ib.completion_signal = completion_signal;
  *out_setup = IREE_HSA_AMD_AQL_FORMAT_PM4_IB;
  *out_header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC, packet_control);
}

static void iree_hal_amdgpu_host_queue_emplace_profile_counter_packet_at(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hsa_amd_aql_pm4_ib_packet_t* source_packet,
    uint64_t first_packet_id, uint32_t packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups) {
  iree_hal_amdgpu_aql_packet_t* packet = iree_hal_amdgpu_aql_ring_packet(
      &queue->aql_ring, first_packet_id + packet_index);
  iree_hal_amdgpu_host_queue_emplace_profile_counter_packet(
      source_packet, packet, packet_control, iree_hsa_signal_null(),
      &packet_headers[packet_index], &packet_setups[packet_index]);
}

void iree_hal_amdgpu_host_queue_emplace_profile_counter_start_packets(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint32_t counter_set_count, uint64_t first_packet_id,
    uint32_t first_packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups) {
  for (uint32_t i = 0; i < counter_set_count; ++i) {
    iree_hal_amdgpu_profile_counter_sample_slot_t* slot =
        iree_hal_amdgpu_host_queue_profile_counter_slot(queue, event_position,
                                                        i);
    iree_hal_amdgpu_host_queue_emplace_profile_counter_packet_at(
        queue, &slot->packets.start_packet, first_packet_id,
        first_packet_index + i, packet_control, packet_headers, packet_setups);
  }
}

void iree_hal_amdgpu_host_queue_emplace_profile_counter_read_stop_packets(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint32_t counter_set_count, uint64_t first_packet_id,
    uint32_t first_packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups) {
  for (uint32_t i = 0; i < counter_set_count; ++i) {
    iree_hal_amdgpu_profile_counter_sample_slot_t* slot =
        iree_hal_amdgpu_host_queue_profile_counter_slot(queue, event_position,
                                                        i);
    iree_hal_amdgpu_host_queue_emplace_profile_counter_packet_at(
        queue, &slot->packets.read_packet, first_packet_id,
        first_packet_index + i * 2u, packet_control, packet_headers,
        packet_setups);
    iree_hal_amdgpu_host_queue_emplace_profile_counter_packet_at(
        queue, &slot->packets.stop_packet, first_packet_id,
        first_packet_index + i * 2u + 1u, packet_control, packet_headers,
        packet_setups);
  }
}

static void iree_hal_amdgpu_host_queue_commit_profile_counter_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hsa_amd_aql_pm4_ib_packet_t* source_packet, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  uint16_t header = 0;
  uint16_t setup = 0;
  iree_hal_amdgpu_host_queue_emplace_profile_counter_packet(
      source_packet, packet, packet_control, iree_hsa_signal_null(), &header,
      &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

void iree_hal_amdgpu_host_queue_commit_profile_counter_start_packets(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint32_t counter_set_count, uint64_t first_packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control) {
  for (uint32_t i = 0; i < counter_set_count; ++i) {
    iree_hal_amdgpu_profile_counter_sample_slot_t* slot =
        iree_hal_amdgpu_host_queue_profile_counter_slot(queue, event_position,
                                                        i);
    iree_hal_amdgpu_host_queue_commit_profile_counter_packet(
        queue, &slot->packets.start_packet, first_packet_id + i,
        packet_control);
  }
}

void iree_hal_amdgpu_host_queue_commit_profile_counter_read_stop_packets(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint32_t counter_set_count, uint64_t first_packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control) {
  for (uint32_t i = 0; i < counter_set_count; ++i) {
    iree_hal_amdgpu_profile_counter_sample_slot_t* slot =
        iree_hal_amdgpu_host_queue_profile_counter_slot(queue, event_position,
                                                        i);
    iree_hal_amdgpu_host_queue_commit_profile_counter_packet(
        queue, &slot->packets.read_packet, first_packet_id + i * 2u,
        packet_control);
    iree_hal_amdgpu_host_queue_commit_profile_counter_packet(
        queue, &slot->packets.stop_packet, first_packet_id + i * 2u + 1u,
        packet_control);
  }
}

static iree_status_t iree_hal_amdgpu_profile_counter_sample_record_size(
    uint32_t sample_value_count, iree_host_size_t* out_record_size) {
  return IREE_STRUCT_LAYOUT(
      0, out_record_size,
      IREE_STRUCT_FIELD(1, iree_hal_profile_counter_sample_record_t, NULL),
      IREE_STRUCT_FIELD(sample_value_count, uint64_t, NULL));
}

static iree_status_t
iree_hal_amdgpu_host_queue_profile_counter_sample_storage_size(
    iree_hal_amdgpu_host_queue_t* queue, iree_host_size_t event_count,
    iree_host_size_t* out_storage_size) {
  iree_host_size_t per_event_storage_size = 0;
  for (uint32_t counter_set_ordinal = 0;
       counter_set_ordinal < queue->profiling.counter_set_count;
       ++counter_set_ordinal) {
    const iree_hal_amdgpu_profile_counter_set_t* counter_set =
        iree_hal_amdgpu_host_queue_profile_counter_set(queue,
                                                       counter_set_ordinal);
    if (IREE_UNLIKELY(!counter_set)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU counter set is not available");
    }
    iree_host_size_t record_size = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_counter_sample_record_size(
        counter_set->sample_value_count, &record_size));
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            per_event_storage_size, record_size, &per_event_storage_size))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU counter sample storage overflow");
    }
  }
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          per_event_storage_size, event_count, out_storage_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU counter sample storage overflow");
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_host_queue_profile_counter_max_counter_count(
    iree_hal_amdgpu_host_queue_t* queue, uint32_t* out_counter_count) {
  uint32_t max_counter_count = 0;
  for (uint32_t counter_set_ordinal = 0;
       counter_set_ordinal < queue->profiling.counter_set_count;
       ++counter_set_ordinal) {
    const iree_hal_amdgpu_profile_counter_set_t* counter_set =
        iree_hal_amdgpu_host_queue_profile_counter_set(queue,
                                                       counter_set_ordinal);
    if (IREE_UNLIKELY(!counter_set)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU counter set is not available");
    }
    max_counter_count = iree_max(max_counter_count, counter_set->counter_count);
  }
  *out_counter_count = max_counter_count;
  return iree_ok_status();
}

static void iree_hal_amdgpu_profile_counter_initialize_sample_record(
    const iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_profile_dispatch_event_t* event,
    const iree_hal_amdgpu_profile_counter_set_t* counter_set,
    const iree_hal_amdgpu_profile_counter_sample_slot_t* slot,
    iree_hal_profile_counter_sample_record_t* out_record) {
  *out_record = iree_hal_profile_counter_sample_record_default();
  out_record->flags = IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DISPATCH_EVENT;
  if (iree_any_bit_set(event->flags,
                       IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER)) {
    out_record->flags |= IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_COMMAND_OPERATION;
  }
  out_record->sample_id = slot->sample_id;
  out_record->counter_set_id = counter_set->counter_set_id;
  out_record->dispatch_event_id = event->event_id;
  out_record->submission_id = event->submission_id;
  out_record->command_buffer_id = event->command_buffer_id;
  out_record->executable_id = event->executable_id;
  out_record->stream_id = iree_hal_amdgpu_host_queue_profile_stream_id(queue);
  out_record->command_index = event->command_index;
  out_record->export_ordinal = event->export_ordinal;
  out_record->physical_device_ordinal =
      iree_hal_amdgpu_host_queue_profile_device_ordinal(queue);
  out_record->queue_ordinal =
      iree_hal_amdgpu_host_queue_profile_queue_ordinal(queue);
  out_record->sample_value_count = counter_set->sample_value_count;
}

static iree_status_t iree_hal_amdgpu_profile_counter_collect_sample_values(
    iree_hal_amdgpu_profile_counter_session_t* session,
    const iree_hal_amdgpu_profile_counter_set_t* counter_set,
    iree_hal_amdgpu_profile_counter_sample_slot_t* slot,
    uint32_t* counter_value_counts, uint64_t* values) {
  memset(counter_value_counts, 0,
         counter_set->counter_count * sizeof(counter_value_counts[0]));
  iree_hal_amdgpu_profile_counter_collect_context_t collect_context = {
      .counter_set = counter_set,
      .values = values,
      .counter_value_counts = counter_value_counts,
  };
  iree_status_t status = iree_status_from_aqlprofile_status(
      &session->libaqlprofile, __FILE__, __LINE__,
      session->libaqlprofile.aqlprofile_pmc_iterate_data(
          slot->handle, iree_hal_amdgpu_profile_counter_collect_callback,
          &collect_context),
      "aqlprofile_pmc_iterate_data", "iterating AMDGPU counter sample values");
  for (uint32_t i = 0;
       i < counter_set->counter_count && iree_status_is_ok(status); ++i) {
    const iree_hal_amdgpu_profile_counter_t* counter =
        &counter_set->counters[i];
    if (counter_value_counts[i] != counter->sample_value_count) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "aqlprofile returned %u values for AMDGPU counter '%.*s' but "
          "metadata declared %u",
          counter_value_counts[i], (int)counter->descriptor->name.size,
          counter->descriptor->name.data, counter->sample_value_count);
    }
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_pack_profile_counter_sample(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_counter_session_t* session, uint64_t event_position,
    const iree_hal_profile_dispatch_event_t* event,
    uint32_t counter_set_ordinal, uint32_t* counter_value_counts,
    uint8_t* storage, iree_host_size_t* out_record_size) {
  *out_record_size = 0;
  const iree_hal_amdgpu_profile_counter_set_t* counter_set =
      iree_hal_amdgpu_host_queue_profile_counter_set(queue,
                                                     counter_set_ordinal);
  if (IREE_UNLIKELY(!counter_set)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU counter set is not available");
  }
  iree_hal_amdgpu_profile_counter_sample_slot_t* slot =
      iree_hal_amdgpu_host_queue_profile_counter_slot(queue, event_position,
                                                      counter_set_ordinal);
  if (IREE_UNLIKELY(!slot->handle.handle || slot->sample_id == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU counter sample slot was not prepared before flush");
  }

  iree_hal_profile_counter_sample_record_t sample_record;
  iree_hal_amdgpu_profile_counter_initialize_sample_record(
      queue, event, counter_set, slot, &sample_record);
  iree_host_size_t record_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_counter_sample_record_size(
      sample_record.sample_value_count, &record_size));
  if (IREE_UNLIKELY(record_size > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU counter sample record exceeds uint32_t");
  }
  sample_record.record_length = (uint32_t)record_size;

  memcpy(storage, &sample_record, sizeof(sample_record));
  uint64_t* values = (uint64_t*)(void*)(storage + sizeof(sample_record));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_counter_collect_sample_values(
      session, counter_set, slot, counter_value_counts, values));
  *out_record_size = record_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_pack_profile_counter_samples(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_counter_session_t* session,
    uint64_t event_read_position, iree_host_size_t event_count,
    const iree_hal_profile_dispatch_event_t* events,
    uint32_t* counter_value_counts, uint8_t* storage) {
  uint8_t* sample_ptr = storage;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t event_ordinal = 0;
       event_ordinal < event_count && iree_status_is_ok(status);
       ++event_ordinal) {
    const uint64_t event_position = event_read_position + event_ordinal;
    const iree_hal_profile_dispatch_event_t* event = &events[event_ordinal];
    for (uint32_t counter_set_ordinal = 0;
         counter_set_ordinal < queue->profiling.counter_set_count &&
         iree_status_is_ok(status);
         ++counter_set_ordinal) {
      iree_host_size_t record_size = 0;
      status = iree_hal_amdgpu_host_queue_pack_profile_counter_sample(
          queue, session, event_position, event, counter_set_ordinal,
          counter_value_counts, sample_ptr, &record_size);
      if (iree_status_is_ok(status)) {
        sample_ptr += record_size;
      }
    }
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_write_profile_counter_chunk(
    const iree_hal_amdgpu_host_queue_t* queue, iree_hal_profile_sink_t* sink,
    uint64_t session_id, const uint8_t* sample_storage,
    iree_host_size_t sample_storage_size) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES;
  metadata.name = iree_make_cstring_view("amdgpu.counter-samples");
  metadata.session_id = session_id;
  metadata.stream_id = iree_hal_amdgpu_host_queue_profile_stream_id(queue);
  metadata.physical_device_ordinal =
      iree_hal_amdgpu_host_queue_profile_device_ordinal(queue);
  metadata.queue_ordinal =
      iree_hal_amdgpu_host_queue_profile_queue_ordinal(queue);
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span(sample_storage, sample_storage_size);
  return iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
}

iree_status_t iree_hal_amdgpu_host_queue_write_profile_counter_samples(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_profile_sink_t* sink,
    uint64_t session_id, uint64_t event_read_position,
    iree_host_size_t event_count,
    const iree_hal_profile_dispatch_event_t* events) {
  if (!sink || !event_count || !queue->profiling.counter_session) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_profile_counter_session_t* session =
      queue->profiling.counter_session;
  iree_host_size_t sample_storage_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_host_queue_profile_counter_sample_storage_size(
              queue, event_count, &sample_storage_size));

  uint32_t max_counter_count = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_host_queue_profile_counter_max_counter_count(
              queue, &max_counter_count));
  iree_host_size_t counter_value_counts_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(0, &counter_value_counts_size,
                         IREE_STRUCT_FIELD(max_counter_count, uint32_t, NULL)));
  uint32_t* counter_value_counts = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(queue->host_allocator, counter_value_counts_size,
                            (void**)&counter_value_counts));

  uint8_t* sample_storage = NULL;
  iree_status_t status = iree_allocator_malloc(
      queue->host_allocator, sample_storage_size, (void**)&sample_storage);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_pack_profile_counter_samples(
        queue, session, event_read_position, event_count, events,
        counter_value_counts, sample_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_write_profile_counter_chunk(
        queue, sink, session_id, sample_storage, sample_storage_size);
  }

  iree_allocator_free(queue->host_allocator, sample_storage);
  iree_allocator_free(queue->host_allocator, counter_value_counts);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
