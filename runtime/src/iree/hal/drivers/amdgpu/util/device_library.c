// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_library.h"

#include "iree/hal/drivers/amdgpu/device/binaries.h"
#include "iree/hal/drivers/amdgpu/device/kernels.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_library_t
//===----------------------------------------------------------------------===//

static iree_status_t iree_file_toc_append_names_to_builder(
    const iree_file_toc_t* file_toc, size_t file_count,
    iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0; i < file_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, file_toc[i].name));
  }
  return iree_ok_status();
}

typedef struct iree_hal_amdgpu_agent_available_isas_t {
  iree_host_size_t count;
  hsa_isa_t values[32];
} iree_hal_amdgpu_agent_available_isas_t;

static hsa_status_t iree_hal_amdgpu_iterate_agent_isa(hsa_isa_t isa,
                                                      void* user_data) {
  iree_hal_amdgpu_agent_available_isas_t* isas =
      (iree_hal_amdgpu_agent_available_isas_t*)user_data;
  if (isas->count >= IREE_ARRAYSIZE(isas->values)) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  isas->values[isas->count++] = isa;
  return HSA_STATUS_SUCCESS;
}

static iree_status_t iree_hal_amdgpu_agent_available_isas_append_to_builder(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_agent_available_isas_t* isas,
    iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0; i < isas->count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    uint32_t isa_name_length = 0;
    IREE_RETURN_IF_ERROR(
        iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isas->values[i],
                                  HSA_ISA_INFO_NAME_LENGTH, &isa_name_length));
    IREE_ASSERT_GT(isa_name_length, 0);  // always +1 in HSA
    char* isa_name = NULL;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(
        builder, isa_name_length - 1, &isa_name));
    IREE_RETURN_IF_ERROR(iree_hsa_isa_get_info_alt(
        IREE_LIBHSA(libhsa), isas->values[i], HSA_ISA_INFO_NAME, isa_name));
  }
  return iree_ok_status();
}

// Selects a device library binary file that supports the ISA of the provided
// |agent|.
static iree_status_t iree_hal_amdgpu_device_library_select_file(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    iree_allocator_t host_allocator, hsa_isa_t* out_isa,
    const iree_file_toc_t** out_file_toc) {
  IREE_ASSERT_ARGUMENT(out_isa);
  IREE_ASSERT_ARGUMENT(out_file_toc);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_isa = (hsa_isa_t){0};
  *out_file_toc = NULL;

  // Query all available ISAs supported by the agent.
  // This list is ordered by descending priority.
  iree_hal_amdgpu_agent_available_isas_t available_isas;
  memset(&available_isas, 0, sizeof(available_isas));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_agent_iterate_isas(IREE_LIBHSA(libhsa), agent,
                                      iree_hal_amdgpu_iterate_agent_isa,
                                      &available_isas));

  // For each ISA in decreasing priority try to find a binary that matches.
  // The binaries are named the same as HSA uses for the ISA name with the .so
  // suffix.
  hsa_isa_t best_isa = {0};
  const iree_file_toc_t* best_file_toc = NULL;
  for (iree_host_size_t i = 0; i < available_isas.count && !best_file_toc;
       ++i) {
    // Get the ISA name - it'll be something like `amdgcn-amd-amdhsa--gfx1100`
    // for some reason.
    hsa_isa_t isa = available_isas.values[i];
    uint32_t isa_name_length = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa,
                                  HSA_ISA_INFO_NAME_LENGTH, &isa_name_length));
    IREE_ASSERT_GT(isa_name_length, 0);  // always +1 in HSA
    char* isa_name_buffer = iree_alloca(isa_name_length);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa,
                                      HSA_ISA_INFO_NAME, isa_name_buffer));
    iree_string_view_t isa_name =
        iree_make_string_view(isa_name_buffer, isa_name_length - /*NUL*/ 1);
    for (iree_host_size_t j = 0; j < iree_hal_amdgpu_device_binaries_size();
         ++j) {
      const iree_file_toc_t* file_toc =
          &iree_hal_amdgpu_device_binaries_create()[j];
      if (iree_string_view_starts_with(IREE_SV(file_toc->name), isa_name)) {
        best_isa = isa;
        best_file_toc = file_toc;
        break;
      }
    }
  }

  // If we found a matching file return that for loading. It _should_ work but
  // is not guaranteed.
  iree_status_t status = iree_ok_status();
  if (best_file_toc) {
    *out_isa = best_isa;
    *out_file_toc = best_file_toc;
    IREE_TRACE_ZONE_APPEND_TEXT(z0, best_file_toc->name);
  } else {
    // Failures get nice errors with available/supported ISAs listed out.
    status = iree_make_status(IREE_STATUS_INCOMPATIBLE,
                              "no device library binary found that matches one "
                              "of the supported ISAs");
#if IREE_STATUS_MODE >= 2
    iree_string_builder_t builder;
    iree_string_builder_initialize(host_allocator, &builder);
    IREE_IGNORE_ERROR(iree_string_builder_append_string(
        &builder, IREE_SV("available in runtime build: [")));
    IREE_IGNORE_ERROR(iree_file_toc_append_names_to_builder(
        iree_hal_amdgpu_device_binaries_create(),
        iree_hal_amdgpu_device_binaries_size(), &builder));
    IREE_IGNORE_ERROR(iree_string_builder_append_string(
        &builder, IREE_SV("], supported by agent: [")));
    IREE_IGNORE_ERROR(iree_hal_amdgpu_agent_available_isas_append_to_builder(
        libhsa, &available_isas, &builder));
    IREE_IGNORE_ERROR(
        iree_string_builder_append_string(&builder, IREE_SV("]")));
    status = iree_status_annotate_f(status, "%.*s",
                                    (int)iree_string_builder_size(&builder),
                                    iree_string_builder_buffer(&builder));
    iree_string_builder_deinitialize(&builder);
#endif  // IREE_STATUS_MODE >= 2
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_device_library_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_amdgpu_device_library_t* out_library) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_library);

  if (IREE_UNLIKELY(topology->gpu_agent_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "topology must have at least one GPU agent");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_library, 0, sizeof(*out_library));
  out_library->libhsa = libhsa;

  // Select (or try to) the binary file for the leading GPU agent.
  // Today we require a single device ISA for all devices as heterogeneous
  // multi-device HAL usage is expected for different devices.
  hsa_isa_t isa = {0};
  const iree_file_toc_t* file_toc = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amdgpu_device_library_select_file(
          libhsa, topology->gpu_agents[0], host_allocator, &isa, &file_toc));

  // TODO(benvanik): figure out what options we could pass? Documentation is ...
  // lacking. These may have only been used for HSAIL anyway.
  const char* options = NULL;

  // Bind a code object reader to the memory sourced from our rodata.
  hsa_code_object_reader_t code_object_reader;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_code_object_reader_create_from_memory(
              IREE_LIBHSA(libhsa), file_toc->data, file_toc->size,
              &code_object_reader));

  // Create the executable that will hold all of the loaded code objects.
  // TODO(benvanik): pass profile/rounding mode from queried info.
  iree_status_t status =
      iree_hsa_executable_create_alt(IREE_LIBHSA(libhsa), HSA_PROFILE_FULL,
                                     HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                     options, &out_library->executable);

  // Load the code object for each agent.
  // Note that we could save off the loaded_code_object per-agent here but then
  // we'd need big fixed storage or dynamically allocated storage - instead we
  // take the hit of doing the n^2 resolve because it's only done once per
  // HAL device initialization. Everything that needs the information from the
  // loaded_code_objects caches the results.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
      status = iree_hsa_executable_load_agent_code_object(
          IREE_LIBHSA(libhsa), out_library->executable, topology->gpu_agents[i],
          code_object_reader, options, NULL);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // Freeze the executable now that loading has completed. Most queries require
  // that the executable be frozen.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_executable_freeze(IREE_LIBHSA(libhsa),
                                        out_library->executable, options);
  }

  // Release the reader now that the executable has been fully loaded.
  IREE_IGNORE_ERROR(iree_hsa_code_object_reader_destroy(IREE_LIBHSA(libhsa),
                                                        code_object_reader));

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_device_library_deinitialize(out_library);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_device_library_deinitialize(
    iree_hal_amdgpu_device_library_t* library) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (library->executable.handle) {
    IREE_IGNORE_ERROR(iree_hsa_executable_destroy(IREE_LIBHSA(library->libhsa),
                                                  library->executable));
  }

  memset(library, 0, sizeof(*library));

  IREE_TRACE_ZONE_END(z0);
}

typedef struct iree_hal_amdgpu_find_loaded_code_object_state_t {
  const iree_hal_amdgpu_libhsa_t* libhsa;
  hsa_agent_t agent;
  hsa_loaded_code_object_t loaded_code_object;
} iree_hal_amdgpu_find_loaded_code_object_state_t;
static hsa_status_t iree_hal_amdgpu_iterate_loaded_code_object(
    hsa_executable_t executable, hsa_loaded_code_object_t loaded_code_object,
    void* user_data) {
  iree_hal_amdgpu_find_loaded_code_object_state_t* find_state =
      (iree_hal_amdgpu_find_loaded_code_object_state_t*)user_data;
  hsa_agent_t agent = {0};
  hsa_status_t hsa_status =
      find_state->libhsa->amd_loader
          .hsa_ven_amd_loader_loaded_code_object_get_info(
              loaded_code_object,
              HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT, &agent);
  if (hsa_status != HSA_STATUS_SUCCESS) return hsa_status;
  if (agent.handle == find_state->agent.handle) {
    find_state->loaded_code_object = loaded_code_object;
    return HSA_STATUS_INFO_BREAK;  // found
  }
  return HSA_STATUS_SUCCESS;  // continue
}

// Finds the loaded code object in |executable| for the given |agent|.
static iree_status_t iree_hal_amdgpu_find_loaded_code_object(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    hsa_agent_t agent, hsa_loaded_code_object_t* out_loaded_code_object) {
  // Iterate over the code objects and find the first that matches the agent.
  iree_hal_amdgpu_find_loaded_code_object_state_t find_state = {
      .libhsa = libhsa,
      .agent = agent,
      .loaded_code_object = {0},
  };
  hsa_status_t hsa_status =
      libhsa->amd_loader
          .hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
              executable, iree_hal_amdgpu_iterate_loaded_code_object,
              &find_state);
  if (hsa_status == HSA_STATUS_SUCCESS) {
    // None found.
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no loaded code object found for the given agent");
  } else if (hsa_status != HSA_STATUS_INFO_BREAK) {
    // Error during iteration.
    return iree_status_from_hsa_status(
        __FILE__, __LINE__, hsa_status,
        "hsa_ven_amd_loader_executable_iterate_loaded_code_objects",
        "iterating loaded code objects");
  }
  *out_loaded_code_object = find_state.loaded_code_object;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_device_library_populate_agent_code_range(
    const iree_hal_amdgpu_device_library_t* library, hsa_agent_t device_agent,
    iree_hal_amdgpu_code_range_t* out_range) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(out_range);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_range, 0, sizeof(*out_range));

  const iree_hal_amdgpu_libhsa_t* libhsa = library->libhsa;

  // Lookup the loaded code object for the given device agent.
  // Each agent has its own copy and the virtual address ranges will differ for
  // each.
  hsa_loaded_code_object_t loaded_code_object = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_find_loaded_code_object(
              libhsa, library->executable, device_agent, &loaded_code_object));

  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;

  // Query the requested information.
  iree_hal_amdgpu_code_range_t range = {0};
  if (hsa_status == HSA_STATUS_SUCCESS) {
    hsa_status =
        libhsa->amd_loader.hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
            &range.device_ptr);
  }
  if (hsa_status == HSA_STATUS_SUCCESS) {
    hsa_status = libhsa->amd_loader.hsa_ven_amd_loader_query_host_address(
        (void*)range.device_ptr, (const void**)&range.host_ptr);
  }
  if (hsa_status == HSA_STATUS_SUCCESS) {
    hsa_status =
        libhsa->amd_loader.hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE, &range.size);
  }

  iree_status_t status = iree_ok_status();
  if (hsa_status == HSA_STATUS_SUCCESS) {
    *out_range = range;
  } else {
    status = iree_status_from_hsa_status(
        __FILE__, __LINE__, hsa_status,
        "hsa_ven_amd_loader_loaded_code_object_get_info",
        "querying code object info");
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_device_library_populate_kernel_args(
    const iree_hal_amdgpu_device_library_t* library, hsa_agent_t device_agent,
    const char* symbol_name, uint16_t workgroup_size_x,
    uint16_t workgroup_size_y, uint16_t workgroup_size_z,
    iree_hal_amdgpu_device_kernel_args_t* out_kernel_args) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, symbol_name);

  const iree_hal_amdgpu_libhsa_t* libhsa = library->libhsa;

  // Lookup the symbol. The `.kd` suffix is required and should have been passed
  // by the caller.
  hsa_executable_symbol_t symbol = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_executable_get_symbol_by_name(IREE_LIBHSA(libhsa),
                                             library->executable, symbol_name,
                                             &device_agent, &symbol),
      "resolving `%s`", symbol_name);

  // All of our kernels assume 3 dimensions.
  out_kernel_args->setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

  // TODO(benvanik): embed this as a custom section or attributes that we could
  // somehow query? For now we hardcode and take directly. This may be fine as
  // we aren't doing anything but blits and probably don't need to tightly
  // optimize workgroup size across architectures. Unfortunately the
  // `reqd_work_group_size` attribute is exactly what we want but clang only
  // allows it on OpenCL kernels (not C ones). Reading it is a PITA (need to
  // crack open the ELF, find the AMDGPU notes section, and decode the msgpack)
  // so unless we absolutely need it alternatives (like extracting from shared
  // headers as part of a build step) may be better.
  out_kernel_args->workgroup_size[0] = workgroup_size_x;
  out_kernel_args->workgroup_size[1] = workgroup_size_y;
  out_kernel_args->workgroup_size[2] = workgroup_size_z;

  // The object pointer is used in dispatch packets from either the host or
  // device.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_executable_symbol_get_info(
          IREE_LIBHSA(libhsa), symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
          &out_kernel_args->kernel_object));

  // Segment size information used to populate dispatch packets and reserve
  // space.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_executable_symbol_get_info(
              IREE_LIBHSA(libhsa), symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
              &out_kernel_args->private_segment_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_executable_symbol_get_info(
              IREE_LIBHSA(libhsa), symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
              &out_kernel_args->group_segment_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_executable_symbol_get_info(
              IREE_LIBHSA(libhsa), symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
              &out_kernel_args->kernarg_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_executable_symbol_get_info(
              IREE_LIBHSA(libhsa), symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
              &out_kernel_args->kernarg_alignment));

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  // TODO(benvanik): intern an export_loc? We don't have a Tracy API for this
  // yet and our option is to leak the value unconditionally.
  out_kernel_args->trace_src_loc = 0;
#else
  out_kernel_args->trace_src_loc = 0;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_device_library_populate_agent_kernels(
    const iree_hal_amdgpu_device_library_t* library, hsa_agent_t device_agent,
    iree_hal_amdgpu_device_kernels_t* out_kernels) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(out_kernels);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_kernels, 0, sizeof(*out_kernels));

#define IREE_HAL_AMDGPU_DEVICE_KERNEL(name, workgroup_size_x,             \
                                      workgroup_size_y, workgroup_size_z) \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                      \
      z0,                                                                 \
      iree_hal_amdgpu_device_library_populate_kernel_args(                \
          library, device_agent, #name ".kd", workgroup_size_x,           \
          workgroup_size_y, workgroup_size_z, &out_kernels->name),        \
      #name);
#include "iree/hal/drivers/amdgpu/device/kernel_tables.h"
#undef IREE_HAL_AMDGPU_DEVICE_KERNEL

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
