// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable.h"

#include "iree/base/internal/debugging.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/hal/utils/executable_debug_info.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/amdgpu_executable_def_reader.h"
#include "iree/schemas/amdgpu_executable_def_verifier.h"

// TODO(benvanik): replace with include when device-side tracing imported.
// #include "iree/hal/drivers/amdgpu/device/tracing.h"
typedef uint32_t iree_hal_amdgpu_trace_color_t;
typedef struct iree_hal_amdgpu_trace_src_loc_t {
  const char* name;
  const char* function;
  const char* file;
  uint32_t line;
  iree_hal_amdgpu_trace_color_t color;
} iree_hal_amdgpu_trace_src_loc_t;

//===----------------------------------------------------------------------===//
// ISA Support
//===----------------------------------------------------------------------===//

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

static iree_status_t iree_hal_amdgpu_verify_isas_equal(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_isa_t isa_a, hsa_isa_t isa_b) {
  uint32_t name_length_a = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_isa_get_info_alt(
      IREE_LIBHSA(libhsa), isa_a, HSA_ISA_INFO_NAME_LENGTH, &name_length_a));
  uint32_t name_length_b = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_isa_get_info_alt(
      IREE_LIBHSA(libhsa), isa_a, HSA_ISA_INFO_NAME_LENGTH, &name_length_b));
  char name_a[64 + /*NUL*/ 1] = {0};
  char name_b[64 + /*NUL*/ 1] = {0};
  if (name_length_a > IREE_ARRAYSIZE(name_a) ||
      name_length_b > IREE_ARRAYSIZE(name_b)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "ISA name too long");
  }
  IREE_RETURN_IF_ERROR(iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa_a,
                                                 HSA_ISA_INFO_NAME, &name_a));
  IREE_RETURN_IF_ERROR(iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa_a,
                                                 HSA_ISA_INFO_NAME, &name_b));
  if (name_length_a != name_length_b ||
      memcmp(name_a, name_b, name_length_a) != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "ISAs do not match: `%s` != `%s`", name_a, name_b);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_verify_device_isa_commonality(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology) {
  // If only one agent then no need to check.
  if (topology->gpu_agent_count == 1) return iree_ok_status();

  IREE_TRACE_ZONE_BEGIN(z0);

  // Query all available ISAs supported by the first GPU agent.
  // We'll use this to compare with all other GPU agents.
  iree_hal_amdgpu_agent_available_isas_t expected_isas;
  memset(&expected_isas, 0, sizeof(expected_isas));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_agent_iterate_isas(
              IREE_LIBHSA(libhsa), topology->gpu_agents[0],
              iree_hal_amdgpu_iterate_agent_isa, &expected_isas));

  // For all subsequent GPU agents ensure their ISAs match.
  for (iree_host_size_t i = 1; i < topology->gpu_agent_count; ++i) {
    // Get ISAs supported by this agent.
    iree_hal_amdgpu_agent_available_isas_t available_isas;
    memset(&available_isas, 0, sizeof(available_isas));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hsa_agent_iterate_isas(
                IREE_LIBHSA(libhsa), topology->gpu_agents[i],
                iree_hal_amdgpu_iterate_agent_isa, &available_isas));

    // Ensure ISAs match.
    // We could be less strict here and require only one matching ISA that we
    // share for all devices but in practice today all devices have a single
    // supported ISA and we expect devices to match exactly so this is just for
    // being thorough.
    if (available_isas.count != expected_isas.count) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                               "runtime currently expects all GPU agents must "
                               "support the same ISAs; gpu_agents[%" PRIhsz
                               "] does not match gpu_agents[0]",
                               i));
    }
    for (iree_host_size_t j = 0; j < expected_isas.count; ++j) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdgpu_verify_isas_equal(libhsa, expected_isas.values[j],
                                                available_isas.values[j]));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_executable_format_supported(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    iree_string_view_t format, bool* out_supported, hsa_isa_t* out_isa) {
  IREE_ASSERT_ARGUMENT(out_supported);
  *out_supported = false;
  if (out_isa) out_isa->handle = 0;

  // Strip hsa-* prefix.
  if (!iree_string_view_starts_with(
          format, iree_make_cstring_view("amdgcn-amd-amdhsa-"))) {
    // Not HSA-like.
    *out_supported = false;
    return iree_ok_status();
  }

  // Query all available ISAs supported by any GPU agent.
  // This list is ordered by descending priority.
  iree_hal_amdgpu_agent_available_isas_t available_isas;
  memset(&available_isas, 0, sizeof(available_isas));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_iterate_isas(
      IREE_LIBHSA(libhsa), device_agent, iree_hal_amdgpu_iterate_agent_isa,
      &available_isas));

  for (iree_host_size_t i = 0; i < available_isas.count; ++i) {
    // Get the ISA name - it'll be something like `amdgcn-amd-amdhsa--gfx1100`
    // for some reason. Note that the docs for HSA_ISA_INFO_NAME_LENGTH say it
    // doesn't include the NUL terminator but it definitely does - for our
    // proper string view usage we have to subtract one. HSA sets length+1 to
    // NUL so we must ensure we have sufficient space.
    hsa_isa_t isa = available_isas.values[i];
    char isa_name_buffer[64 + /*NUL*/ 1];
    uint32_t isa_name_length = 0;
    IREE_RETURN_IF_ERROR(iree_hsa_isa_get_info_alt(
        IREE_LIBHSA(libhsa), isa, HSA_ISA_INFO_NAME_LENGTH, &isa_name_length));
    if (isa_name_length == 0 ||
        isa_name_length > IREE_ARRAYSIZE(isa_name_buffer)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "ISA name invalid (empty or too long: %u)",
                              isa_name_length);
    }
    IREE_RETURN_IF_ERROR(iree_hsa_isa_get_info_alt(
        IREE_LIBHSA(libhsa), isa, HSA_ISA_INFO_NAME, isa_name_buffer));
    iree_string_view_t isa_name =
        iree_make_string_view(isa_name_buffer, isa_name_length - /*NUL*/ 1);

    // Compare exactly.
    if (iree_string_view_equal(format, isa_name)) {
      *out_supported = true;
      if (out_isa) *out_isa = isa;
      return iree_ok_status();
    }
  }

  // No compatible ISAs found.
  *out_supported = false;
  if (out_isa) out_isa->handle = 0;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Executable Verification
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_device_limits_t {
  // HSA_ISA_INFO_WORKGROUP_MAX_SIZE
  uint32_t max_workgroup_size;
  // HSA_ISA_INFO_WORKGROUP_MAX_DIM
  uint16_t max_workgroup_size_per_dim[3];
} iree_hal_amdgpu_device_limits_t;
static iree_status_t iree_hal_amdgpu_query_device_limits(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    hsa_isa_t isa, iree_hal_amdgpu_device_limits_t* out_limits) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_limits, 0, sizeof(*out_limits));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa,
                                    HSA_ISA_INFO_WORKGROUP_MAX_SIZE,
                                    &out_limits->max_workgroup_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa,
                                    HSA_ISA_INFO_WORKGROUP_MAX_DIM,
                                    &out_limits->max_workgroup_size_per_dim));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Verifies the structure of the flatbuffer.
//
// There are still some conditions we must be aware of (such as omitted names on
// functions with internal linkage), however we shouldn't need to bounds check
// anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_amdgpu_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data,
    const iree_hal_amdgpu_device_limits_t* limits) {
  if (!flatbuffer_data.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer data is not present");
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  const int verify_ret = iree_hal_amdgpu_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_amdgpu_ExecutableDef_table_t executable_def =
      iree_hal_amdgpu_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_amdgpu_ModuleDef_vec_t modules_vec =
      iree_hal_amdgpu_ExecutableDef_modules_get(executable_def);
  const iree_host_size_t module_count =
      iree_hal_amdgpu_ModuleDef_vec_len(modules_vec);
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_hal_amdgpu_ModuleDef_table_t module_def =
        iree_hal_amdgpu_ModuleDef_vec_at(modules_vec, i);
    if (!module_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "modules[%" PRIhsz "] is NULL", i);
    }
    if (flatbuffers_string_len(
            iree_hal_amdgpu_ModuleDef_image_get(module_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "modules[%" PRIhsz "] contents are empty", i);
    }
  }

  iree_hal_amdgpu_ExportDef_vec_t exports_vec =
      iree_hal_amdgpu_ExecutableDef_exports_get(executable_def);
  const iree_host_size_t export_count =
      iree_hal_amdgpu_ExportDef_vec_len(exports_vec);
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_amdgpu_ExportDef_table_t export_def =
        iree_hal_amdgpu_ExportDef_vec_at(exports_vec, i);
    if (!export_def) continue;

    if (flatbuffers_string_len(
            iree_hal_amdgpu_ExportDef_symbol_name_get(export_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz "] name is empty", i);
    }

    if (iree_hal_amdgpu_ExportDef_workgroup_size_is_present(export_def)) {
      const iree_hal_amdgpu_Dims_struct_t workgroup_size =
          iree_hal_amdgpu_ExportDef_workgroup_size_get(export_def);
      if (workgroup_size->x > limits->max_workgroup_size_per_dim[0] ||
          workgroup_size->y > limits->max_workgroup_size_per_dim[1] ||
          workgroup_size->z > limits->max_workgroup_size_per_dim[2]) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "exports[%" PRIhsz
            "] workgroup size dims %ux%ux%u exceeds device maximum %ux%ux%u",
            i, workgroup_size->x, workgroup_size->y, workgroup_size->z,
            limits->max_workgroup_size_per_dim[0],
            limits->max_workgroup_size_per_dim[1],
            limits->max_workgroup_size_per_dim[2]);
      }
      const uint32_t total_workgroup_size =
          workgroup_size->x * workgroup_size->y * workgroup_size->z;
      if (total_workgroup_size > limits->max_workgroup_size) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "exports[%" PRIhsz
            "] workgroup size total %u exceeds device maximum %u",
            i, total_workgroup_size, limits->max_workgroup_size);
      }
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%" PRIhsz "] workgroup size dims are missing", i);
    }

    const uint32_t constant_count =
        iree_hal_amdgpu_ExportDef_constant_count_get(export_def);
    if (constant_count > IREE_HAL_AMDGPU_MAX_DISPATCH_CONSTANT_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%" PRIhsz "] constant_count %u exceeds maximum of %u", i,
          constant_count, IREE_HAL_AMDGPU_MAX_DISPATCH_CONSTANT_COUNT);
    }

    iree_hal_amdgpu_BindingBits_vec_t binding_flags_vec =
        iree_hal_amdgpu_ExportDef_binding_flags_get(export_def);
    if (iree_hal_amdgpu_BindingBits_vec_len(binding_flags_vec) >
        IREE_HAL_AMDGPU_MAX_DISPATCH_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%" PRIhsz "] binding_flags count %zu exceeds maximum of %u",
          i, iree_hal_amdgpu_BindingBits_vec_len(binding_flags_vec),
          IREE_HAL_AMDGPU_MAX_DISPATCH_BINDING_COUNT);
    }

    IREE_RETURN_IF_ERROR(iree_hal_debug_verify_export_def(
        iree_hal_amdgpu_ExportDef_debug_info_get(export_def)));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Executable Loading
//===----------------------------------------------------------------------===//

// Loads an executable ELF from memory for all agents in |topology| and stores
// the frozen executable in |out_handle|.
static iree_status_t iree_hal_amdgpu_executable_load_modules(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_amdgpu_ModuleDef_vec_t module_defs, hsa_executable_t* out_handle) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_handle = (hsa_executable_t){0};

  // Today we require a single module.
  // We could support multiple and link them together by loading their code
  // objects in the order specified. This could be useful if we ever made our
  // own fat binaries or wanted to reuse shared ELFs across multiple executables
  // by having them reference the same ranges in a larger file.
  if (iree_hal_amdgpu_ModuleDef_vec_len(module_defs) != 1) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "only a single ModuleDef per ExecutableDef is "
                             "supported; executable declares %zu modules",
                             iree_hal_amdgpu_ModuleDef_vec_len(module_defs)));
  }
  flatbuffers_string_t image = iree_hal_amdgpu_ModuleDef_image_get(
      iree_hal_amdgpu_ModuleDef_vec_at(module_defs, 0));

  // TODO(#18877): support executable constants in HSA executables.
  // We currently don't support executable constants but we could by way of
  // global symbols. We should be using externs for the constants and then
  // hsa_executable_readonly_variable_define to specify each. We have to
  // allocate one copy of the constant table per agent. I don't know if it's
  // best to have one base symbol pointing to the table or one symbol per
  // constant in the table.
  if (executable_params->constant_count != 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "executable constants not yet implemented"));
  }

  // TODO(benvanik): figure out what options we could pass? Documentation is ...
  // lacking. These may have only been used for HSAIL anyway.
  const char* options = NULL;

  // Bind a code object reader to the memory sourced from our rodata.
  hsa_code_object_reader_t code_object_reader;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_code_object_reader_create_from_memory(
              IREE_LIBHSA(libhsa), image, flatbuffers_string_len(image),
              &code_object_reader));

  // Create the executable that will hold all of the loaded code objects.
  // TODO(benvanik): pass profile/rounding mode from queried info.
  hsa_executable_t handle = {0};
  iree_status_t status = iree_hsa_executable_create_alt(
      IREE_LIBHSA(libhsa), HSA_PROFILE_FULL,
      HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, options, &handle);

  // Load the code object for each agent.
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < topology->gpu_agent_count; ++i) {
    status = iree_hsa_executable_load_agent_code_object(
        IREE_LIBHSA(libhsa), handle, topology->gpu_agents[i],
        code_object_reader, options, NULL);
  }

  // Freeze the executable now that loading has completed. Most queries require
  // that the executable be frozen.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_executable_freeze(IREE_LIBHSA(libhsa), handle, options);
  }

  // Release the reader now that the executable has been fully loaded.
  IREE_IGNORE_ERROR(iree_hsa_code_object_reader_destroy(IREE_LIBHSA(libhsa),
                                                        code_object_reader));

  if (iree_status_is_ok(status)) {
    *out_handle = handle;
  } else if (handle.handle) {
    IREE_IGNORE_ERROR(iree_hsa_executable_destroy(IREE_LIBHSA(libhsa), handle));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Resolves the uniform kernel arguments that are the same on all GPU device
// agents in the topology (since we assume all are the same device type).
// All fields besides `kernel_object` will have valid values.
static iree_status_t iree_hal_amdgpu_executable_resolve_kernel_args(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    iree_hal_amdgpu_ExportDef_table_t export_def,
    const iree_hal_amdgpu_trace_src_loc_t* export_loc,
    hsa_agent_t any_device_agent,
    iree_hal_amdgpu_device_kernel_args_t* out_kernel_args) {
  IREE_ASSERT_ARGUMENT(out_kernel_args);
  IREE_TRACE_ZONE_BEGIN(z0);

  const char* symbol_name =
      iree_hal_amdgpu_ExportDef_symbol_name_get(export_def);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, symbol_name);

  // Lookup the symbol on any device. All devices today must be the same so the
  // parameters will match (except the kernel_object pointer).
  //
  // NOTE: must include `.kd` suffix.
  hsa_executable_symbol_t symbol = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_executable_get_symbol_by_name(IREE_LIBHSA(libhsa),
                                                 executable, symbol_name,
                                                 &any_device_agent, &symbol));

  // All of our kernels assume 3 dimensions.
  out_kernel_args->setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

  // TODO(benvanik): embed this as a custom section or attributes that we could
  // somehow query? For now we need the flatbuffer.
  const iree_hal_amdgpu_Dims_struct_t workgroup_size =
      iree_hal_amdgpu_ExportDef_workgroup_size_get(export_def);
  out_kernel_args->workgroup_size[0] = workgroup_size->x;
  out_kernel_args->workgroup_size[1] = workgroup_size->y;
  out_kernel_args->workgroup_size[2] = workgroup_size->z;

  // NOTE: the object pointer is per-device and we populate that when uploading
  // device tables.
  out_kernel_args->kernel_object = 0;

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

  iree_hal_amdgpu_BindingBits_vec_t binding_bits =
      iree_hal_amdgpu_ExportDef_binding_flags_get(export_def);
  out_kernel_args->binding_count =
      (uint16_t)iree_hal_amdgpu_BindingBits_vec_len(binding_bits);
  out_kernel_args->constant_count =
      (uint16_t)iree_hal_amdgpu_ExportDef_constant_count_get(export_def);

  // Interned debugging info for the lifetime of the process. This is required
  // so tracing tools can access the values while flushing when the process
  // exits. If no debugging info was available or it's not enabled in the build
  // this will be 0/NULL.
  out_kernel_args->trace_src_loc = (uint64_t)export_loc;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Allocates (and leaks) a table of source locations for each of |export_defs|.
// The returned table matches 1:1 and will persist for the lifetime of the
// process.
static iree_status_t iree_hal_amdgpu_executable_intern_trace_locs(
    iree_hal_amdgpu_ExportDef_vec_t export_defs,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_trace_src_loc_t** out_export_locs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t export_count =
      iree_hal_amdgpu_ExportDef_vec_len(export_defs);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, export_count);

  // Sum up the total storage required for all information.
  iree_host_size_t total_size =
      export_count * sizeof(iree_hal_amdgpu_trace_src_loc_t);
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_amdgpu_ExportDef_table_t export_def =
        iree_hal_amdgpu_ExportDef_vec_at(export_defs, i);
    iree_hal_debug_ExportDef_table_t debug_def =
        iree_hal_amdgpu_ExportDef_debug_info_get(export_def);
    total_size +=
        flatbuffers_string_len(iree_hal_debug_ExportDef_name_get(debug_def)) +
        1;
    iree_hal_debug_FileLineLocDef_table_t loc_def =
        iree_hal_debug_ExportDef_location_get(debug_def);
    if (loc_def) {
      total_size += flatbuffers_string_len(
                        iree_hal_debug_FileLineLocDef_filename_get(loc_def)) +
                    1;
    }
  }

  // Allocate persistent storage.
  iree_hal_amdgpu_trace_src_loc_t* export_locs = NULL;
  IREE_LEAK_CHECK_DISABLE_PUSH();
  export_locs = (iree_hal_amdgpu_trace_src_loc_t*)malloc(total_size);
  IREE_LEAK_CHECK_DISABLE_POP();
  char* char_buffer = (char*)&export_locs[export_count];

  // Populate table and fill the buffer. The only pointers used are those
  // pointing into the persistent allocation.
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_amdgpu_trace_src_loc_t* export_loc = &export_locs[i];
    export_loc->name = NULL;  // not needed

    iree_hal_amdgpu_ExportDef_table_t export_def =
        iree_hal_amdgpu_ExportDef_vec_at(export_defs, i);
    iree_hal_debug_ExportDef_table_t debug_def =
        iree_hal_amdgpu_ExportDef_debug_info_get(export_def);

    flatbuffers_string_t function =
        iree_hal_debug_ExportDef_name_get(debug_def);
    iree_host_size_t function_len = flatbuffers_string_len(function);
    memcpy(char_buffer, function, function_len);
    export_loc->function = char_buffer;
    char_buffer += function_len + 1;

    iree_hal_debug_FileLineLocDef_table_t loc_def =
        iree_hal_debug_ExportDef_location_get(debug_def);
    if (loc_def) {
      flatbuffers_string_t file =
          iree_hal_debug_FileLineLocDef_filename_get(loc_def);
      iree_host_size_t file_len = flatbuffers_string_len(file);
      memcpy(char_buffer, file, file_len);
      export_loc->file = char_buffer;
      char_buffer += file_len + 1;
      export_loc->line = iree_hal_debug_FileLineLocDef_line_get(loc_def);
    }

    // We could do something clever here to ensure consistent colors, like
    // hashing based on name.
    export_loc->color = 0;
  }

  *out_export_locs = export_locs;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Uploads the provided kernel table to |device_agent| and returns the pointer.
// |host_kernel_args| will have its `kernel_object` fields mutated during the
// upload.
static iree_status_t iree_hal_amdgpu_executable_upload_kernel_table(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    iree_hal_amdgpu_ExportDef_vec_t export_defs, iree_host_size_t kernel_count,
    iree_hal_amdgpu_device_kernel_args_t* host_kernel_args,
    hsa_agent_t device_agent,
    IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t**
        out_device_kernel_args) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device_kernel_args = NULL;

  // Upload copies of kernel arguments for each device.
  // We reuse the host storage we already allocated to make it possible to
  // memcpy the entire table in one go from host memory.
  // Resolve all kernel object pointers for the device agent.
  for (iree_host_size_t kernel_ordinal = 0; kernel_ordinal < kernel_count;
       ++kernel_ordinal) {
    iree_hal_amdgpu_ExportDef_table_t export_def =
        iree_hal_amdgpu_ExportDef_vec_at(export_defs, kernel_ordinal);
    const char* symbol_name =
        iree_hal_amdgpu_ExportDef_symbol_name_get(export_def);

    // NOTE: must include `.kd` suffix.
    hsa_executable_symbol_t symbol = {0};
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hsa_executable_get_symbol_by_name(
                                          IREE_LIBHSA(libhsa), executable,
                                          symbol_name, &device_agent, &symbol),
                                      "resolving `%s`", symbol_name);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hsa_executable_symbol_get_info(
                IREE_LIBHSA(libhsa), symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                &host_kernel_args[kernel_ordinal].kernel_object));
  }

  // Find a memory pool on the agent where we can upload the table.
  hsa_amd_memory_pool_t memory_pool = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amdgpu_find_coarse_global_memory_pool(libhsa, device_agent,
                                                     &memory_pool),
      "finding memory pool for storing kernel arg tables");

  // Allocate device kernel argument storage.
  const iree_host_size_t kernel_args_table_size =
      kernel_count * sizeof(host_kernel_args[0]);
  iree_hal_amdgpu_device_kernel_args_t* device_kernel_args = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_memory_pool_allocate(
              IREE_LIBHSA(libhsa), memory_pool, kernel_args_table_size,
              HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&device_kernel_args));

  // Copy the entire table to device memory.
  iree_status_t status =
      iree_hsa_memory_copy(IREE_LIBHSA(libhsa), device_kernel_args,
                           host_kernel_args, kernel_args_table_size);

  if (iree_status_is_ok(status)) {
    *out_device_kernel_args = device_kernel_args;
  } else {
    IREE_IGNORE_ERROR(
        iree_hsa_amd_memory_pool_free(IREE_LIBHSA(libhsa), device_kernel_args));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Unowned HSA API handle. Must remain valid for the lifetime of the
  // executable.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Loaded HSA executable with a code object for each device.
  hsa_executable_t handle;

  // Total number of exports in the executable.
  iree_host_size_t kernel_count;
  // Table of kernel args stored in host memory. We have them local so that
  // host-side command buffer recording doesn't need to access device memory.
  // The kernel object specified in each is invalid as it's agent-specific.
  iree_hal_amdgpu_device_kernel_args_t* host_kernel_args /*[kernel_count]*/;

  // Total number of GPU devices in the system that the executable kernel arg
  // table has been uploaded to.
  iree_host_size_t device_count;
  // Table of kernel args stored in device memory, one copy per device.
  // Each device has an entire `kernel_count` set of args.
  IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t*
      device_kernel_args[/*device_count*/];
} iree_hal_amdgpu_executable_t;

static const iree_hal_executable_vtable_t iree_hal_amdgpu_executable_vtable;

static iree_hal_amdgpu_executable_t* iree_hal_amdgpu_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_executable_vtable);
  return (iree_hal_amdgpu_executable_t*)base_value;
}

static const iree_hal_amdgpu_executable_t*
iree_hal_amdgpu_executable_const_cast(const iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_executable_vtable);
  return (const iree_hal_amdgpu_executable_t*)base_value;
}

iree_status_t iree_hal_amdgpu_executable_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): use executable_params->queue_affinity instead of the raw
  // topology - the affinity will tell us exactly which physical devices we need
  // to load the executable on. We have to map from queue affinity to GPU agent
  // and don't have a utility for that accessible here yet.

  // Pick a device to be our template for device queries. All devices in the
  // topology are expected to be the same. This should have been checked
  // earlier but we do it here in case the user is bypassing that code.
  IREE_ASSERT_GT(topology->gpu_agent_count, 1);
  if (IREE_UNLIKELY(topology->gpu_agent_count == 0)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "topology must have at least one GPU device"));
  }
  hsa_agent_t any_device_agent = topology->gpu_agents[0];

  // Check that the executable is supported and get the ISA it matches.
  bool supported = false;
  hsa_isa_t isa = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_executable_format_supported(
              libhsa, any_device_agent, executable_params->executable_format,
              &supported, &isa));
  if (!supported) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INCOMPATIBLE,
                             "executable format `%.*s` not supported by the "
                             "devices in the topology",
                             (int)executable_params->executable_format.size,
                             executable_params->executable_format.data));
  }

  // Verify the flatbuffer is valid.
  // Doing this first ensures we don't need to check the structure of the
  // flatbuffer during loading (though things like optional fields still need to
  // be checked!).
  iree_hal_amdgpu_device_limits_t limits = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_query_device_limits(libhsa, any_device_agent, isa,
                                              &limits));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_executable_flatbuffer_verify(
              executable_params->executable_data, &limits));

  // Dereference the flatbuffer.
  iree_hal_amdgpu_ExecutableDef_table_t executable_def =
      iree_hal_amdgpu_ExecutableDef_as_root(
          executable_params->executable_data.data);
  iree_hal_amdgpu_ExportDef_vec_t export_defs =
      iree_hal_amdgpu_ExecutableDef_exports_get(executable_def);
  const iree_host_size_t export_count =
      iree_hal_amdgpu_ExportDef_vec_len(export_defs);

  // Allocate storage for the executable and its associated data structures.
  iree_hal_amdgpu_executable_t* executable = NULL;
  const iree_host_size_t total_size =
      sizeof(*executable) +
      export_count * sizeof(executable->host_kernel_args[0]) +
      topology->gpu_agent_count * sizeof(executable->device_kernel_args[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  iree_hal_resource_initialize(&iree_hal_amdgpu_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->libhsa = libhsa;
  executable->kernel_count = export_count;
  executable->host_kernel_args =
      (iree_hal_amdgpu_device_kernel_args_t*)(((uint8_t*)executable) +
                                              sizeof(*executable));
  executable->device_count = topology->gpu_agent_count;

  // Publish any embedded source files to the tracing infrastructure.
  iree_hal_debug_publish_source_files(
      iree_hal_amdgpu_ExecutableDef_source_files_get(executable_def));

  iree_status_t status = iree_ok_status();

  // Intern source locations for all exported functions. These will persist for
  // the lifetime of the process and be passed to tooling as if they were in a
  // rodata segment.
  iree_hal_amdgpu_trace_src_loc_t* export_locs = NULL;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_intern_trace_locs(
        export_defs, host_allocator, &export_locs);
  }
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

  // Load executable and register it with all GPU agents.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_load_modules(
        libhsa, topology, executable_params,
        iree_hal_amdgpu_ExecutableDef_modules_get(executable_def),
        &executable->handle);
  }

  // Resolve kernel args for each export.
  // These parameters should be the same for all devices as we require all
  // devices have the same ISA. The only thing that will differ is the
  // kernel_object pointer and we handle that per-device during table upload.
  for (iree_host_size_t kernel_ordinal = 0;
       iree_status_is_ok(status) && kernel_ordinal < executable->kernel_count;
       ++kernel_ordinal) {
    iree_hal_amdgpu_ExportDef_table_t export_def =
        iree_hal_amdgpu_ExportDef_vec_at(export_defs, kernel_ordinal);
    const iree_hal_amdgpu_trace_src_loc_t* export_loc =
        export_locs ? &export_locs[kernel_ordinal] : NULL;
    status = iree_status_annotate_f(
        iree_hal_amdgpu_executable_resolve_kernel_args(
            libhsa, executable->handle, export_def, export_loc,
            any_device_agent, &executable->host_kernel_args[kernel_ordinal]),
        "resolving kernel args for `%s`",
        iree_hal_amdgpu_ExportDef_symbol_name_get(export_def));
  }

  // Upload copies of kernel arguments for each device.
  // We reuse the host storage we already allocated to make it possible to
  // memcpy the entire table in one go from host memory.
  for (iree_host_size_t device_ordinal = 0;
       iree_status_is_ok(status) && device_ordinal < executable->device_count;
       ++device_ordinal) {
    status = iree_hal_amdgpu_executable_upload_kernel_table(
        libhsa, executable->handle, export_defs, executable->kernel_count,
        executable->host_kernel_args, topology->gpu_agents[device_ordinal],
        &executable->device_kernel_args[device_ordinal]);
  }

  // Invalidate the kernel object pointer in all host args so that we don't
  // accidentally use it instead of the device-specific one.
  for (iree_host_size_t kernel_ordinal = 0;
       kernel_ordinal < executable->kernel_count; ++kernel_ordinal) {
    executable->host_kernel_args[kernel_ordinal].kernel_object = 0;
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t device_ordinal = 0;
       device_ordinal < executable->device_count; ++device_ordinal) {
    void* kernel_args = (void*)executable->device_kernel_args[device_ordinal];
    if (kernel_args) {
      IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(
          IREE_LIBHSA(executable->libhsa), kernel_args));
    }
  }

  if (executable->handle.handle) {
    IREE_IGNORE_ERROR(iree_hsa_executable_destroy(
        IREE_LIBHSA(executable->libhsa), executable->handle));
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_host(
    iree_hal_executable_t* base_executable, iree_host_size_t entry_point,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args) {
  const iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_const_cast(base_executable);
  *out_kernel_args = NULL;

  if (IREE_UNLIKELY(entry_point >= executable->kernel_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIhsz
                            " out of range; executable has %" PRIhsz " exports",
                            entry_point, executable->kernel_count);
  }

  *out_kernel_args = &executable->host_kernel_args[entry_point];

  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_device(
    iree_hal_executable_t* base_executable, iree_host_size_t entry_point,
    iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args) {
  const iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_const_cast(base_executable);
  *out_kernel_args = NULL;

  if (IREE_UNLIKELY(entry_point >= executable->kernel_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIhsz
                            " out of range; executable has %" PRIhsz " exports",
                            entry_point, executable->kernel_count);
  } else if (IREE_UNLIKELY(device_ordinal >= executable->device_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "device ordinal %" PRIhsz
                            " out of range; executable was loaded on %" PRIhsz
                            " devices",
                            device_ordinal, executable->device_count);
  }

  *out_kernel_args =
      &executable->device_kernel_args[device_ordinal][entry_point];

  return iree_ok_status();
}

static const iree_hal_executable_vtable_t iree_hal_amdgpu_executable_vtable = {
    .destroy = iree_hal_amdgpu_executable_destroy,
};
