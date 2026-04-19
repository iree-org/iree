// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable.h"

#include "iree/base/internal/debugging.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"
#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/hal/utils/elf_format.h"
#include "iree/hal/utils/executable_debug_info.h"
#include "iree/hal/utils/executable_header.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/amdgpu_executable_def_reader.h"
#include "iree/schemas/amdgpu_executable_def_verifier.h"

// TODO(benvanik): replace with include when device-side tracing imported.
// #include "iree/hal/drivers/amdgpu/device/tracing.h"
typedef uint32_t iree_hal_amdgpu_trace_color_t;
typedef struct iree_hal_amdgpu_trace_src_loc_t {
  // Trace label shown by device-side instrumentation.
  const char* name;
  // Source function name for the trace location.
  const char* function;
  // Source file path for the trace location.
  const char* file;
  // One-based source line for the trace location.
  uint32_t line;
  // Packed trace color used by the device-side tracing ABI.
  iree_hal_amdgpu_trace_color_t color;
} iree_hal_amdgpu_trace_src_loc_t;

//===----------------------------------------------------------------------===//
// ISA Support
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_agent_available_isas_t {
  // Number of valid entries in |values|.
  iree_host_size_t count;
  // Fixed-capacity ISA list populated by HSA iteration callbacks.
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
      IREE_LIBHSA(libhsa), isa_b, HSA_ISA_INFO_NAME_LENGTH, &name_length_b));
  char name_a[64 + /*NUL*/ 1] = {0};
  char name_b[64 + /*NUL*/ 1] = {0};
  if (name_length_a > IREE_ARRAYSIZE(name_a) ||
      name_length_b > IREE_ARRAYSIZE(name_b)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "ISA name too long");
  }
  IREE_RETURN_IF_ERROR(iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa_a,
                                                 HSA_ISA_INFO_NAME, &name_a));
  IREE_RETURN_IF_ERROR(iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), isa_b,
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

  const iree_string_view_t hsa_triple_prefix =
      iree_make_cstring_view("amdgcn-amd-amdhsa-");
  const iree_string_view_t hsa_short_arch_prefix =
      iree_make_cstring_view("amdgcn-amd-amdhsa--");
  const bool format_is_hsa_isa =
      iree_string_view_starts_with(format, hsa_triple_prefix);

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

    // Compare exact HSA ISA names or compiler target-architecture names.
    iree_string_view_t comparable_isa_name = isa_name;
    if (!format_is_hsa_isa &&
        iree_string_view_starts_with(isa_name, hsa_short_arch_prefix)) {
      comparable_isa_name = iree_string_view_substr(
          isa_name, hsa_short_arch_prefix.size, IREE_STRING_VIEW_NPOS);
    }
    if (iree_string_view_equal(format, comparable_isa_name)) {
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
  // Maximum total workgroup size from HSA_ISA_INFO_WORKGROUP_MAX_SIZE.
  uint32_t max_workgroup_size;
  // Maximum workgroup size per dimension from HSA_ISA_INFO_WORKGROUP_MAX_DIM.
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

static bool iree_hal_amdgpu_executable_data_is_wrapped_flatbuffer(
    iree_const_byte_span_t executable_data) {
  if (executable_data.data_length != 0 &&
      executable_data.data_length < sizeof(uint32_t)) {
    return false;
  }
  iree_const_byte_span_t identifier_data =
      iree_make_const_byte_span(executable_data.data, sizeof(uint32_t));
  if (iree_const_byte_span_is_empty(identifier_data)) {
    return false;
  }
  return memcmp(identifier_data.data,
                iree_hal_amdgpu_ExecutableDef_file_identifier,
                identifier_data.data_length) == 0;
}

static iree_status_t iree_hal_amdgpu_executable_normalize_isa_format(
    iree_string_view_t isa_name, iree_host_size_t executable_format_capacity,
    char* executable_format) {
  if (iree_string_view_is_empty(isa_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable missing target ISA");
  }

  const iree_string_view_t hsa_triple_prefix =
      iree_make_cstring_view("amdgcn-amd-amdhsa-");
  const iree_string_view_t hsa_short_arch_prefix =
      iree_make_cstring_view("amdgcn-amd-amdhsa--");
  if (!iree_string_view_starts_with(isa_name, hsa_triple_prefix)) {
    // Some compiler-produced AMDGPU executable flatbuffers use the target
    // architecture by itself (`gfx1100`) instead of the canonical HSA ISA name.
    // The HAL executable format is the canonical HSA name, so normalize it here
    // instead of leaking the short compiler spelling to runtime callers.
    const iree_host_size_t executable_format_length =
        hsa_short_arch_prefix.size + isa_name.size;
    if (executable_format_length >= executable_format_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "executable format buffer too small");
    }
    memcpy(executable_format, hsa_short_arch_prefix.data,
           hsa_short_arch_prefix.size);
    memcpy(executable_format + hsa_short_arch_prefix.size, isa_name.data,
           isa_name.size);
    executable_format[executable_format_length] = 0;
  } else if (isa_name.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  } else {
    memcpy(executable_format, isa_name.data, isa_name.size);
    executable_format[isa_name.size] = 0;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_allocator_t host_allocator, iree_host_size_t* out_inferred_size) {
  const bool is_wrapped_flatbuffer =
      iree_hal_amdgpu_executable_data_is_wrapped_flatbuffer(executable_data);

  // Read the header prefix (with unsafe inference if size is unknown).
  const bool unsafe_infer_size = (executable_data.data_length == 0);
  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  if (is_wrapped_flatbuffer) {
    IREE_RETURN_IF_ERROR(iree_hal_read_executable_flatbuffer_header(
        executable_data, unsafe_infer_size,
        iree_hal_amdgpu_ExecutableDef_file_identifier, &flatbuffer_data));

    // Verify the flatbuffer structure.
    const int verify_ret = iree_hal_amdgpu_ExecutableDef_verify_as_root(
        flatbuffer_data.data, flatbuffer_data.data_length);
    if (verify_ret != flatcc_verify_ok) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "flatbuffer verification failed: %s",
                              flatcc_verify_error_string(verify_ret));
    }

    // Get the ISA name from the flatbuffer.
    iree_hal_amdgpu_ExecutableDef_table_t executable_def =
        iree_hal_amdgpu_ExecutableDef_as_root(flatbuffer_data.data);
    flatbuffers_string_t isa =
        iree_hal_amdgpu_ExecutableDef_isa_get(executable_def);
    if (!isa) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable missing target ISA");
    }

    // Write the format string (ISA name).
    iree_string_view_t isa_name =
        iree_make_string_view(isa, flatbuffers_string_len(isa));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_normalize_isa_format(
        isa_name, executable_format_capacity, executable_format));

    // Return the total size (header + flatbuffer).
    *out_inferred_size =
        sizeof(iree_flatbuffer_file_header_t) + flatbuffer_data.data_length;
    return iree_ok_status();
  } else {
    iree_const_byte_span_t hsaco_data = executable_data;
    if (unsafe_infer_size) {
      iree_host_size_t hsaco_size = 0;
      IREE_RETURN_IF_ERROR(iree_hal_elf_calculate_size(hsaco_data, &hsaco_size),
                           "calculating raw HSACO ELF size");
      hsaco_data = iree_make_const_byte_span(executable_data.data, hsaco_size);
    }

    iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
        hsaco_data, host_allocator, &hsaco_metadata));
    iree_status_t status = iree_hal_amdgpu_executable_normalize_isa_format(
        hsaco_metadata.target, executable_format_capacity, executable_format);
    iree_hal_amdgpu_hsaco_metadata_deinitialize(&hsaco_metadata);
    IREE_RETURN_IF_ERROR(status);

    *out_inferred_size = hsaco_data.data_length;
    return iree_ok_status();
  }
}

// Verifies the structure of the flatbuffer.
//
// There are still some conditions we must be aware of (such as omitted names on
// functions with internal linkage), however we shouldn't need to bounds check
// anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_amdgpu_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data,
    const iree_hal_amdgpu_device_limits_t* limits) {
  IREE_ASSERT(flatbuffer_data.data && flatbuffer_data.data_length >= 16);

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
      const uint64_t total_workgroup_size =
          (uint64_t)workgroup_size->x * workgroup_size->y * workgroup_size->z;
      if (total_workgroup_size > limits->max_workgroup_size) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "exports[%" PRIhsz "] workgroup size total %" PRIu64
            " exceeds device maximum %u",
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

static iree_status_t iree_hal_amdgpu_executable_get_single_module_image(
    iree_hal_amdgpu_ModuleDef_vec_t module_defs,
    iree_const_byte_span_t* out_code_object_data) {
  *out_code_object_data = iree_const_byte_span_empty();

  // Today we require a single module. We could support multiple and link them
  // together by loading code objects in the order specified. This could be
  // useful if we ever made our own fat binaries or wanted to reuse shared ELFs
  // across multiple executables by having them reference the same ranges in a
  // larger file.
  const iree_host_size_t module_count =
      iree_hal_amdgpu_ModuleDef_vec_len(module_defs);
  if (module_count != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only a single ModuleDef per ExecutableDef is "
                            "supported; executable declares %" PRIhsz
                            " modules",
                            module_count);
  }
  iree_hal_amdgpu_ModuleDef_table_t module_def =
      iree_hal_amdgpu_ModuleDef_vec_at(module_defs, 0);
  if (!module_def) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "module is NULL");
  }
  flatbuffers_string_t image = iree_hal_amdgpu_ModuleDef_image_get(module_def);
  if (!image) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module image is empty");
  }
  const iree_host_size_t image_size = flatbuffers_string_len(image);
  if (image_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module image is empty");
  }
  *out_code_object_data =
      iree_make_const_byte_span((const uint8_t*)image, image_size);
  return iree_ok_status();
}

static bool iree_hal_amdgpu_physical_device_mask_contains(
    uint64_t physical_device_mask, iree_host_size_t physical_device_ordinal) {
  return physical_device_ordinal < IREE_HAL_MAX_QUEUES &&
         iree_all_bits_set(physical_device_mask,
                           ((uint64_t)1) << physical_device_ordinal);
}

static iree_status_t iree_hal_amdgpu_executable_select_physical_devices(
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        out_physical_devices) {
  memset(out_physical_devices, 0, sizeof(*out_physical_devices));

  iree_hal_amdgpu_queue_affinity_domain_t queue_affinity_domain = {
      .supported_affinity = 0,
      .physical_device_count = topology->gpu_agent_count,
      .queue_count_per_physical_device = topology->gpu_agent_queue_count,
  };

  for (iree_host_size_t physical_device_ordinal = 0;
       physical_device_ordinal < topology->gpu_agent_count;
       ++physical_device_ordinal) {
    iree_hal_queue_affinity_t physical_device_affinity = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_for_physical_device(
        queue_affinity_domain, physical_device_ordinal,
        &physical_device_affinity));
    iree_hal_queue_affinity_or_into(queue_affinity_domain.supported_affinity,
                                    physical_device_affinity);
  }

  return iree_hal_amdgpu_queue_affinity_select_physical_devices(
      queue_affinity_domain, requested_affinity, out_physical_devices);
}

// Loads an executable ELF from memory for selected agents in |topology| and
// stores the frozen executable in |out_handle|.
static iree_status_t iree_hal_amdgpu_executable_load_module(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        physical_devices,
    const iree_hal_executable_params_t* executable_params,
    iree_const_byte_span_t code_object_data, hsa_executable_t* out_handle) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_handle = (hsa_executable_t){0};

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
              IREE_LIBHSA(libhsa), (const char*)code_object_data.data,
              code_object_data.data_length, &code_object_reader));

  // Create the executable that will hold all of the loaded code objects.
  // TODO(benvanik): pass profile/rounding mode from queried info.
  hsa_executable_t handle = {0};
  iree_status_t status = iree_hsa_executable_create_alt(
      IREE_LIBHSA(libhsa), HSA_PROFILE_FULL,
      HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, options, &handle);

  // Load the code object for each selected agent.
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < topology->gpu_agent_count; ++i) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            physical_devices->physical_device_mask, i)) {
      continue;
    }
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
  status =
      iree_status_join(status, iree_hsa_code_object_reader_destroy(
                                   IREE_LIBHSA(libhsa), code_object_reader));

  if (iree_status_is_ok(status)) {
    *out_handle = handle;
  } else if (handle.handle) {
    status = iree_status_join(
        status, iree_hsa_executable_destroy(IREE_LIBHSA(libhsa), handle));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_executable_add_trace_string_storage(
    flatbuffers_string_t value, iree_host_size_t* inout_total_size) {
  if (!value) return iree_ok_status();
  const iree_host_size_t value_length = flatbuffers_string_len(value);
  if (value_length == 0) return iree_ok_status();

  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_add(value_length, 1, &storage_size) ||
      !iree_host_size_checked_add(*inout_total_size, storage_size,
                                  inout_total_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU executable trace source location storage size overflow");
  }
  return iree_ok_status();
}

static const char* iree_hal_amdgpu_executable_copy_trace_string(
    flatbuffers_string_t value, char** inout_char_buffer) {
  if (!value) return NULL;
  const iree_host_size_t value_length = flatbuffers_string_len(value);
  if (value_length == 0) return NULL;

  char* storage = *inout_char_buffer;
  memcpy(storage, value, value_length);
  storage[value_length] = 0;
  *inout_char_buffer = storage + value_length + 1;
  return storage;
}

static bool iree_hal_amdgpu_trace_src_loc_has_data(
    const iree_hal_amdgpu_trace_src_loc_t* trace_src_loc) {
  return trace_src_loc && (trace_src_loc->function || trace_src_loc->file ||
                           trace_src_loc->line != 0);
}

typedef struct iree_hal_amdgpu_executable_find_loaded_code_object_state_t {
  // Borrowed HSA API table used for loader extension queries.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // HSA agent whose loaded code object is being searched.
  hsa_agent_t agent;
  // Loaded code object matching |agent| when found.
  hsa_loaded_code_object_t loaded_code_object;
} iree_hal_amdgpu_executable_find_loaded_code_object_state_t;

static hsa_status_t iree_hal_amdgpu_executable_iterate_loaded_code_object(
    hsa_executable_t executable, hsa_loaded_code_object_t loaded_code_object,
    void* user_data) {
  iree_hal_amdgpu_executable_find_loaded_code_object_state_t* find_state =
      (iree_hal_amdgpu_executable_find_loaded_code_object_state_t*)user_data;
  hsa_agent_t agent = {0};
  hsa_status_t hsa_status =
      find_state->libhsa->amd_loader
          .hsa_ven_amd_loader_loaded_code_object_get_info(
              loaded_code_object,
              HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT, &agent);
  if (hsa_status != HSA_STATUS_SUCCESS) return hsa_status;
  if (agent.handle == find_state->agent.handle) {
    find_state->loaded_code_object = loaded_code_object;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

static iree_status_t iree_hal_amdgpu_executable_find_loaded_code_object(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    hsa_agent_t agent, hsa_loaded_code_object_t* out_loaded_code_object) {
  *out_loaded_code_object = (hsa_loaded_code_object_t){0};
  iree_hal_amdgpu_executable_find_loaded_code_object_state_t find_state = {
      .libhsa = libhsa,
      .agent = agent,
      .loaded_code_object = {0},
  };
  hsa_status_t hsa_status =
      libhsa->amd_loader
          .hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
              executable, iree_hal_amdgpu_executable_iterate_loaded_code_object,
              &find_state);
  if (hsa_status == HSA_STATUS_SUCCESS) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no loaded code object found for agent");
  }
  if (hsa_status != HSA_STATUS_INFO_BREAK) {
    return iree_status_from_hsa_status(
        __FILE__, __LINE__, hsa_status,
        "hsa_ven_amd_loader_executable_iterate_loaded_code_objects",
        "iterating loaded executable code objects");
  }
  *out_loaded_code_object = find_state.loaded_code_object;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_executable_populate_profile_code_object_load_info(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    uint32_t physical_device_ordinal, hsa_agent_t device_agent,
    iree_hal_amdgpu_profile_code_object_load_info_t* out_load_info) {
  memset(out_load_info, 0, sizeof(*out_load_info));
  out_load_info->physical_device_ordinal = physical_device_ordinal;

  hsa_loaded_code_object_t loaded_code_object = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_find_loaded_code_object(
      libhsa, executable, device_agent, &loaded_code_object));

  hsa_status_t hsa_status =
      libhsa->amd_loader.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
          &out_load_info->load_delta);
  if (hsa_status == HSA_STATUS_SUCCESS) {
    hsa_status =
        libhsa->amd_loader.hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
            &out_load_info->load_size);
  }
  return iree_status_from_hsa_status(
      __FILE__, __LINE__, hsa_status,
      "hsa_ven_amd_loader_loaded_code_object_get_info",
      "querying loaded executable code-object profile metadata");
}

#define IREE_HAL_AMDGPU_MAX_STACK_SYMBOL_NAME_LENGTH \
  ((iree_host_size_t)(4 * 1024))

static iree_status_t iree_hal_amdgpu_executable_get_symbol_by_cstring(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    const char* symbol_name, hsa_agent_t device_agent,
    hsa_executable_symbol_t* out_symbol) {
  // NOTE: AMDGPU kernel symbols must include the `.kd` suffix.
  return iree_hsa_executable_get_symbol_by_name(
      IREE_LIBHSA(libhsa), executable, symbol_name, &device_agent, out_symbol);
}

static iree_status_t iree_hal_amdgpu_executable_get_raw_hsaco_symbol_by_name(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    iree_string_view_t symbol_name, hsa_agent_t device_agent,
    hsa_executable_symbol_t* out_symbol) {
  if (iree_string_view_is_empty(symbol_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable kernel symbol name is empty");
  }
  if (symbol_name.size > IREE_HAL_AMDGPU_MAX_STACK_SYMBOL_NAME_LENGTH) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "executable kernel symbol name `%.*s` exceeds maximum length %" PRIhsz,
        (int)symbol_name.size, symbol_name.data,
        IREE_HAL_AMDGPU_MAX_STACK_SYMBOL_NAME_LENGTH);
  }

  // AMDGPU MessagePack strings are length-delimited and not NUL-terminated.
  // Copy only at the HSA API boundary so ROCR can use its internal symbol map.
  char* symbol_name_storage = (char*)iree_alloca(symbol_name.size + 1);
  memcpy(symbol_name_storage, symbol_name.data, symbol_name.size);
  symbol_name_storage[symbol_name.size] = 0;
  return iree_hal_amdgpu_executable_get_symbol_by_cstring(
      libhsa, executable, symbol_name_storage, device_agent, out_symbol);
}

// Resolves the uniform kernel arguments that are the same on all GPU device
// agents in the topology (since we assume all are the same device type).
// All fields besides `kernel_object` will have valid values.
static iree_status_t iree_hal_amdgpu_executable_resolve_kernel_args_from_symbol(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_symbol_t symbol,
    const uint32_t workgroup_size[3], uint16_t constant_count,
    uint16_t binding_count, const iree_hal_amdgpu_trace_src_loc_t* export_loc,
    iree_hal_amdgpu_device_kernel_args_t* out_kernel_args) {
  IREE_ASSERT_ARGUMENT(out_kernel_args);
  IREE_TRACE_ZONE_BEGIN(z0);

  // All of our kernels assume 3 dimensions.
  out_kernel_args->setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

  out_kernel_args->workgroup_size[0] = workgroup_size[0];
  out_kernel_args->workgroup_size[1] = workgroup_size[1];
  out_kernel_args->workgroup_size[2] = workgroup_size[2];

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

  out_kernel_args->binding_count = binding_count;
  out_kernel_args->constant_count = constant_count;

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
    iree_hal_amdgpu_trace_src_loc_t** out_export_locs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t export_count =
      iree_hal_amdgpu_ExportDef_vec_len(export_defs);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, export_count);

  // Sum up the total storage required for all information.
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul(
          export_count, sizeof(iree_hal_amdgpu_trace_src_loc_t), &total_size)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "AMDGPU executable trace source location table size overflow"));
  }
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_amdgpu_ExportDef_table_t export_def =
        iree_hal_amdgpu_ExportDef_vec_at(export_defs, i);
    iree_hal_debug_ExportDef_table_t debug_def =
        iree_hal_amdgpu_ExportDef_debug_info_get(export_def);
    if (!debug_def) continue;

    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdgpu_executable_add_trace_string_storage(
                iree_hal_debug_ExportDef_name_get(debug_def), &total_size));
    iree_hal_debug_FileLineLocDef_table_t loc_def =
        iree_hal_debug_ExportDef_location_get(debug_def);
    if (loc_def) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdgpu_executable_add_trace_string_storage(
                  iree_hal_debug_FileLineLocDef_filename_get(loc_def),
                  &total_size));
    }
  }

  if (total_size == 0) {
    *out_export_locs = NULL;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Allocate persistent storage.
  iree_hal_amdgpu_trace_src_loc_t* export_locs = NULL;
  IREE_LEAK_CHECK_DISABLE_PUSH();
  export_locs = (iree_hal_amdgpu_trace_src_loc_t*)malloc(total_size);
  IREE_LEAK_CHECK_DISABLE_POP();
  if (!export_locs) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_RESOURCE_EXHAUSTED,
                "AMDGPU executable trace source location storage allocation "
                "failed"));
  }
  memset(export_locs, 0, total_size);
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
    if (!debug_def) continue;

    export_loc->function = iree_hal_amdgpu_executable_copy_trace_string(
        iree_hal_debug_ExportDef_name_get(debug_def), &char_buffer);

    iree_hal_debug_FileLineLocDef_table_t loc_def =
        iree_hal_debug_ExportDef_location_get(debug_def);
    if (loc_def) {
      export_loc->file = iree_hal_amdgpu_executable_copy_trace_string(
          iree_hal_debug_FileLineLocDef_filename_get(loc_def), &char_buffer);
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

static iree_status_t iree_hal_amdgpu_executable_resolve_kernel_object(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_symbol_t symbol,
    uint64_t* out_kernel_object) {
  return iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
      out_kernel_object);
}

// Uploads the provided kernel table to |device_agent| and returns the pointer.
// |host_kernel_args| must already have device-specific `kernel_object` fields.
static iree_status_t iree_hal_amdgpu_executable_upload_resolved_kernel_table(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_host_size_t kernel_count,
    iree_hal_amdgpu_device_kernel_args_t* host_kernel_args,
    hsa_agent_t device_agent,
    IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t**
        out_device_kernel_args) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device_kernel_args = NULL;

  if (kernel_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
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
    status = iree_status_join(
        status,
        iree_hsa_amd_memory_pool_free(IREE_LIBHSA(libhsa), device_kernel_args));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_executable_upload_flatbuffer_kernel_table(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    iree_hal_amdgpu_ExportDef_vec_t export_defs,
    iree_hal_amdgpu_device_kernel_args_t* host_kernel_args,
    hsa_agent_t device_agent,
    IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t**
        out_device_kernel_args) {
  const iree_host_size_t kernel_count =
      iree_hal_amdgpu_ExportDef_vec_len(export_defs);
  for (iree_host_size_t kernel_ordinal = 0; kernel_ordinal < kernel_count;
       ++kernel_ordinal) {
    iree_hal_amdgpu_ExportDef_table_t export_def =
        iree_hal_amdgpu_ExportDef_vec_at(export_defs, kernel_ordinal);
    flatbuffers_string_t symbol_name =
        iree_hal_amdgpu_ExportDef_symbol_name_get(export_def);
    iree_string_view_t symbol_name_view =
        symbol_name ? iree_make_string_view(symbol_name,
                                            flatbuffers_string_len(symbol_name))
                    : iree_string_view_empty();
    if (iree_string_view_is_empty(symbol_name_view)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable kernel symbol name is empty");
    }
    hsa_executable_symbol_t symbol = {0};
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_get_symbol_by_cstring(
            libhsa, executable, symbol_name, device_agent, &symbol),
        "resolving `%.*s`", (int)symbol_name_view.size, symbol_name_view.data);
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_resolve_kernel_object(
        libhsa, symbol, &host_kernel_args[kernel_ordinal].kernel_object));
  }
  return iree_hal_amdgpu_executable_upload_resolved_kernel_table(
      libhsa, kernel_count, host_kernel_args, device_agent,
      out_device_kernel_args);
}

static iree_status_t iree_hal_amdgpu_executable_upload_raw_hsaco_kernel_table(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_hal_amdgpu_device_kernel_args_t* host_kernel_args,
    hsa_agent_t device_agent,
    IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t**
        out_device_kernel_args) {
  for (iree_host_size_t kernel_ordinal = 0;
       kernel_ordinal < hsaco_metadata->kernel_count; ++kernel_ordinal) {
    iree_string_view_t symbol_name =
        hsaco_metadata->kernels[kernel_ordinal].symbol_name;
    hsa_executable_symbol_t symbol = {0};
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_get_raw_hsaco_symbol_by_name(
            libhsa, executable, symbol_name, device_agent, &symbol),
        "resolving `%.*s`", (int)symbol_name.size, symbol_name.data);
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_resolve_kernel_object(
        libhsa, symbol, &host_kernel_args[kernel_ordinal].kernel_object));
  }
  return iree_hal_amdgpu_executable_upload_resolved_kernel_table(
      libhsa, hsaco_metadata->kernel_count, host_kernel_args, device_agent,
      out_device_kernel_args);
}

static iree_status_t iree_hal_amdgpu_executable_calculate_kernarg_block_count(
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout,
    uint32_t* out_kernarg_block_count) {
  iree_host_size_t kernarg_block_count = iree_host_size_ceil_div(
      layout->total_kernarg_size, sizeof(iree_hal_amdgpu_kernarg_block_t));
  if (kernarg_block_count == 0) {
    kernarg_block_count = 1;
  }
  if (IREE_UNLIKELY(kernarg_block_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch kernargs require too many blocks (%" PRIhsz ", max=%u)",
        kernarg_block_count, UINT32_MAX);
  }
  *out_kernarg_block_count = (uint32_t)kernarg_block_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_initialize_dispatch_descriptor(
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    iree_hal_amdgpu_executable_dispatch_descriptor_t* out_descriptor) {
  memset(out_descriptor, 0, sizeof(*out_descriptor));

  if (IREE_UNLIKELY(
          !iree_host_size_is_power_of_two(kernel_args->kernarg_alignment))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable kernel kernarg alignment must be a power of two (got %u)",
        kernel_args->kernarg_alignment);
  }
  if (IREE_UNLIKELY(kernel_args->kernarg_alignment >
                    iree_alignof(iree_hal_amdgpu_kernarg_block_t))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable kernel kernarg alignment %u exceeds queue kernarg ring "
        "alignment %" PRIhsz,
        kernel_args->kernarg_alignment,
        (iree_host_size_t)iree_alignof(iree_hal_amdgpu_kernarg_block_t));
  }

  out_descriptor->kernel_args = *kernel_args;
  // Kernel metadata reports the bytes actually consumed by the compiled
  // kernel. That may be smaller than the HAL ABI explicit argument footprint
  // when optimization removes unused bindings/constants; we still reserve and
  // populate the public HAL ABI bytes for dispatch.
  out_descriptor->hal_kernarg_layout =
      iree_hal_amdgpu_device_dispatch_make_hal_kernarg_layout(kernel_args);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_calculate_kernarg_block_count(
      &out_descriptor->hal_kernarg_layout,
      &out_descriptor->hal_kernarg_block_count));

  out_descriptor->custom_kernarg_layout =
      iree_hal_amdgpu_device_dispatch_make_custom_kernarg_layout(
          kernel_args->kernarg_size);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_calculate_kernarg_block_count(
      &out_descriptor->custom_kernarg_layout,
      &out_descriptor->custom_kernarg_block_count));

  for (iree_host_size_t i = 0; i < 3; ++i) {
    if (IREE_UNLIKELY(kernel_args->workgroup_size[i] == 0)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "executable kernel workgroup size dimension %" PRIhsz " is zero", i);
    }
    out_descriptor->max_workgroup_count[i] =
        UINT32_MAX / kernel_args->workgroup_size[i];
  }
  out_descriptor->max_dynamic_workgroup_local_memory =
      UINT32_MAX - kernel_args->group_segment_size;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_executable_t {
  // HAL executable resource header.
  iree_hal_resource_t resource;
  // Host allocator used for executable-owned metadata tables.
  iree_allocator_t host_allocator;

  // Unowned HSA API handle. Must remain valid for the lifetime of the
  // executable.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Loaded HSA executable with a code object for each device.
  hsa_executable_t handle;

  // Producer-local profile executable id assigned at creation.
  uint64_t profile_id;
  // Stable content hash for the exact loaded HSACO/code-object bytes.
  uint64_t profile_code_object_hash[2];

  // Total number of exports in the executable.
  iree_host_size_t kernel_count;
  // Host-resident reflection information for each export.
  iree_hal_executable_export_info_t* export_infos /*[kernel_count]*/;
  // Prefix-sum offsets into |export_parameters| for each export plus a
  // sentinel.
  iree_host_size_t* export_parameter_offsets /*[kernel_count + 1]*/;
  // Host-resident parameter reflection records for all exports.
  iree_hal_executable_export_parameter_t* export_parameters;
  // Table of kernel args stored in host memory. We have them local so that
  // host-side command buffer recording doesn't need to access device memory.
  // The kernel object specified in each is invalid as it's agent-specific.
  iree_hal_amdgpu_device_kernel_args_t* host_kernel_args /*[kernel_count]*/;
  // Host-resident dispatch descriptors stored as [device_count][kernel_count].
  iree_hal_amdgpu_executable_dispatch_descriptor_t*
      host_dispatch_descriptors /*[device_count * kernel_count]*/;

  // Queue affinity this executable was loaded for after normalization.
  iree_hal_queue_affinity_t queue_affinity;
  // Bitmask of physical GPU device ordinals with loaded code objects.
  uint64_t loaded_physical_device_mask;
  // Number of loaded physical GPU devices in |loaded_physical_device_mask|.
  iree_host_size_t loaded_physical_device_count;
  // Total number of GPU devices in the topology used for per-device tables.
  iree_host_size_t device_count;
  // Table of kernel args stored in device memory, one copy per device.
  // Selected devices have an entire `kernel_count` set of args; unselected
  // devices remain NULL and fail lookup.
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

static iree_string_view_t iree_hal_amdgpu_executable_flatbuffer_string_view(
    flatbuffers_string_t value) {
  return value ? iree_make_string_view(value, flatbuffers_string_len(value))
               : iree_string_view_empty();
}

static iree_string_view_t iree_hal_amdgpu_executable_export_reflection_name(
    iree_hal_amdgpu_ExportDef_table_t export_def) {
  iree_hal_debug_ExportDef_table_t debug_def =
      iree_hal_amdgpu_ExportDef_debug_info_get(export_def);
  if (debug_def) {
    iree_string_view_t debug_name =
        iree_hal_amdgpu_executable_flatbuffer_string_view(
            iree_hal_debug_ExportDef_name_get(debug_def));
    if (!iree_string_view_is_empty(debug_name)) return debug_name;
  }

  iree_string_view_t symbol_name =
      iree_hal_amdgpu_executable_flatbuffer_string_view(
          iree_hal_amdgpu_ExportDef_symbol_name_get(export_def));
  return iree_string_view_strip_suffix(symbol_name, IREE_SV(".kd"));
}

static iree_status_t iree_hal_amdgpu_executable_allocate(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        physical_devices,
    iree_host_size_t export_count, iree_host_size_t export_name_storage_size,
    iree_host_size_t export_parameter_count,
    iree_host_size_t export_parameter_name_storage_size,
    iree_allocator_t host_allocator, char** out_export_name_storage,
    char** out_export_parameter_name_storage,
    iree_hal_amdgpu_executable_t** out_executable) {
  *out_export_name_storage = NULL;
  *out_export_parameter_name_storage = NULL;
  *out_executable = NULL;

  iree_host_size_t dispatch_descriptor_count = 0;
  if (!iree_host_size_checked_mul(topology->gpu_agent_count, export_count,
                                  &dispatch_descriptor_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch descriptor table size overflow");
  }

  iree_host_size_t export_parameter_offset_count = 0;
  if (!iree_host_size_checked_add(export_count, 1,
                                  &export_parameter_offset_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export parameter offset table size overflow");
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t export_infos_offset = 0;
  iree_host_size_t export_name_storage_offset = 0;
  iree_host_size_t export_parameter_offsets_offset = 0;
  iree_host_size_t export_parameters_offset = 0;
  iree_host_size_t export_parameter_name_storage_offset = 0;
  iree_host_size_t host_kernel_args_offset = 0;
  iree_host_size_t host_dispatch_descriptors_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_executable_t), &total_size,
      IREE_STRUCT_FIELD_FAM(
          topology->gpu_agent_count,
          IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t*),
      IREE_STRUCT_FIELD(export_count, iree_hal_executable_export_info_t,
                        &export_infos_offset),
      IREE_STRUCT_FIELD(export_name_storage_size, char,
                        &export_name_storage_offset),
      IREE_STRUCT_FIELD(export_parameter_offset_count, iree_host_size_t,
                        &export_parameter_offsets_offset),
      IREE_STRUCT_FIELD(export_parameter_count,
                        iree_hal_executable_export_parameter_t,
                        &export_parameters_offset),
      IREE_STRUCT_FIELD(export_parameter_name_storage_size, char,
                        &export_parameter_name_storage_offset),
      IREE_STRUCT_FIELD(export_count, iree_hal_amdgpu_device_kernel_args_t,
                        &host_kernel_args_offset),
      IREE_STRUCT_FIELD(dispatch_descriptor_count,
                        iree_hal_amdgpu_executable_dispatch_descriptor_t,
                        &host_dispatch_descriptors_offset)));

  iree_hal_amdgpu_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  memset(executable, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_amdgpu_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->libhsa = libhsa;
  executable->kernel_count = export_count;
  uint8_t* executable_storage = (uint8_t*)executable;
  executable->export_infos =
      (iree_hal_executable_export_info_t*)(executable_storage +
                                           export_infos_offset);
  executable->export_parameter_offsets =
      (iree_host_size_t*)(executable_storage + export_parameter_offsets_offset);
  executable->export_parameters =
      export_parameter_count
          ? (iree_hal_executable_export_parameter_t*)(executable_storage +
                                                      export_parameters_offset)
          : NULL;
  executable->host_kernel_args =
      (iree_hal_amdgpu_device_kernel_args_t*)(executable_storage +
                                              host_kernel_args_offset);
  executable->host_dispatch_descriptors =
      (iree_hal_amdgpu_executable_dispatch_descriptor_t*)(executable_storage +
                                                          host_dispatch_descriptors_offset);
  executable->queue_affinity = physical_devices->queue_affinity;
  executable->loaded_physical_device_mask =
      physical_devices->physical_device_mask;
  executable->loaded_physical_device_count =
      physical_devices->physical_device_count;
  executable->device_count = topology->gpu_agent_count;

  *out_export_name_storage =
      (char*)executable_storage + export_name_storage_offset;
  *out_export_parameter_name_storage =
      (char*)executable_storage + export_parameter_name_storage_offset;
  *out_executable = executable;
  return iree_ok_status();
}

static void iree_hal_amdgpu_executable_invalidate_host_kernel_objects(
    iree_hal_amdgpu_executable_t* executable) {
  if (!executable) return;
  for (iree_host_size_t kernel_ordinal = 0;
       kernel_ordinal < executable->kernel_count; ++kernel_ordinal) {
    executable->host_kernel_args[kernel_ordinal].kernel_object = 0;
  }
}

static iree_status_t
iree_hal_amdgpu_executable_initialize_dispatch_descriptors_for_device(
    iree_hal_amdgpu_executable_t* executable, iree_host_size_t device_ordinal) {
  for (iree_host_size_t kernel_ordinal = 0;
       kernel_ordinal < executable->kernel_count; ++kernel_ordinal) {
    const iree_host_size_t descriptor_ordinal =
        device_ordinal * executable->kernel_count + kernel_ordinal;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_initialize_dispatch_descriptor(
            &executable->host_kernel_args[kernel_ordinal],
            &executable->host_dispatch_descriptors[descriptor_ordinal]),
        "initializing dispatch descriptor for device %" PRIhsz
        " export %" PRIhsz,
        device_ordinal, kernel_ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_register_profile_artifacts(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_const_byte_span_t code_object_data,
    iree_hal_amdgpu_executable_t* executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t load_info_storage_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &load_info_storage_size,
              IREE_STRUCT_FIELD(executable->loaded_physical_device_count,
                                iree_hal_amdgpu_profile_code_object_load_info_t,
                                NULL)));

  iree_hal_amdgpu_profile_code_object_load_info_t* load_infos = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(executable->host_allocator,
                                load_info_storage_size, (void**)&load_infos));

  iree_status_t status = iree_ok_status();
  iree_host_size_t load_info_ordinal = 0;
  for (iree_host_size_t device_ordinal = 0;
       device_ordinal < topology->gpu_agent_count && iree_status_is_ok(status);
       ++device_ordinal) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            executable->loaded_physical_device_mask, device_ordinal)) {
      continue;
    }
    if (IREE_UNLIKELY(device_ordinal > UINT32_MAX)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "profile executable physical device ordinal exceeds uint32_t");
    } else {
      status =
          iree_hal_amdgpu_executable_populate_profile_code_object_load_info(
              libhsa, executable->handle, (uint32_t)device_ordinal,
              topology->gpu_agents[device_ordinal],
              &load_infos[load_info_ordinal]);
      if (iree_status_is_ok(status)) ++load_info_ordinal;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_metadata_register_executable_artifacts(
        profile_metadata, executable->profile_id, code_object_data,
        executable->profile_code_object_hash,
        executable->loaded_physical_device_count, load_infos);
  }

  iree_allocator_free(executable->host_allocator, load_infos);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_executable_register_profile_metadata(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_const_byte_span_t code_object_data, bool retain_profile_artifacts,
    iree_hal_amdgpu_executable_t* executable) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_metadata_register_executable(
      profile_metadata, executable->kernel_count, executable->export_infos,
      executable->export_parameter_offsets,
      executable->profile_code_object_hash, executable->host_kernel_args,
      &executable->profile_id));

  // Code-object image and loader load-range capture is optional for normal
  // execution, but explicit retention requests must succeed while the borrowed
  // code-object bytes are still in scope.
  if (retain_profile_artifacts) {
    return iree_hal_amdgpu_executable_register_profile_artifacts(
        libhsa, topology, profile_metadata, code_object_data, executable);
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_executable_validate_export_parameter_requirements(
    iree_hal_amdgpu_ExportDef_table_t export_def,
    iree_string_view_t symbol_name,
    const iree_hal_amdgpu_hsaco_metadata_export_parameter_requirements_t*
        requirements) {
  // The flatbuffer owns the HAL ABI counts for wrapped executables. HSACO
  // metadata may omit arguments that LLVM optimized away, but it must not
  // require more visible arguments than the flatbuffer layout can supply.
  const uint32_t expected_constant_count =
      iree_hal_amdgpu_ExportDef_constant_count_get(export_def);
  if (requirements->constant_count > expected_constant_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HSACO metadata for export `%.*s` declares %u reflected constants but "
        "ExecutableDef only declares %u",
        (int)symbol_name.size, symbol_name.data,
        (uint32_t)requirements->constant_count, expected_constant_count);
  }

  iree_hal_amdgpu_BindingBits_vec_t binding_flags =
      iree_hal_amdgpu_ExportDef_binding_flags_get(export_def);
  const iree_host_size_t expected_binding_count =
      iree_hal_amdgpu_BindingBits_vec_len(binding_flags);
  if (requirements->binding_count > expected_binding_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HSACO metadata for export `%.*s` declares %u reflected bindings but "
        "ExecutableDef only declares %" PRIhsz,
        (int)symbol_name.size, symbol_name.data,
        (uint32_t)requirements->binding_count, expected_binding_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_calculate_reflection_storage(
    iree_hal_amdgpu_ExportDef_vec_t export_defs,
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_host_size_t* out_export_name_storage_size,
    iree_host_size_t* out_export_parameter_count,
    iree_host_size_t* out_export_parameter_name_storage_size) {
  iree_host_size_t export_name_storage_size = 0;
  iree_host_size_t export_parameter_count = 0;
  iree_host_size_t export_parameter_name_storage_size = 0;
  const iree_host_size_t export_count =
      iree_hal_amdgpu_ExportDef_vec_len(export_defs);
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_amdgpu_ExportDef_table_t export_def =
        iree_hal_amdgpu_ExportDef_vec_at(export_defs, i);
    iree_string_view_t name =
        iree_hal_amdgpu_executable_export_reflection_name(export_def);
    if (!iree_host_size_checked_add(export_name_storage_size, name.size,
                                    &export_name_storage_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "export name storage size overflow");
    }

    iree_string_view_t symbol_name =
        iree_hal_amdgpu_executable_flatbuffer_string_view(
            iree_hal_amdgpu_ExportDef_symbol_name_get(export_def));
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_metadata_find_kernel_by_symbol(
                             hsaco_metadata, symbol_name, &kernel),
                         "looking up HSACO metadata for export `%.*s`",
                         (int)symbol_name.size, symbol_name.data);

    iree_hal_amdgpu_hsaco_metadata_export_parameter_requirements_t requirements;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_metadata_calculate_default_export_parameter_requirements(
            kernel, &requirements),
        "projecting HSACO parameters for export `%.*s`", (int)symbol_name.size,
        symbol_name.data);
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_validate_export_parameter_requirements(
            export_def, symbol_name, &requirements));

    if (!iree_host_size_checked_add(export_parameter_count,
                                    requirements.parameter_count,
                                    &export_parameter_count) ||
        !iree_host_size_checked_add(export_parameter_name_storage_size,
                                    requirements.name_storage_size,
                                    &export_parameter_name_storage_size)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "export parameter reflection storage size overflow");
    }
  }
  *out_export_name_storage_size = export_name_storage_size;
  *out_export_parameter_count = export_parameter_count;
  *out_export_parameter_name_storage_size = export_parameter_name_storage_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_verify_raw_hsaco_kernel(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    const iree_hal_amdgpu_device_limits_t* limits) {
  if (!kernel->has_required_workgroup_size) {
    return iree_ok_status();
  }

  const uint32_t* workgroup_size = kernel->required_workgroup_size;
  if (workgroup_size[0] > limits->max_workgroup_size_per_dim[0] ||
      workgroup_size[1] > limits->max_workgroup_size_per_dim[1] ||
      workgroup_size[2] > limits->max_workgroup_size_per_dim[2]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw HSACO kernel `%.*s` workgroup size dims %ux%ux%u exceed device "
        "maximum %ux%ux%u",
        (int)kernel->symbol_name.size, kernel->symbol_name.data,
        workgroup_size[0], workgroup_size[1], workgroup_size[2],
        limits->max_workgroup_size_per_dim[0],
        limits->max_workgroup_size_per_dim[1],
        limits->max_workgroup_size_per_dim[2]);
  }
  const uint64_t total_workgroup_size =
      (uint64_t)workgroup_size[0] * workgroup_size[1] * workgroup_size[2];
  if (total_workgroup_size > limits->max_workgroup_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw HSACO kernel `%.*s` workgroup size total %" PRIu64
        " exceeds device "
        "maximum %u",
        (int)kernel->symbol_name.size, kernel->symbol_name.data,
        total_workgroup_size, limits->max_workgroup_size);
  }
  return iree_ok_status();
}

static void iree_hal_amdgpu_executable_raw_hsaco_workgroup_size(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    uint32_t out_workgroup_size[3]) {
  if (kernel->has_required_workgroup_size) {
    out_workgroup_size[0] = kernel->required_workgroup_size[0];
    out_workgroup_size[1] = kernel->required_workgroup_size[1];
    out_workgroup_size[2] = kernel->required_workgroup_size[2];
  } else {
    // Raw HSACO without `.reqd_workgroup_size` is represented as a dynamic
    // workgroup-size export with 1x1x1 minimum granularity. The actual launch
    // geometry must come from the dispatch config.
    out_workgroup_size[0] = 1;
    out_workgroup_size[1] = 1;
    out_workgroup_size[2] = 1;
  }
}

static iree_status_t
iree_hal_amdgpu_executable_calculate_raw_hsaco_reflection_storage(
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    const iree_hal_amdgpu_device_limits_t* limits,
    iree_host_size_t* out_export_name_storage_size,
    iree_host_size_t* out_export_parameter_count,
    iree_host_size_t* out_export_parameter_name_storage_size) {
  if (iree_string_view_is_empty(hsaco_metadata->target)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw HSACO metadata is missing `amdhsa.target`; direct loading "
        "requires the code object to declare its target ISA");
  }

  iree_host_size_t export_name_storage_size = 0;
  iree_host_size_t export_parameter_count = 0;
  iree_host_size_t export_parameter_name_storage_size = 0;
  for (iree_host_size_t i = 0; i < hsaco_metadata->kernel_count; ++i) {
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel =
        &hsaco_metadata->kernels[i];
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_verify_raw_hsaco_kernel(kernel, limits));
    if (!iree_host_size_checked_add(export_name_storage_size,
                                    kernel->reflection_name.size,
                                    &export_name_storage_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "export name storage size overflow");
    }

    iree_hal_amdgpu_hsaco_metadata_export_parameter_requirements_t requirements;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_metadata_calculate_default_export_parameter_requirements(
            kernel, &requirements),
        "projecting HSACO parameters for raw kernel `%.*s`",
        (int)kernel->symbol_name.size, kernel->symbol_name.data);
    if (!iree_host_size_checked_add(export_parameter_count,
                                    requirements.parameter_count,
                                    &export_parameter_count) ||
        !iree_host_size_checked_add(export_parameter_name_storage_size,
                                    requirements.name_storage_size,
                                    &export_parameter_name_storage_size)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "export parameter reflection storage size overflow");
    }
  }
  *out_export_name_storage_size = export_name_storage_size;
  *out_export_parameter_count = export_parameter_count;
  *out_export_parameter_name_storage_size = export_parameter_name_storage_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_initialize_export_infos(
    iree_hal_amdgpu_ExportDef_vec_t export_defs,
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_hal_executable_export_info_t* export_infos,
    iree_host_size_t* export_parameter_offsets,
    iree_hal_executable_export_parameter_t* export_parameters,
    char* export_name_storage, char* export_parameter_name_storage) {
  iree_host_size_t export_parameter_offset = 0;
  const iree_host_size_t export_count =
      iree_hal_amdgpu_ExportDef_vec_len(export_defs);
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_amdgpu_ExportDef_table_t export_def =
        iree_hal_amdgpu_ExportDef_vec_at(export_defs, i);
    iree_hal_executable_export_info_t* info = &export_infos[i];
    export_parameter_offsets[i] = export_parameter_offset;

    iree_string_view_t symbol_name =
        iree_hal_amdgpu_executable_flatbuffer_string_view(
            iree_hal_amdgpu_ExportDef_symbol_name_get(export_def));
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_metadata_find_kernel_by_symbol(
                             hsaco_metadata, symbol_name, &kernel),
                         "looking up HSACO metadata for export `%.*s`",
                         (int)symbol_name.size, symbol_name.data);

    iree_hal_amdgpu_hsaco_metadata_export_parameter_requirements_t requirements;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_metadata_calculate_default_export_parameter_requirements(
            kernel, &requirements),
        "projecting HSACO parameters for export `%.*s`", (int)symbol_name.size,
        symbol_name.data);
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_validate_export_parameter_requirements(
            export_def, symbol_name, &requirements));

    iree_string_view_t name =
        iree_hal_amdgpu_executable_export_reflection_name(export_def);
    if (!iree_string_view_is_empty(name)) {
      memcpy(export_name_storage, name.data, name.size);
    }

    memset(info, 0, sizeof(*info));
    info->name = iree_make_string_view(export_name_storage, name.size);
    info->flags = IREE_HAL_EXECUTABLE_EXPORT_FLAG_NONE;
    // Preserve the flatbuffer ABI counts even when the HSACO metadata has lost
    // optimized-unused arguments.
    info->constant_count =
        (uint16_t)iree_hal_amdgpu_ExportDef_constant_count_get(export_def);
    iree_hal_amdgpu_BindingBits_vec_t binding_flags =
        iree_hal_amdgpu_ExportDef_binding_flags_get(export_def);
    info->binding_count =
        (uint16_t)iree_hal_amdgpu_BindingBits_vec_len(binding_flags);
    info->parameter_count = requirements.parameter_count;
    const iree_hal_amdgpu_Dims_struct_t workgroup_size =
        iree_hal_amdgpu_ExportDef_workgroup_size_get(export_def);
    info->workgroup_size[0] = workgroup_size->x;
    info->workgroup_size[1] = workgroup_size->y;
    info->workgroup_size[2] = workgroup_size->z;

    iree_hal_executable_export_parameter_t* export_parameter_base =
        requirements.parameter_count
            ? &export_parameters[export_parameter_offset]
            : NULL;
    char* export_parameter_name_base =
        requirements.name_storage_size ? export_parameter_name_storage : NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_metadata_populate_default_export_parameters(
            kernel, requirements.parameter_count, export_parameter_base,
            requirements.name_storage_size, export_parameter_name_base),
        "populating reflected parameters for export `%.*s`",
        (int)symbol_name.size, symbol_name.data);

    export_name_storage += name.size;
    export_parameter_offset += requirements.parameter_count;
    export_parameter_name_storage += requirements.name_storage_size;
  }
  export_parameter_offsets[export_count] = export_parameter_offset;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_executable_initialize_raw_hsaco_export_infos(
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_hal_executable_export_info_t* export_infos,
    iree_host_size_t* export_parameter_offsets,
    iree_hal_executable_export_parameter_t* export_parameters,
    char* export_name_storage, char* export_parameter_name_storage) {
  iree_host_size_t export_parameter_offset = 0;
  for (iree_host_size_t i = 0; i < hsaco_metadata->kernel_count; ++i) {
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel =
        &hsaco_metadata->kernels[i];
    iree_hal_executable_export_info_t* info = &export_infos[i];
    export_parameter_offsets[i] = export_parameter_offset;

    iree_hal_amdgpu_hsaco_metadata_export_parameter_requirements_t requirements;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_metadata_calculate_default_export_parameter_requirements(
            kernel, &requirements),
        "projecting HSACO parameters for raw kernel `%.*s`",
        (int)kernel->symbol_name.size, kernel->symbol_name.data);

    iree_string_view_t name = kernel->reflection_name;
    if (!iree_string_view_is_empty(name)) {
      memcpy(export_name_storage, name.data, name.size);
    }

    memset(info, 0, sizeof(*info));
    info->name = iree_make_string_view(export_name_storage, name.size);
    info->flags = kernel->has_required_workgroup_size
                      ? IREE_HAL_EXECUTABLE_EXPORT_FLAG_NONE
                      : IREE_HAL_EXECUTABLE_EXPORT_FLAG_WORKGROUP_SIZE_DYNAMIC;
    info->constant_count = requirements.constant_count;
    info->binding_count = requirements.binding_count;
    info->parameter_count = requirements.parameter_count;
    iree_hal_amdgpu_executable_raw_hsaco_workgroup_size(kernel,
                                                        info->workgroup_size);

    iree_hal_executable_export_parameter_t* export_parameter_base =
        requirements.parameter_count
            ? &export_parameters[export_parameter_offset]
            : NULL;
    char* export_parameter_name_base =
        requirements.name_storage_size ? export_parameter_name_storage : NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_metadata_populate_default_export_parameters(
            kernel, requirements.parameter_count, export_parameter_base,
            requirements.name_storage_size, export_parameter_name_base),
        "populating reflected parameters for raw kernel `%.*s`",
        (int)kernel->symbol_name.size, kernel->symbol_name.data);

    export_name_storage += name.size;
    export_parameter_offset += requirements.parameter_count;
    export_parameter_name_storage += requirements.name_storage_size;
  }
  export_parameter_offsets[hsaco_metadata->kernel_count] =
      export_parameter_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_resolve_flatbuffer_kernel_args(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    iree_hal_amdgpu_ExportDef_vec_t export_defs,
    const iree_hal_amdgpu_trace_src_loc_t* export_locs,
    hsa_agent_t any_device_agent,
    iree_hal_amdgpu_device_kernel_args_t* host_kernel_args) {
  const iree_host_size_t kernel_count =
      iree_hal_amdgpu_ExportDef_vec_len(export_defs);
  for (iree_host_size_t kernel_ordinal = 0; kernel_ordinal < kernel_count;
       ++kernel_ordinal) {
    iree_hal_amdgpu_ExportDef_table_t export_def =
        iree_hal_amdgpu_ExportDef_vec_at(export_defs, kernel_ordinal);
    flatbuffers_string_t symbol_name =
        iree_hal_amdgpu_ExportDef_symbol_name_get(export_def);
    iree_string_view_t symbol_name_view =
        iree_hal_amdgpu_executable_flatbuffer_string_view(symbol_name);
    if (iree_string_view_is_empty(symbol_name_view)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable kernel symbol name is empty");
    }

    hsa_executable_symbol_t symbol = {0};
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_get_symbol_by_cstring(
            libhsa, executable, symbol_name, any_device_agent, &symbol),
        "looking up HSA symbol for export `%.*s`", (int)symbol_name_view.size,
        symbol_name_view.data);

    const iree_hal_amdgpu_Dims_struct_t flatbuffer_workgroup_size =
        iree_hal_amdgpu_ExportDef_workgroup_size_get(export_def);
    const uint32_t workgroup_size[3] = {
        flatbuffer_workgroup_size->x,
        flatbuffer_workgroup_size->y,
        flatbuffer_workgroup_size->z,
    };
    const uint16_t constant_count =
        (uint16_t)iree_hal_amdgpu_ExportDef_constant_count_get(export_def);
    iree_hal_amdgpu_BindingBits_vec_t binding_bits =
        iree_hal_amdgpu_ExportDef_binding_flags_get(export_def);
    const uint16_t binding_count =
        (uint16_t)iree_hal_amdgpu_BindingBits_vec_len(binding_bits);
    const iree_hal_amdgpu_trace_src_loc_t* export_loc = NULL;
    if (export_locs &&
        iree_hal_amdgpu_trace_src_loc_has_data(&export_locs[kernel_ordinal])) {
      export_loc = &export_locs[kernel_ordinal];
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_resolve_kernel_args_from_symbol(
            libhsa, symbol, workgroup_size, constant_count, binding_count,
            export_loc, &host_kernel_args[kernel_ordinal]),
        "resolving kernel args for `%.*s`", (int)symbol_name_view.size,
        symbol_name_view.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_resolve_raw_hsaco_kernel_args(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    hsa_agent_t any_device_agent,
    iree_hal_amdgpu_device_kernel_args_t* host_kernel_args) {
  for (iree_host_size_t kernel_ordinal = 0;
       kernel_ordinal < hsaco_metadata->kernel_count; ++kernel_ordinal) {
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel =
        &hsaco_metadata->kernels[kernel_ordinal];
    iree_string_view_t symbol_name = kernel->symbol_name;

    hsa_executable_symbol_t symbol = {0};
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_get_raw_hsaco_symbol_by_name(
            libhsa, executable, symbol_name, any_device_agent, &symbol),
        "looking up HSA symbol for raw kernel `%.*s`", (int)symbol_name.size,
        symbol_name.data);

    iree_hal_amdgpu_hsaco_metadata_export_parameter_requirements_t requirements;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_metadata_calculate_default_export_parameter_requirements(
            kernel, &requirements),
        "projecting HSACO parameters for raw kernel `%.*s`",
        (int)symbol_name.size, symbol_name.data);

    uint32_t workgroup_size[3] = {0};
    iree_hal_amdgpu_executable_raw_hsaco_workgroup_size(kernel, workgroup_size);
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_executable_resolve_kernel_args_from_symbol(
            libhsa, symbol, workgroup_size, requirements.constant_count,
            requirements.binding_count, /*export_loc=*/NULL,
            &host_kernel_args[kernel_ordinal]),
        "resolving kernel args for raw kernel `%.*s`", (int)symbol_name.size,
        symbol_name.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_create_from_flatbuffer(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        physical_devices,
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_amdgpu_device_limits_t* limits, hsa_agent_t any_device_agent,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    bool retain_profile_artifacts, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  *out_executable = NULL;

  iree_const_byte_span_t executable_flatbuffer = iree_const_byte_span_empty();
  iree_hal_amdgpu_ExecutableDef_table_t executable_def = 0;
  iree_hal_amdgpu_ExportDef_vec_t export_defs = 0;
  iree_const_byte_span_t code_object_data = iree_const_byte_span_empty();
  iree_host_size_t export_count = 0;
  iree_status_t status = iree_hal_read_executable_flatbuffer_header(
      executable_params->executable_data, /*unsafe_infer_size=*/false,
      iree_hal_amdgpu_ExecutableDef_file_identifier, &executable_flatbuffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_flatbuffer_verify(executable_flatbuffer,
                                                          limits);
  }
  if (iree_status_is_ok(status)) {
    executable_def =
        iree_hal_amdgpu_ExecutableDef_as_root(executable_flatbuffer.data);
    export_defs = iree_hal_amdgpu_ExecutableDef_exports_get(executable_def);
    export_count = iree_hal_amdgpu_ExportDef_vec_len(export_defs);
    iree_hal_amdgpu_ModuleDef_vec_t module_defs =
        iree_hal_amdgpu_ExecutableDef_modules_get(executable_def);
    status = iree_hal_amdgpu_executable_get_single_module_image(
        module_defs, &code_object_data);
  }

  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
        code_object_data, host_allocator, &hsaco_metadata);
  }

  iree_host_size_t export_name_storage_size = 0;
  iree_host_size_t export_parameter_count = 0;
  iree_host_size_t export_parameter_name_storage_size = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_calculate_reflection_storage(
        export_defs, &hsaco_metadata, &export_name_storage_size,
        &export_parameter_count, &export_parameter_name_storage_size);
  }

  iree_hal_amdgpu_executable_t* executable = NULL;
  char* export_name_storage = NULL;
  char* export_parameter_name_storage = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_allocate(
        libhsa, topology, physical_devices, export_count,
        export_name_storage_size, export_parameter_count,
        export_parameter_name_storage_size, host_allocator,
        &export_name_storage, &export_parameter_name_storage, &executable);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_initialize_export_infos(
        export_defs, &hsaco_metadata, executable->export_infos,
        executable->export_parameter_offsets, executable->export_parameters,
        export_name_storage, export_parameter_name_storage);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_profile_metadata_hash_code_object(
        code_object_data, executable->profile_code_object_hash);
  }

  // Publish any embedded source files to the tracing infrastructure.
  if (iree_status_is_ok(status)) {
    iree_hal_debug_publish_source_files(
        iree_hal_amdgpu_ExecutableDef_source_files_get(executable_def));
  }

  // Intern source locations for all exported functions. These will persist for
  // the lifetime of the process and be passed to tooling as if they were in a
  // rodata segment.
  iree_hal_amdgpu_trace_src_loc_t* export_locs = NULL;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_executable_intern_trace_locs(export_defs, &export_locs);
  }
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

  // Load executable and register it with selected GPU agents.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_load_module(
        libhsa, topology, physical_devices, executable_params, code_object_data,
        &executable->handle);
  }

  // Resolve kernel args for each export.
  // These parameters should be the same for all devices as we require all
  // devices have the same ISA. The only thing that will differ is the
  // kernel_object pointer and we handle that per-device during table upload.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_resolve_flatbuffer_kernel_args(
        libhsa, executable->handle, export_defs, export_locs, any_device_agent,
        executable->host_kernel_args);
  }

  // Upload copies of kernel arguments for each device.
  // We reuse the host storage we already allocated to make it possible to
  // memcpy the entire table in one go from host memory.
  for (iree_host_size_t device_ordinal = 0;
       iree_status_is_ok(status) && device_ordinal < executable->device_count;
       ++device_ordinal) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            executable->loaded_physical_device_mask, device_ordinal)) {
      continue;
    }
    status = iree_hal_amdgpu_executable_upload_flatbuffer_kernel_table(
        libhsa, executable->handle, export_defs, executable->host_kernel_args,
        topology->gpu_agents[device_ordinal],
        &executable->device_kernel_args[device_ordinal]);
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_amdgpu_executable_initialize_dispatch_descriptors_for_device(
              executable, device_ordinal);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_register_profile_metadata(
        libhsa, topology, profile_metadata, code_object_data,
        retain_profile_artifacts, executable);
  }

  // Invalidate the kernel object pointer in all host args so that we don't
  // accidentally use it instead of the device-specific one.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_executable_invalidate_host_kernel_objects(executable);
  }

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&hsaco_metadata);

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else if (executable) {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_executable_create_from_raw_hsaco(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        physical_devices,
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_amdgpu_device_limits_t* limits, hsa_agent_t any_device_agent,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    bool retain_profile_artifacts, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  *out_executable = NULL;

  iree_const_byte_span_t code_object_data = executable_params->executable_data;
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {0};
  iree_status_t status = iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      code_object_data, host_allocator, &hsaco_metadata);

  iree_host_size_t export_name_storage_size = 0;
  iree_host_size_t export_parameter_count = 0;
  iree_host_size_t export_parameter_name_storage_size = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_calculate_raw_hsaco_reflection_storage(
        &hsaco_metadata, limits, &export_name_storage_size,
        &export_parameter_count, &export_parameter_name_storage_size);
  }

  iree_hal_amdgpu_executable_t* executable = NULL;
  char* export_name_storage = NULL;
  char* export_parameter_name_storage = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_allocate(
        libhsa, topology, physical_devices, hsaco_metadata.kernel_count,
        export_name_storage_size, export_parameter_count,
        export_parameter_name_storage_size, host_allocator,
        &export_name_storage, &export_parameter_name_storage, &executable);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_initialize_raw_hsaco_export_infos(
        &hsaco_metadata, executable->export_infos,
        executable->export_parameter_offsets, executable->export_parameters,
        export_name_storage, export_parameter_name_storage);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_profile_metadata_hash_code_object(
        code_object_data, executable->profile_code_object_hash);
  }

  // Load executable and register it with selected GPU agents.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_load_module(
        libhsa, topology, physical_devices, executable_params, code_object_data,
        &executable->handle);
  }

  // Resolve kernel args for each export.
  // These parameters should be the same for all devices as we require all
  // devices have the same ISA. The only thing that will differ is the
  // kernel_object pointer and we handle that per-device during table upload.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_resolve_raw_hsaco_kernel_args(
        libhsa, executable->handle, &hsaco_metadata, any_device_agent,
        executable->host_kernel_args);
  }

  // Upload copies of kernel arguments for each device.
  // We reuse the host storage we already allocated to make it possible to
  // memcpy the entire table in one go from host memory.
  for (iree_host_size_t device_ordinal = 0;
       iree_status_is_ok(status) && device_ordinal < executable->device_count;
       ++device_ordinal) {
    if (!iree_hal_amdgpu_physical_device_mask_contains(
            executable->loaded_physical_device_mask, device_ordinal)) {
      continue;
    }
    status = iree_hal_amdgpu_executable_upload_raw_hsaco_kernel_table(
        libhsa, executable->handle, &hsaco_metadata,
        executable->host_kernel_args, topology->gpu_agents[device_ordinal],
        &executable->device_kernel_args[device_ordinal]);
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_amdgpu_executable_initialize_dispatch_descriptors_for_device(
              executable, device_ordinal);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_executable_register_profile_metadata(
        libhsa, topology, profile_metadata, code_object_data,
        retain_profile_artifacts, executable);
  }

  // Invalidate the kernel object pointer in all host args so that we don't
  // accidentally use it instead of the device-specific one.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_executable_invalidate_host_kernel_objects(executable);
  }

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&hsaco_metadata);

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else if (executable) {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_executable_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    bool retain_profile_artifacts, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(profile_metadata);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (IREE_UNLIKELY(topology->gpu_agent_count == 0)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "topology must have at least one GPU device"));
  }

  // Resolve the executable queue affinity to the physical devices that need
  // code-object loads and per-device kernel tables.
  iree_hal_amdgpu_queue_affinity_physical_device_set_t physical_devices;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_executable_select_physical_devices(
              topology, executable_params->queue_affinity, &physical_devices));

  // Pick a selected device to be our template for device queries. All devices
  // in the topology are expected to be the same. This should have been checked
  // earlier but we do it here in case the user is bypassing that code.
  hsa_agent_t any_device_agent =
      topology->gpu_agents[physical_devices.first_physical_device_ordinal];

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

  iree_hal_amdgpu_device_limits_t limits = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_query_device_limits(libhsa, any_device_agent, isa,
                                              &limits));

  iree_status_t status = iree_ok_status();
  if (iree_hal_amdgpu_executable_data_is_wrapped_flatbuffer(
          executable_params->executable_data)) {
    status = iree_hal_amdgpu_executable_create_from_flatbuffer(
        libhsa, topology, &physical_devices, executable_params, &limits,
        any_device_agent, profile_metadata, retain_profile_artifacts,
        host_allocator, out_executable);
  } else {
    status = iree_hal_amdgpu_executable_create_from_raw_hsaco(
        libhsa, topology, &physical_devices, executable_params, &limits,
        any_device_agent, profile_metadata, retain_profile_artifacts,
        host_allocator, out_executable);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_executable_release(*out_executable);
    *out_executable = NULL;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

uint64_t iree_hal_amdgpu_executable_profile_id(
    iree_hal_executable_t* base_executable) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  return executable->profile_id;
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
      iree_hal_amdgpu_hsa_cleanup_assert_success(
          iree_hsa_amd_memory_pool_free_raw(executable->libhsa, kernel_args));
    }
  }

  if (executable->handle.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_executable_destroy_raw(
        executable->libhsa, executable->handle));
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_host(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args) {
  const iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_const_cast(base_executable);
  *out_kernel_args = NULL;

  if (IREE_UNLIKELY(export_ordinal >= executable->kernel_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIu32
                            " out of range; executable has %" PRIhsz " exports",
                            export_ordinal, executable->kernel_count);
  }

  *out_kernel_args = &executable->host_kernel_args[export_ordinal];

  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_device(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args) {
  const iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_const_cast(base_executable);
  *out_kernel_args = NULL;

  if (IREE_UNLIKELY(export_ordinal >= executable->kernel_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIu32
                            " out of range; executable has %" PRIhsz " exports",
                            export_ordinal, executable->kernel_count);
  } else if (IREE_UNLIKELY(device_ordinal >= executable->device_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "device ordinal %" PRIhsz
                            " out of range; executable topology has %" PRIhsz
                            " physical devices",
                            device_ordinal, executable->device_count);
  } else if (IREE_UNLIKELY(!iree_hal_amdgpu_physical_device_mask_contains(
                 executable->loaded_physical_device_mask, device_ordinal))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device ordinal %" PRIhsz
                            " is not in executable queue affinity 0x%" PRIx64,
                            device_ordinal, executable->queue_affinity);
  }

  *out_kernel_args =
      &executable->device_kernel_args[device_ordinal][export_ordinal];

  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_executable_dispatch_descriptor_t** out_descriptor) {
  const iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_const_cast(base_executable);
  *out_descriptor = NULL;

  if (IREE_UNLIKELY(export_ordinal >= executable->kernel_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIu32
                            " out of range; executable has %" PRIhsz " exports",
                            export_ordinal, executable->kernel_count);
  } else if (IREE_UNLIKELY(device_ordinal >= executable->device_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "device ordinal %" PRIhsz
                            " out of range; executable topology has %" PRIhsz
                            " physical devices",
                            device_ordinal, executable->device_count);
  } else if (IREE_UNLIKELY(!iree_hal_amdgpu_physical_device_mask_contains(
                 executable->loaded_physical_device_mask, device_ordinal))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device ordinal %" PRIhsz
                            " is not in executable queue affinity 0x%" PRIx64,
                            device_ordinal, executable->queue_affinity);
  }

  const iree_host_size_t descriptor_ordinal =
      device_ordinal * executable->kernel_count + export_ordinal;
  *out_descriptor = &executable->host_dispatch_descriptors[descriptor_ordinal];
  return iree_ok_status();
}

static iree_host_size_t iree_hal_amdgpu_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  return executable->kernel_count;
}

static iree_status_t iree_hal_amdgpu_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  memset(out_info, 0, sizeof(*out_info));
  if (IREE_UNLIKELY(export_ordinal >= executable->kernel_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIu32
                            " out of range; executable has %" PRIhsz " exports",
                            export_ordinal, executable->kernel_count);
  }
  *out_info = executable->export_infos[export_ordinal];
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  IREE_ASSERT_ARGUMENT(out_parameters || capacity == 0);
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  if (IREE_UNLIKELY(export_ordinal >= executable->kernel_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIu32
                            " out of range; executable has %" PRIhsz " exports",
                            export_ordinal, executable->kernel_count);
  }
  const iree_host_size_t parameter_begin =
      executable->export_parameter_offsets[export_ordinal];
  const iree_host_size_t parameter_end =
      executable->export_parameter_offsets[export_ordinal + 1];
  const iree_host_size_t parameter_count = parameter_end - parameter_begin;
  const iree_host_size_t copy_count = iree_min(capacity, parameter_count);
  if (copy_count > 0) {
    memcpy(out_parameters, &executable->export_parameters[parameter_begin],
           copy_count * sizeof(out_parameters[0]));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  iree_hal_amdgpu_executable_t* executable =
      iree_hal_amdgpu_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->kernel_count; ++i) {
    iree_string_view_t export_name = executable->export_infos[i].name;
    if (iree_string_view_equal(export_name, name)) {
      *out_export_ordinal = (iree_hal_executable_export_ordinal_t)i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "export '%.*s' not found in executable",
                          (int)name.size, name.data);
}

static const iree_hal_executable_vtable_t iree_hal_amdgpu_executable_vtable = {
    .destroy = iree_hal_amdgpu_executable_destroy,
    .export_count = iree_hal_amdgpu_executable_export_count,
    .export_info = iree_hal_amdgpu_executable_export_info,
    .export_parameters = iree_hal_amdgpu_executable_export_parameters,
    .lookup_export_by_name = iree_hal_amdgpu_executable_lookup_export_by_name,
};
