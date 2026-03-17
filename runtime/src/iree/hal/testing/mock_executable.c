// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/testing/mock_executable.h"

static const iree_hal_executable_vtable_t iree_hal_mock_executable_vtable;

typedef struct iree_hal_mock_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_host_size_t export_count;
} iree_hal_mock_executable_t;

static void iree_hal_mock_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_mock_executable_t* executable =
      (iree_hal_mock_executable_t*)base_executable;
  iree_allocator_free(executable->host_allocator, executable);
}

static iree_host_size_t iree_hal_mock_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_mock_executable_t* executable =
      (iree_hal_mock_executable_t*)base_executable;
  return executable->export_count;
}

static iree_status_t iree_hal_mock_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  iree_hal_mock_executable_t* executable =
      (iree_hal_mock_executable_t*)base_executable;
  if (export_ordinal >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %u >= export count %" PRIhsz,
                            export_ordinal, executable->export_count);
  }
  memset(out_info, 0, sizeof(*out_info));
  out_info->workgroup_size[0] = 1;
  out_info->workgroup_size[1] = 1;
  out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "mock executable does not support name lookup");
}

iree_status_t iree_hal_mock_executable_create(
    iree_host_size_t export_count, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  iree_hal_mock_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable));
  iree_hal_resource_initialize(&iree_hal_mock_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->export_count = export_count;

  *out_executable = (iree_hal_executable_t*)executable;
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t iree_hal_mock_executable_vtable = {
    .destroy = iree_hal_mock_executable_destroy,
    .export_count = iree_hal_mock_executable_export_count,
    .export_info = iree_hal_mock_executable_export_info,
    .export_parameters = iree_hal_mock_executable_export_parameters,
    .lookup_export_by_name = iree_hal_mock_executable_lookup_export_by_name,
};
