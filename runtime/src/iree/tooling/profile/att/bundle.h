// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_ATT_BUNDLE_H_
#define IREE_TOOLING_PROFILE_ATT_BUNDLE_H_

#include "iree/io/file_contents.h"
#include "iree/tooling/profile/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_profile_att_code_object_t {
  // Producer-local executable identifier that owns this code object.
  uint64_t executable_id;
  // Producer-local code-object marker identifier used in ATT packets.
  uint64_t code_object_id;
  // Borrowed exact HSACO bytes from the mapped profile bundle.
  iree_const_byte_span_t data;
} iree_profile_att_code_object_t;

typedef struct iree_profile_att_code_object_load_t {
  // Session-local physical device ordinal where the code object was loaded.
  uint32_t physical_device_ordinal;
  // Producer-local executable identifier that owns this code object.
  uint64_t executable_id;
  // Producer-local code-object marker identifier used in ATT packets.
  uint64_t code_object_id;
  // AMD loader delta passed to the ROCm trace decoder as load address.
  int64_t load_delta;
  // Loaded code-object memory size reported by the AMD loader.
  uint64_t load_size;
} iree_profile_att_code_object_load_t;

typedef struct iree_profile_att_export_t {
  // Producer-local executable identifier that owns this export.
  uint64_t executable_id;
  // Export ordinal referenced by dispatch and trace records.
  uint32_t export_ordinal;
  // Borrowed export name bytes from the mapped profile bundle.
  iree_string_view_t name;
} iree_profile_att_export_t;

typedef struct iree_profile_att_dispatch_t {
  // Dispatch event record copied from the profile bundle.
  iree_hal_profile_dispatch_event_t record;
  // Physical device ordinal from the containing chunk metadata.
  uint32_t physical_device_ordinal;
  // Queue ordinal from the containing chunk metadata.
  uint32_t queue_ordinal;
} iree_profile_att_dispatch_t;

typedef struct iree_profile_att_trace_t {
  // Executable trace record copied from the profile bundle.
  iree_hal_profile_executable_trace_record_t record;
  // Borrowed raw ATT/SQTT trace bytes from the mapped profile bundle.
  iree_const_byte_span_t data;
} iree_profile_att_trace_t;

typedef struct iree_profile_att_profile_t {
  // Host allocator used for dynamic index arrays.
  iree_allocator_t host_allocator;
  // Mapped profile file contents retaining all borrowed record data.
  iree_io_file_contents_t* file_contents;
  // Dynamic array of embedded code-object images.
  iree_profile_att_code_object_t* code_objects;
  // Number of valid entries in |code_objects|.
  iree_host_size_t code_object_count;
  // Capacity of |code_objects| in entries.
  iree_host_size_t code_object_capacity;
  // Dynamic array of per-device code-object load records.
  iree_profile_att_code_object_load_t* code_object_loads;
  // Number of valid entries in |code_object_loads|.
  iree_host_size_t code_object_load_count;
  // Capacity of |code_object_loads| in entries.
  iree_host_size_t code_object_load_capacity;
  // Dynamic array of executable export names.
  iree_profile_att_export_t* exports;
  // Number of valid entries in |exports|.
  iree_host_size_t export_count;
  // Capacity of |exports| in entries.
  iree_host_size_t export_capacity;
  // Dynamic array of dispatch events.
  iree_profile_att_dispatch_t* dispatches;
  // Number of valid entries in |dispatches|.
  iree_host_size_t dispatch_count;
  // Capacity of |dispatches| in entries.
  iree_host_size_t dispatch_capacity;
  // Dynamic array of executable trace artifacts.
  iree_profile_att_trace_t* traces;
  // Number of valid entries in |traces|.
  iree_host_size_t trace_count;
  // Capacity of |traces| in entries.
  iree_host_size_t trace_capacity;
} iree_profile_att_profile_t;

// Initializes |out_profile| for ATT bundle indexing.
void iree_profile_att_profile_initialize(
    iree_allocator_t host_allocator, iree_profile_att_profile_t* out_profile);

// Releases dynamic arrays and any retained mapped bundle contents.
void iree_profile_att_profile_deinitialize(iree_profile_att_profile_t* profile);

// Processes one profile file record into |profile| when it carries ATT inputs.
iree_status_t iree_profile_att_profile_parse_record(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record);

// Opens |path| and indexes ATT-relevant records into |profile|.
//
// Borrowed code-object, export-name, and trace payload spans remain valid until
// |profile| is deinitialized.
iree_status_t iree_profile_att_profile_parse_file(
    iree_string_view_t path, iree_profile_att_profile_t* profile);

const iree_profile_att_code_object_t* iree_profile_att_profile_find_code_object(
    const iree_profile_att_profile_t* profile, uint64_t executable_id,
    uint64_t code_object_id);

const iree_profile_att_export_t* iree_profile_att_profile_find_export(
    const iree_profile_att_profile_t* profile, uint64_t executable_id,
    uint32_t export_ordinal);

const iree_profile_att_dispatch_t* iree_profile_att_profile_find_dispatch(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_ATT_BUNDLE_H_
