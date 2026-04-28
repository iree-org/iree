// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_builtins.h"

#include "iree/hal/drivers/webgpu/webgpu.h"
#include "iree/hal/drivers/webgpu/webgpu_imports.h"

//===----------------------------------------------------------------------===//
// WGSL shader source
//===----------------------------------------------------------------------===//

// Fills a byte range [offset, offset+length) with a repeating 4-byte pattern.
// Each thread handles one u32 word that overlaps the fill range.
//
// Interior words (fully covered) use a direct write with a pre-rotated pattern
// that accounts for the fill offset's misalignment. This makes the fast path
// work regardless of whether the fill starts at a 4-byte boundary.
//
// Edge words (first/last, partially covered) do read-modify-write with per-byte
// pattern extraction.
//
// Params (uniform, 16 bytes):
//   offset: u32  — byte offset in buffer where fill begins
//   length: u32  — number of bytes to fill
//   pattern: u32 — 4-byte replicated fill pattern
//   _pad: u32
//
// Dispatch: ceil(word_count / 256) workgroups where
//   word_count = ceil((offset + length - align_down(offset, 4)) / 4)
static const char iree_hal_webgpu_builtin_fill_wgsl[] =
    "struct Params { offset: u32, length: u32, pattern: u32, _pad: u32, };\n"
    "@group(0) @binding(0) var<storage, read_write> buffer: array<u32>;\n"
    "@group(0) @binding(1) var<uniform> params: Params;\n"
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
    "  let fill_end = params.offset + params.length;\n"
    "  let wi = (params.offset >> 2u) + gid.x;\n"
    "  let wb = wi << 2u;\n"
    "  if (wb >= fill_end) { return; }\n"
    "  let lo = max(wb, params.offset);\n"
    "  let hi = min(wb + 4u, fill_end);\n"
    // Pre-rotate pattern to account for fill offset misalignment. When the
    // fill starts at byte offset N (not a multiple of 4), each interior word
    // needs the pattern rotated right by (N & 3) bytes. This is computed once
    // and reused for all interior words.
    "  let sh = (params.offset & 3u) * 8u;\n"
    "  let rp = select((params.pattern >> sh) | (params.pattern << (32u - sh))"
    ", params.pattern, sh == 0u);\n"
    "  if (lo == wb && hi == wb + 4u) {\n"
    "    buffer[wi] = rp;\n"
    "    return;\n"
    "  }\n"
    // Edge word: read-modify-write with per-byte pattern extraction.
    "  var v: u32 = buffer[wi];\n"
    "  for (var i = lo; i < hi; i++) {\n"
    "    let pb = (params.pattern >> (((i - params.offset) & 3u) * 8u))"
    " & 0xffu;\n"
    "    let s = (i & 3u) * 8u;\n"
    "    v = (v & ~(0xffu << s)) | (pb << s);\n"
    "  }\n"
    "  buffer[wi] = v;\n"
    "}\n";

// Copies [src_offset, src_offset+length) to [dst_offset, dst_offset+length)
// with arbitrary byte alignment. Each thread handles one u32 word in the
// destination.
//
// Interior words (fully covered) use a barrel-shift reassembly: two adjacent
// source words are read and shifted to produce the correctly-aligned output
// word. When source and destination have the same alignment, this reduces to
// a single-word copy. This gives the fast path for ANY alignment combination,
// not just when both sides are 4-byte aligned.
//
// Edge words (first/last, partially covered) do read-modify-write with
// per-byte extraction.
//
// Params (uniform, 16 bytes):
//   src_offset: u32 — byte offset in source buffer
//   dst_offset: u32 — byte offset in destination buffer
//   length: u32     — number of bytes to copy
//   _pad: u32
//
// Dispatch: ceil(word_count / 256) workgroups
static const char iree_hal_webgpu_builtin_copy_wgsl[] =
    "struct Params { src_offset: u32, dst_offset: u32, length: u32,"
    " _pad: u32, };\n"
    "@group(0) @binding(0) var<storage> src: array<u32>;\n"
    "@group(0) @binding(1) var<storage, read_write> dst: array<u32>;\n"
    "@group(0) @binding(2) var<uniform> params: Params;\n"
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
    "  let dst_end = params.dst_offset + params.length;\n"
    "  let wi = (params.dst_offset >> 2u) + gid.x;\n"
    "  let wb = wi << 2u;\n"
    "  if (wb >= dst_end) { return; }\n"
    "  let lo = max(wb, params.dst_offset);\n"
    "  let hi = min(wb + 4u, dst_end);\n"
    // Interior word: barrel-shift reassembly from two adjacent source words.
    // The 4 source bytes for this destination word always span at most 2
    // source words. When source is word-aligned (sh == 0), one read suffices.
    // Otherwise, read both words and shift/OR to produce the output.
    //
    // Bounds safety: for interior words, all 4 source bytes are within the
    // copy range (src_offset + length), so src[sw] and src[sw + 1] are both
    // within the source buffer allocation.
    "  if (lo == wb && hi == wb + 4u) {\n"
    "    let sb = params.src_offset + (wb - params.dst_offset);\n"
    "    let sw = sb >> 2u;\n"
    "    let sh = (sb & 3u) * 8u;\n"
    "    dst[wi] = select("
    "(src[sw] >> sh) | (src[sw + 1u] << (32u - sh))"
    ", src[sw], sh == 0u);\n"
    "    return;\n"
    "  }\n"
    // Edge word: read-modify-write with per-byte extraction.
    "  var v: u32 = dst[wi];\n"
    "  for (var i = lo; i < hi; i++) {\n"
    "    let so = params.src_offset + (i - params.dst_offset);\n"
    "    let sb = (src[so >> 2u] >> ((so & 3u) * 8u)) & 0xffu;\n"
    "    let s = (i & 3u) * 8u;\n"
    "    v = (v & ~(0xffu << s)) | (sb << s);\n"
    "  }\n"
    "  dst[wi] = v;\n"
    "}\n";

//===----------------------------------------------------------------------===//
// Bind group layout entry (matches bridge protocol)
//===----------------------------------------------------------------------===//

// Wire format for a single bind group layout entry as expected by
// device_create_bind_group_layout. 6 uint32_t values = 24 bytes.
typedef struct iree_hal_webgpu_bind_group_layout_entry_t {
  uint32_t binding;
  uint32_t visibility;
  uint32_t buffer_type;
  uint32_t has_dynamic_offset;
  uint32_t min_binding_size_lo;
  uint32_t min_binding_size_hi;
} iree_hal_webgpu_bind_group_layout_entry_t;

//===----------------------------------------------------------------------===//
// Pipeline creation helpers
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_webgpu_builtins_create_pipeline(
    iree_hal_webgpu_handle_t device_handle, const char* wgsl_source,
    iree_host_size_t wgsl_length, const char* entry_point,
    const iree_hal_webgpu_bind_group_layout_entry_t* layout_entries,
    iree_host_size_t layout_entry_count, iree_hal_webgpu_handle_t* out_pipeline,
    iree_hal_webgpu_handle_t* out_bind_group_layout) {
  // Create the bind group layout.
  iree_hal_webgpu_handle_t bind_group_layout =
      iree_hal_webgpu_import_device_create_bind_group_layout(
          device_handle, (uint32_t)(uintptr_t)layout_entries,
          (uint32_t)layout_entry_count);
  if (!bind_group_layout) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "device.createBindGroupLayout() returned null for builtin shader");
  }

  // Create the pipeline layout from the single bind group layout.
  iree_hal_webgpu_handle_t pipeline_layout =
      iree_hal_webgpu_import_device_create_pipeline_layout(
          device_handle, (uint32_t)(uintptr_t)&bind_group_layout,
          /*layout_count=*/1);
  if (!pipeline_layout) {
    iree_hal_webgpu_import_handle_release(bind_group_layout);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "device.createPipelineLayout() returned null for builtin shader");
  }

  // Create the compute pipeline.
  iree_hal_webgpu_handle_t pipeline =
      iree_hal_webgpu_import_device_create_compute_pipeline(
          device_handle, pipeline_layout, (uint32_t)(uintptr_t)wgsl_source,
          (uint32_t)wgsl_length, (uint32_t)(uintptr_t)entry_point,
          (uint32_t)strlen(entry_point));

  // The pipeline layout handle is consumed by the pipeline — release ours.
  iree_hal_webgpu_import_handle_release(pipeline_layout);

  if (!pipeline) {
    iree_hal_webgpu_import_handle_release(bind_group_layout);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "device.createComputePipeline() returned null for builtin shader");
  }

  *out_pipeline = pipeline;
  *out_bind_group_layout = bind_group_layout;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_builtins_initialize(
    iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_builtins_t* out_builtins) {
  IREE_ASSERT_ARGUMENT(out_builtins);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_builtins, 0, sizeof(*out_builtins));

  // Fill shader: binding 0 = storage(rw), binding 1 = uniform(16 bytes).
  iree_hal_webgpu_bind_group_layout_entry_t fill_entries[2] = {
      {0, IREE_HAL_WEBGPU_SHADER_STAGE_COMPUTE,
       IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_STORAGE, 0, 0, 0},
      {1, IREE_HAL_WEBGPU_SHADER_STAGE_COMPUTE,
       IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_UNIFORM, 0, 16, 0},
  };
  iree_status_t status = iree_hal_webgpu_builtins_create_pipeline(
      device_handle, iree_hal_webgpu_builtin_fill_wgsl,
      strlen(iree_hal_webgpu_builtin_fill_wgsl), "main", fill_entries,
      IREE_ARRAYSIZE(fill_entries), &out_builtins->fill_pipeline,
      &out_builtins->fill_bind_group_layout);

  // Copy shader: binding 0 = storage(ro), binding 1 = storage(rw),
  //              binding 2 = uniform(16 bytes).
  if (iree_status_is_ok(status)) {
    iree_hal_webgpu_bind_group_layout_entry_t copy_entries[3] = {
        {0, IREE_HAL_WEBGPU_SHADER_STAGE_COMPUTE,
         IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, 0, 0, 0},
        {1, IREE_HAL_WEBGPU_SHADER_STAGE_COMPUTE,
         IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_STORAGE, 0, 0, 0},
        {2, IREE_HAL_WEBGPU_SHADER_STAGE_COMPUTE,
         IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_UNIFORM, 0, 16, 0},
    };
    status = iree_hal_webgpu_builtins_create_pipeline(
        device_handle, iree_hal_webgpu_builtin_copy_wgsl,
        strlen(iree_hal_webgpu_builtin_copy_wgsl), "main", copy_entries,
        IREE_ARRAYSIZE(copy_entries), &out_builtins->copy_pipeline,
        &out_builtins->copy_bind_group_layout);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_webgpu_builtins_deinitialize(out_builtins);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_webgpu_builtins_deinitialize(
    iree_hal_webgpu_builtins_t* builtins) {
  iree_hal_webgpu_import_handle_release(builtins->fill_pipeline);
  iree_hal_webgpu_import_handle_release(builtins->fill_bind_group_layout);
  iree_hal_webgpu_import_handle_release(builtins->copy_pipeline);
  iree_hal_webgpu_import_handle_release(builtins->copy_bind_group_layout);
  memset(builtins, 0, sizeof(*builtins));
}
