// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Wasm import declarations for the WebGPU bridge.
//
// When compiling for wasm32, these are declared as wasm imports using
// __attribute__((import_module("iree_webgpu"), import_name(...))). The JS
// host provides the implementations via webgpu_imports.mjs.
//
// When compiling for native (unit tests, Dawn-backed development), static
// inline stubs return null handles (0) and are no-ops. This allows code that
// depends on these declarations to compile and link on the host. The actual
// native (Dawn) implementations will be provided by a separate library that
// implements these functions using Dawn's webgpu.h C API.
//
// All values crossing the wasm boundary are integers or pointers into wasm
// linear memory. WebGPU objects are referenced by uint32_t handles (indices
// into the JS-side handle table). Handle 0 is null/invalid.
//
// Async operations (request_adapter, adapter_request_device, buffer_map_async,
// queue_on_submitted_work_done) accept a |token| parameter. When the operation
// completes, JS writes {token, status_code} to the proactor's completion ring.
// The proactor dispatches it to the registered callback on the next poll().

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_IMPORTS_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_IMPORTS_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Device / Adapter
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WASM)

// Requests a GPU adapter. Async: completion delivered via proactor ring with
// |token|. On success (status_code == 0), the adapter handle is written to
// |out_adapter_handle_ptr| in wasm linear memory before the completion is
// posted. |options_flags| is reserved for future use (power preference, etc.).
__attribute__((import_module("iree_webgpu"),
               import_name("request_adapter"))) extern void
iree_hal_webgpu_import_request_adapter(uint32_t options_flags,
                                       uint32_t out_adapter_handle_ptr,
                                       uint32_t token);

// Requests a device from an adapter. Async: completion delivered via proactor
// ring with |token|. On success (status_code == 0), the device handle is
// written to |out_device_handle_ptr| in wasm linear memory before the
// completion is posted.
__attribute__((import_module("iree_webgpu"),
               import_name("adapter_request_device"))) extern void
iree_hal_webgpu_import_adapter_request_device(uint32_t adapter_handle,
                                              uint32_t out_device_handle_ptr,
                                              uint32_t token);

// Destroys a device and releases its handle.
__attribute__((import_module("iree_webgpu"),
               import_name("device_destroy"))) extern void
iree_hal_webgpu_import_device_destroy(uint32_t device_handle);

// Returns the queue handle associated with a device.
// The queue is implicitly created with the device (WebGPU has one queue per
// device). The returned handle is valid as long as the device is alive.
__attribute__((import_module("iree_webgpu"),
               import_name("device_get_queue"))) extern uint32_t
iree_hal_webgpu_import_device_get_queue(uint32_t device_handle);

//===----------------------------------------------------------------------===//
// Buffers
//===----------------------------------------------------------------------===//

// Creates a GPU buffer. Returns the buffer handle.
// |usage| is a GPUBufferUsage bitmask.
// |size| is the buffer size in bytes.
// |mapped_at_creation| is 1 to create in mapped state, 0 otherwise.
__attribute__((import_module("iree_webgpu"),
               import_name("device_create_buffer"))) extern uint32_t
iree_hal_webgpu_import_device_create_buffer(uint32_t device_handle,
                                            uint32_t usage, uint64_t size,
                                            uint32_t mapped_at_creation);

// Destroys a buffer and releases its handle.
__attribute__((import_module("iree_webgpu"),
               import_name("buffer_destroy"))) extern void
iree_hal_webgpu_import_buffer_destroy(uint32_t buffer_handle);

// Initiates an async buffer map operation. Completion delivered via
// proactor ring with |token|.
// |mode|: 1 = MAP_READ, 2 = MAP_WRITE.
__attribute__((import_module("iree_webgpu"),
               import_name("buffer_map_async"))) extern void
iree_hal_webgpu_import_buffer_map_async(uint32_t buffer_handle, uint32_t mode,
                                        uint64_t offset, uint64_t size,
                                        uint32_t token);

// Copies data from a mapped GPU buffer range into wasm linear memory.
// The buffer must be in mapped state. |dest_ptr| is a wasm memory pointer.
__attribute__((import_module("iree_webgpu"),
               import_name("buffer_get_mapped_range"))) extern void
iree_hal_webgpu_import_buffer_get_mapped_range(uint32_t buffer_handle,
                                               uint64_t offset, uint64_t size,
                                               uint32_t dest_ptr);

// Copies data from wasm linear memory into a mapped GPU buffer range.
// The buffer must be in mapped state. |src_ptr| is a wasm memory pointer.
__attribute__((import_module("iree_webgpu"),
               import_name("buffer_set_mapped_range"))) extern void
iree_hal_webgpu_import_buffer_set_mapped_range(uint32_t buffer_handle,
                                               uint64_t offset, uint64_t size,
                                               uint32_t src_ptr);

// Unmaps a previously mapped buffer. The buffer returns to unmapped state.
__attribute__((import_module("iree_webgpu"),
               import_name("buffer_unmap"))) extern void
iree_hal_webgpu_import_buffer_unmap(uint32_t buffer_handle);

//===----------------------------------------------------------------------===//
// Command Encoding
//===----------------------------------------------------------------------===//

// Creates a command encoder for the device. Returns the encoder handle.
__attribute__((import_module("iree_webgpu"),
               import_name("device_create_command_encoder"))) extern uint32_t
iree_hal_webgpu_import_device_create_command_encoder(uint32_t device_handle);

// Begins a compute pass on the encoder. Returns the compute pass handle.
// The pass handle is temporary — valid only until pass_end is called.
__attribute__((import_module("iree_webgpu"),
               import_name("encoder_begin_compute_pass"))) extern uint32_t
iree_hal_webgpu_import_encoder_begin_compute_pass(uint32_t encoder_handle);

// Sets the compute pipeline for the current pass.
__attribute__((import_module("iree_webgpu"),
               import_name("pass_set_pipeline"))) extern void
iree_hal_webgpu_import_pass_set_pipeline(uint32_t pass_handle,
                                         uint32_t pipeline_handle);

// Binds a bind group at the specified index in the current pass.
__attribute__((import_module("iree_webgpu"),
               import_name("pass_set_bind_group"))) extern void
iree_hal_webgpu_import_pass_set_bind_group(uint32_t pass_handle, uint32_t index,
                                           uint32_t bind_group_handle,
                                           uint32_t dynamic_offsets_ptr,
                                           uint32_t dynamic_offsets_count);

// Dispatches compute work.
__attribute__((import_module("iree_webgpu"),
               import_name("pass_dispatch_workgroups"))) extern void
iree_hal_webgpu_import_pass_dispatch_workgroups(uint32_t pass_handle,
                                                uint32_t x, uint32_t y,
                                                uint32_t z);

// Ends the current compute pass.
__attribute__((import_module("iree_webgpu"),
               import_name("pass_end"))) extern void
iree_hal_webgpu_import_pass_end(uint32_t pass_handle);

// Copies data between GPU buffers via the command encoder.
__attribute__((import_module("iree_webgpu"),
               import_name("encoder_copy_buffer_to_buffer"))) extern void
iree_hal_webgpu_import_encoder_copy_buffer_to_buffer(
    uint32_t encoder_handle, uint32_t src_handle, uint64_t src_offset,
    uint32_t dst_handle, uint64_t dst_offset, uint64_t size);

// Finishes recording and returns a command buffer handle.
// The encoder handle is consumed (invalid after this call).
__attribute__((import_module("iree_webgpu"),
               import_name("encoder_finish"))) extern uint32_t
iree_hal_webgpu_import_encoder_finish(uint32_t encoder_handle);

//===----------------------------------------------------------------------===//
// Pipeline / Bind Group
//===----------------------------------------------------------------------===//

// Creates a compute pipeline from WGSL source code.
// |wgsl_ptr|/|wgsl_length|: pointer and length of WGSL source in wasm memory.
// |entry_point_ptr|/|entry_point_length|: pointer and length of the entry
// point name in wasm memory.
// Returns the pipeline handle.
__attribute__((import_module("iree_webgpu"),
               import_name("device_create_compute_pipeline"))) extern uint32_t
iree_hal_webgpu_import_device_create_compute_pipeline(
    uint32_t device_handle, uint32_t layout_handle, uint32_t wgsl_ptr,
    uint32_t wgsl_length, uint32_t entry_point_ptr,
    uint32_t entry_point_length);

// Creates a pipeline layout from bind group layouts.
// |layouts_ptr|: pointer to array of uint32_t bind group layout handles in
// wasm memory.
// |layout_count|: number of bind group layouts.
// Returns the pipeline layout handle.
__attribute__((import_module("iree_webgpu"),
               import_name("device_create_pipeline_layout"))) extern uint32_t
iree_hal_webgpu_import_device_create_pipeline_layout(uint32_t device_handle,
                                                     uint32_t layouts_ptr,
                                                     uint32_t layout_count);

// Creates a bind group layout.
// |entries_ptr|: pointer to array of bind group layout entry descriptors in
// wasm memory. Each entry is:
//   {uint32_t binding, uint32_t visibility, uint32_t buffer_type,
//    uint32_t has_dynamic_offset, uint64_t min_binding_size}
// |entry_count|: number of entries.
// Returns the bind group layout handle.
__attribute__((import_module("iree_webgpu"),
               import_name("device_create_bind_group_layout"))) extern uint32_t
iree_hal_webgpu_import_device_create_bind_group_layout(uint32_t device_handle,
                                                       uint32_t entries_ptr,
                                                       uint32_t entry_count);

// Creates a bind group.
// |entries_ptr|: pointer to array of bind group entry descriptors in wasm
// memory. Each entry is:
//   {uint32_t binding, uint32_t buffer_handle, uint64_t offset, uint64_t size}
// |entry_count|: number of entries.
// Returns the bind group handle.
__attribute__((import_module("iree_webgpu"),
               import_name("device_create_bind_group"))) extern uint32_t
iree_hal_webgpu_import_device_create_bind_group(uint32_t device_handle,
                                                uint32_t layout_handle,
                                                uint32_t entries_ptr,
                                                uint32_t entry_count);

//===----------------------------------------------------------------------===//
// Instruction stream execution
//===----------------------------------------------------------------------===//

// Executes an instruction stream directly from wasm memory. Used for one-shot
// command buffers and scratch builders (queue operations).
//
// |block_table_ptr|: pointer to uint32_t*[] array of block data pointers.
// |block_count|: number of blocks in the table.
// |block_word_capacity|: words per block (uniform across all blocks).
// |last_block_word_count|: valid words in the final block.
// |binding_table_ptr|: pointer to binding table entries in wasm memory.
//   Each entry is {uint32 gpu_buffer_handle, uint32 base_offset}.
// |binding_count|: number of binding table entries.
// |builtins_ptr|: pointer to iree_hal_webgpu_isa_builtins_descriptor_t in wasm
//   memory (4 uint32 handles: fill_pipeline, fill_bgl, copy_pipeline,
//   copy_bgl).
// Returns 0 on success, non-zero on error.
__attribute__((import_module("iree_webgpu"),
               import_name("execute_instructions"))) extern uint32_t
iree_hal_webgpu_import_execute_instructions(
    uint32_t device_handle, uint32_t queue_handle, uint32_t block_table_ptr,
    uint32_t block_count, uint32_t block_word_capacity,
    uint32_t last_block_word_count, uint32_t binding_table_ptr,
    uint32_t binding_count, uint32_t builtins_ptr);

// Creates a recording: copies instruction stream blocks to JS, resolves static
// bindings, caches builtins. Returns a handle to the JS-side recording object.
//
// |block_table_ptr|/|block_count|/|block_word_capacity|/
//   |last_block_word_count|: instruction stream blocks to copy into JS memory.
// |static_binding_table_ptr|: pointer to static binding table entries.
//   Each entry is {uint32 gpu_buffer_handle, uint32 base_offset}.
// |static_binding_count|: number of static entries.
// |dynamic_binding_count|: number of dynamic slots [0, dynamic_binding_count).
// |builtins_ptr|: pointer to builtins descriptor.
// Returns the recording handle, or 0 on failure.
__attribute__((import_module("iree_webgpu"),
               import_name("create_recording"))) extern uint32_t
iree_hal_webgpu_import_create_recording(
    uint32_t device_handle, uint32_t block_table_ptr, uint32_t block_count,
    uint32_t block_word_capacity, uint32_t last_block_word_count,
    uint32_t static_binding_table_ptr, uint32_t static_binding_count,
    uint32_t dynamic_binding_count, uint32_t builtins_ptr);

// Executes a cached recording. Only dynamic bindings are read from wasm memory.
// After execution, dynamic GPUBuffer references are released.
//
// |recording_handle|: handle from create_recording.
// |queue_handle|: WebGPU queue to submit to.
// |dynamic_binding_table_ptr|: pointer to dynamic binding table entries in wasm
//   memory. Each entry is {uint32 gpu_buffer_handle, uint32 base_offset}.
//   Must have exactly dynamic_binding_count entries (set at recording
//   creation).
// Returns 0 on success, non-zero on error.
__attribute__((import_module("iree_webgpu"),
               import_name("execute_recording"))) extern uint32_t
iree_hal_webgpu_import_execute_recording(uint32_t recording_handle,
                                         uint32_t queue_handle,
                                         uint32_t dynamic_binding_table_ptr);

//===----------------------------------------------------------------------===//
// Queue
//===----------------------------------------------------------------------===//

// Submits a command buffer to the queue for execution.
// The command buffer handle is consumed (invalid after this call).
__attribute__((import_module("iree_webgpu"),
               import_name("queue_submit"))) extern void
iree_hal_webgpu_import_queue_submit(uint32_t queue_handle,
                                    uint32_t command_buffer_handle);

// Registers a completion callback for when all currently submitted work on
// the queue finishes. Completion delivered via proactor ring with |token|.
__attribute__((import_module("iree_webgpu"),
               import_name("queue_on_submitted_work_done"))) extern void
iree_hal_webgpu_import_queue_on_submitted_work_done(uint32_t queue_handle,
                                                    uint32_t token);

// Copies data from wasm linear memory to a GPU buffer via the queue.
// This is a convenience function that stages data internally — the source
// memory can be reused immediately after the call returns.
// |data_ptr|: pointer to source data in wasm memory.
// |data_size|: number of bytes to write.
__attribute__((import_module("iree_webgpu"),
               import_name("queue_write_buffer"))) extern void
iree_hal_webgpu_import_queue_write_buffer(uint32_t queue_handle,
                                          uint32_t buffer_handle,
                                          uint64_t buffer_offset,
                                          uint32_t data_ptr,
                                          uint64_t data_size);

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

// Writes adapter info into wasm linear memory at |dest_ptr|.
// The info struct layout is defined by iree_hal_webgpu_adapter_info_t.
__attribute__((import_module("iree_webgpu"),
               import_name("adapter_get_info"))) extern void
iree_hal_webgpu_import_adapter_get_info(uint32_t adapter_handle,
                                        uint32_t dest_ptr);

// Writes device limits into wasm linear memory at |dest_ptr|.
// The limits struct layout is defined by iree_hal_webgpu_device_limits_t.
__attribute__((import_module("iree_webgpu"),
               import_name("device_get_limits"))) extern void
iree_hal_webgpu_import_device_get_limits(uint32_t device_handle,
                                         uint32_t dest_ptr);

//===----------------------------------------------------------------------===//
// File ↔ GPU transfer
//===----------------------------------------------------------------------===//

// Writes data from a file object directly to a GPU buffer via the queue,
// bypassing wasm linear memory entirely. |fd| is a file object table index
// (from the iree_file import module). The JS side reads bytes from the file
// object and calls queue.writeBuffer() with the data.
// This is the zero-copy path for model weight loading.
__attribute__((import_module("iree_webgpu"),
               import_name("queue_write_buffer_from_file"))) extern void
iree_hal_webgpu_import_queue_write_buffer_from_file(
    uint32_t queue_handle, uint32_t buffer_handle, uint64_t buffer_offset,
    uint32_t fd, uint64_t file_offset, uint64_t data_size);

// Returns the byte length of a file object. |fd| is a file object table index.
// Returns 0 if the fd is invalid.
__attribute__((import_module("iree_webgpu"),
               import_name("file_get_length"))) extern uint64_t
iree_hal_webgpu_import_file_get_length(uint32_t fd);

// Copies data from a mapped GPU buffer into an existing file object.
// The buffer must be in mapped state (via buffer_map_async).
// |fd| is the target file object table index. The JS side reads from the
// mapped buffer range and writes into the file object at |file_offset|.
__attribute__((import_module("iree_webgpu"),
               import_name("file_write_from_mapped"))) extern void
iree_hal_webgpu_import_file_write_from_mapped(uint32_t buffer_handle,
                                              uint64_t buffer_offset,
                                              uint64_t size, uint32_t fd,
                                              uint64_t file_offset);

// Exports the mapped range of a GPU buffer as a file object. The buffer must
// be in mapped state (via buffer_map_async). The JS side copies the mapped
// data into a new ArrayBuffer and registers it in the file object table.
// Returns the fd (file object table index), or 0 on failure.
// The returned fd is independent of the GPU buffer — it remains valid after
// the buffer is unmapped.
__attribute__((import_module("iree_webgpu"),
               import_name("buffer_export_mapped_to_file"))) extern uint32_t
iree_hal_webgpu_import_buffer_export_mapped_to_file(uint32_t buffer_handle,
                                                    uint64_t offset,
                                                    uint64_t size);

//===----------------------------------------------------------------------===//
// Resource cleanup
//===----------------------------------------------------------------------===//

// Releases a handle from the JS-side handle table. Use this for objects that
// don't have a specific destroy function (bind groups, bind group layouts,
// pipeline layouts, compute pass encoders after end, shader modules).
__attribute__((import_module("iree_webgpu"),
               import_name("handle_release"))) extern void
iree_hal_webgpu_import_handle_release(uint32_t handle);

//===----------------------------------------------------------------------===//
// Execution context queries
//===----------------------------------------------------------------------===//

// Returns 1 if the current execution context supports blocking waits
// (Atomics.wait), 0 otherwise. Blocking is available on Web Workers with
// cross-origin isolation (SharedArrayBuffer support) but never on the main
// thread where Atomics.wait throws TypeError.
//
// The device queries this once at creation time and uses the result to gate
// HOST_WAIT in semaphore compatibility — callers that check compatibility
// before waiting will never attempt a blocking wait on the main thread.
__attribute__((import_module("iree_webgpu"),
               import_name("can_block"))) extern uint32_t
iree_hal_webgpu_import_can_block(void);

// Returns a non-zero bridge handle for a GPUDevice that the JS host
// pre-configured, or 0 if no device was pre-configured. Used by the driver's
// create_device_by_id to support inline-mode deployments where the host creates
// the GPUDevice before wasm starts (browser main thread, node.js with dawn).
__attribute__((import_module("iree_webgpu"),
               import_name("get_preconfigured_device"))) extern uint32_t
iree_hal_webgpu_import_get_preconfigured_device(void);

#else  // !IREE_PLATFORM_WASM — native stubs

static inline void iree_hal_webgpu_import_request_adapter(
    uint32_t options_flags, uint32_t out_adapter_handle_ptr, uint32_t token) {
  (void)options_flags;
  (void)out_adapter_handle_ptr;
  (void)token;
}
static inline void iree_hal_webgpu_import_adapter_request_device(
    uint32_t adapter_handle, uint32_t out_device_handle_ptr, uint32_t token) {
  (void)adapter_handle;
  (void)out_device_handle_ptr;
  (void)token;
}
static inline void iree_hal_webgpu_import_device_destroy(
    uint32_t device_handle) {
  (void)device_handle;
}
static inline uint32_t iree_hal_webgpu_import_get_preconfigured_device(void) {
  return 0;
}
static inline uint32_t iree_hal_webgpu_import_device_get_queue(
    uint32_t device_handle) {
  (void)device_handle;
  return 0;
}
static inline uint32_t iree_hal_webgpu_import_device_create_buffer(
    uint32_t device_handle, uint32_t usage, uint64_t size,
    uint32_t mapped_at_creation) {
  (void)device_handle;
  (void)usage;
  (void)size;
  (void)mapped_at_creation;
  return 0;
}
static inline void iree_hal_webgpu_import_buffer_destroy(
    uint32_t buffer_handle) {
  (void)buffer_handle;
}
static inline void iree_hal_webgpu_import_buffer_map_async(
    uint32_t buffer_handle, uint32_t mode, uint64_t offset, uint64_t size,
    uint32_t token) {
  (void)buffer_handle;
  (void)mode;
  (void)offset;
  (void)size;
  (void)token;
}
static inline void iree_hal_webgpu_import_buffer_get_mapped_range(
    uint32_t buffer_handle, uint64_t offset, uint64_t size, uint32_t dest_ptr) {
  (void)buffer_handle;
  (void)offset;
  (void)size;
  (void)dest_ptr;
}
static inline void iree_hal_webgpu_import_buffer_set_mapped_range(
    uint32_t buffer_handle, uint64_t offset, uint64_t size, uint32_t src_ptr) {
  (void)buffer_handle;
  (void)offset;
  (void)size;
  (void)src_ptr;
}
static inline void iree_hal_webgpu_import_buffer_unmap(uint32_t buffer_handle) {
  (void)buffer_handle;
}
static inline uint32_t iree_hal_webgpu_import_device_create_command_encoder(
    uint32_t device_handle) {
  (void)device_handle;
  return 0;
}
static inline uint32_t iree_hal_webgpu_import_encoder_begin_compute_pass(
    uint32_t encoder_handle) {
  (void)encoder_handle;
  return 0;
}
static inline void iree_hal_webgpu_import_pass_set_pipeline(
    uint32_t pass_handle, uint32_t pipeline_handle) {
  (void)pass_handle;
  (void)pipeline_handle;
}
static inline void iree_hal_webgpu_import_pass_set_bind_group(
    uint32_t pass_handle, uint32_t index, uint32_t bind_group_handle,
    uint32_t dynamic_offsets_ptr, uint32_t dynamic_offsets_count) {
  (void)pass_handle;
  (void)index;
  (void)bind_group_handle;
  (void)dynamic_offsets_ptr;
  (void)dynamic_offsets_count;
}
static inline void iree_hal_webgpu_import_pass_dispatch_workgroups(
    uint32_t pass_handle, uint32_t x, uint32_t y, uint32_t z) {
  (void)pass_handle;
  (void)x;
  (void)y;
  (void)z;
}
static inline void iree_hal_webgpu_import_pass_end(uint32_t pass_handle) {
  (void)pass_handle;
}
static inline void iree_hal_webgpu_import_encoder_copy_buffer_to_buffer(
    uint32_t encoder_handle, uint32_t src_handle, uint64_t src_offset,
    uint32_t dst_handle, uint64_t dst_offset, uint64_t size) {
  (void)encoder_handle;
  (void)src_handle;
  (void)src_offset;
  (void)dst_handle;
  (void)dst_offset;
  (void)size;
}
static inline uint32_t iree_hal_webgpu_import_encoder_finish(
    uint32_t encoder_handle) {
  (void)encoder_handle;
  return 0;
}
static inline uint32_t iree_hal_webgpu_import_device_create_compute_pipeline(
    uint32_t device_handle, uint32_t layout_handle, uint32_t wgsl_ptr,
    uint32_t wgsl_length, uint32_t entry_point_ptr,
    uint32_t entry_point_length) {
  (void)device_handle;
  (void)layout_handle;
  (void)wgsl_ptr;
  (void)wgsl_length;
  (void)entry_point_ptr;
  (void)entry_point_length;
  return 0;
}
static inline uint32_t iree_hal_webgpu_import_device_create_pipeline_layout(
    uint32_t device_handle, uint32_t layouts_ptr, uint32_t layout_count) {
  (void)device_handle;
  (void)layouts_ptr;
  (void)layout_count;
  return 0;
}
static inline uint32_t iree_hal_webgpu_import_device_create_bind_group_layout(
    uint32_t device_handle, uint32_t entries_ptr, uint32_t entry_count) {
  (void)device_handle;
  (void)entries_ptr;
  (void)entry_count;
  return 0;
}
static inline uint32_t iree_hal_webgpu_import_device_create_bind_group(
    uint32_t device_handle, uint32_t layout_handle, uint32_t entries_ptr,
    uint32_t entry_count) {
  (void)device_handle;
  (void)layout_handle;
  (void)entries_ptr;
  (void)entry_count;
  return 0;
}
static inline uint32_t iree_hal_webgpu_import_execute_instructions(
    uint32_t device_handle, uint32_t queue_handle, uint32_t block_table_ptr,
    uint32_t block_count, uint32_t block_word_capacity,
    uint32_t last_block_word_count, uint32_t binding_table_ptr,
    uint32_t binding_count, uint32_t builtins_ptr) {
  (void)device_handle;
  (void)queue_handle;
  (void)block_table_ptr;
  (void)block_count;
  (void)block_word_capacity;
  (void)last_block_word_count;
  (void)binding_table_ptr;
  (void)binding_count;
  (void)builtins_ptr;
  return 0;
}
static inline uint32_t iree_hal_webgpu_import_create_recording(
    uint32_t device_handle, uint32_t block_table_ptr, uint32_t block_count,
    uint32_t block_word_capacity, uint32_t last_block_word_count,
    uint32_t static_binding_table_ptr, uint32_t static_binding_count,
    uint32_t dynamic_binding_count, uint32_t builtins_ptr) {
  (void)device_handle;
  (void)block_table_ptr;
  (void)block_count;
  (void)block_word_capacity;
  (void)last_block_word_count;
  (void)static_binding_table_ptr;
  (void)static_binding_count;
  (void)dynamic_binding_count;
  (void)builtins_ptr;
  return 0;
}
static inline uint32_t iree_hal_webgpu_import_execute_recording(
    uint32_t recording_handle, uint32_t queue_handle,
    uint32_t dynamic_binding_table_ptr) {
  (void)recording_handle;
  (void)queue_handle;
  (void)dynamic_binding_table_ptr;
  return 0;
}
static inline void iree_hal_webgpu_import_queue_submit(
    uint32_t queue_handle, uint32_t command_buffer_handle) {
  (void)queue_handle;
  (void)command_buffer_handle;
}
static inline void iree_hal_webgpu_import_queue_on_submitted_work_done(
    uint32_t queue_handle, uint32_t token) {
  (void)queue_handle;
  (void)token;
}
static inline void iree_hal_webgpu_import_queue_write_buffer(
    uint32_t queue_handle, uint32_t buffer_handle, uint64_t buffer_offset,
    uint32_t data_ptr, uint64_t data_size) {
  (void)queue_handle;
  (void)buffer_handle;
  (void)buffer_offset;
  (void)data_ptr;
  (void)data_size;
}
static inline void iree_hal_webgpu_import_adapter_get_info(
    uint32_t adapter_handle, uint32_t dest_ptr) {
  (void)adapter_handle;
  (void)dest_ptr;
}
static inline void iree_hal_webgpu_import_device_get_limits(
    uint32_t device_handle, uint32_t dest_ptr) {
  (void)device_handle;
  (void)dest_ptr;
}
static inline void iree_hal_webgpu_import_queue_write_buffer_from_file(
    uint32_t queue_handle, uint32_t buffer_handle, uint64_t buffer_offset,
    uint32_t fd, uint64_t file_offset, uint64_t data_size) {
  (void)queue_handle;
  (void)buffer_handle;
  (void)buffer_offset;
  (void)fd;
  (void)file_offset;
  (void)data_size;
}
static inline uint64_t iree_hal_webgpu_import_file_get_length(uint32_t fd) {
  (void)fd;
  return 0;
}
static inline void iree_hal_webgpu_import_file_write_from_mapped(
    uint32_t buffer_handle, uint64_t buffer_offset, uint64_t size, uint32_t fd,
    uint64_t file_offset) {
  (void)buffer_handle;
  (void)buffer_offset;
  (void)size;
  (void)fd;
  (void)file_offset;
}
static inline uint32_t iree_hal_webgpu_import_buffer_export_mapped_to_file(
    uint32_t buffer_handle, uint64_t offset, uint64_t size) {
  (void)buffer_handle;
  (void)offset;
  (void)size;
  return 0;
}
static inline void iree_hal_webgpu_import_handle_release(uint32_t handle) {
  (void)handle;
}
static inline uint32_t iree_hal_webgpu_import_can_block(void) {
  return 1;  // Native always supports blocking.
}

#endif  // IREE_PLATFORM_WASM

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_IMPORTS_H_
