// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_TRACE_REPLAY_H_
#define IREE_TOOLING_TRACE_REPLAY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/tooling/yaml_util.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_trace_replay_t iree_trace_replay_t;

enum iree_trace_replay_flag_bits_e {
  IREE_TRACE_REPLAY_FLAG_NONE = 0u,
  // Prints statistics on replay shutdown.
  IREE_TRACE_REPLAY_FLAG_PRINT_STATISTICS = 1u << 0,
  // Only creates devices once and reuses them for all execution contexts.
  // When omitted new devices will be created each time the HAL module is
  // loaded and existing devices will be destroyed when new contexts are loaded.
  IREE_TRACE_REPLAY_FLAG_REUSE_DEVICES = 1u << 1,
  // Only loads modules once and reuses them for all execution contexts.
  // When omitted modules will be created each time a module load is encountered
  // and destroyed when new contexts are loaded. Context-specific module state
  // is not preserved.
  IREE_TRACE_REPLAY_FLAG_REUSE_MODULES = 1u << 2,
};
typedef uint32_t iree_trace_replay_flags_t;

// Optional set of callbacks around a replay event function call.
// Functions not required by the caller may be omitted.
typedef struct iree_trace_replay_call_hooks_t {
  // User context passed to each callback.
  void* user_data;
  // Issued before the call begins with the call inputs.
  iree_status_t (*before)(void* user_data, iree_trace_replay_t* replay,
                          yaml_document_t* document, yaml_node_t* event_node,
                          iree_vm_function_t function,
                          iree_vm_list_t* input_list);
  // Issued after the call completes successfully with the call outputs.
  iree_status_t (*after)(void* user_data, iree_trace_replay_t* replay,
                         yaml_document_t* document, yaml_node_t* event_node,
                         iree_vm_function_t function,
                         iree_vm_list_t* output_list);
  // Issued only when the call fails and not the replay operation itself.
  // |status| is as returned from the call and ownership is transferred to the
  // hook.
  iree_status_t (*error)(void* user_data, iree_trace_replay_t* replay,
                         yaml_document_t* document, yaml_node_t* event_node,
                         iree_vm_function_t function, iree_status_t status);
} iree_trace_replay_call_hooks_t;

typedef struct iree_trace_replay_t {
  iree_allocator_t host_allocator;
  iree_string_view_t root_path;

  iree_trace_replay_flags_t replay_flags;

  iree_vm_instance_t* instance;
  iree_vm_context_flags_t context_flags;

  // Optional copy of stdin owned by the caller. When non-empty this will be
  // used in place of the real stdin stream and otherwise the real stream will
  // be read on each event. Must remain valid for the lifetime of the replay.
  iree_const_byte_span_t stdin_contents;

  iree_hal_driver_registry_t* driver_registry;
  iree_host_size_t device_uri_count;
  const iree_string_view_t* device_uris;

  // All loaded modules, reset each context load unless
  // IREE_TRACE_REPLAY_FLAG_REUSE_MODULES is set to preserve them.
  // Note that this is only the module itself and not its per-context state.
  iree_host_size_t module_count;
  iree_vm_module_t* modules[16];

  // Context used within the replay, modules registered on-demand.
  iree_vm_context_t* context;

  // Active HAL device if any. Will be initialized on the first HAL module load
  // and reset each context load unless IREE_TRACE_REPLAY_FLAG_REUSE_DEVICES is
  // set to reuse it.
  iree_hal_device_t* device;

  // Optional inputs available via `!input.get`/`!input.take`.
  iree_vm_list_t* inputs;
  // Optional outputs populated via `!output.set`/`!output.push`.
  iree_vm_list_t* outputs;
  // Blackboard used to track state within the trace.
  iree_vm_list_t* blackboard;

  // Optional call hooks allowing reflection of calls and their I/O.
  iree_trace_replay_call_hooks_t call_hooks;
} iree_trace_replay_t;

// Initializes a trace replay context.
// Relative paths will be joined with |root_path| to form a fully-qualified
// path (may be cwd, may be file source, etc).
iree_status_t iree_trace_replay_initialize(
    iree_string_view_t root_path, iree_vm_instance_t* instance,
    iree_trace_replay_flags_t replay_flags,
    iree_vm_context_flags_t context_flags,
    iree_hal_driver_registry_t* driver_registry,
    iree_allocator_t host_allocator, iree_trace_replay_t* out_replay);

// Deinitializes a trace replay context and releases all resources.
void iree_trace_replay_deinitialize(iree_trace_replay_t* replay);

// TODO(#5724): remove this and instead provide a device set on initialize.
// Overrides the HAL driver used in the trace with the given |driver|.
// |device_uris| must remain valid for the lifetime of the replay instance.
void iree_trace_replay_set_hal_devices_override(
    iree_trace_replay_t* replay, iree_host_size_t device_uri_count,
    const iree_string_view_t* device_uris);

// Resets replay input/output/blackboard state.
void iree_trace_replay_reset(iree_trace_replay_t* replay);

// Replays the given |event_node| against the replay context.
// Automatically switches between the default iree_trace_replay_event_* methods.
iree_status_t iree_trace_replay_event(iree_trace_replay_t* replay,
                                      yaml_document_t* document,
                                      yaml_node_t* event_node);

// Replays a `context_load` event against the replay context.
iree_status_t iree_trace_replay_event_context_load(iree_trace_replay_t* replay,
                                                   yaml_document_t* document,
                                                   yaml_node_t* event_node);

// Replays a `module_load` event against the replay context.
iree_status_t iree_trace_replay_event_module_load(iree_trace_replay_t* replay,
                                                  yaml_document_t* document,
                                                  yaml_node_t* event_node);

// Prepares to replay a `call` event.
// |out_function| will contain the function to invoke and |out_input_list| will
// contain the caller-owned inputs to the invocation.
iree_status_t iree_trace_replay_event_call_prepare(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node, iree_vm_function_t* out_function,
    iree_vm_list_t** out_input_list);

// Replays a `call` event against the replay context.
// Optionally |hooks| may be specified to inspect the inputs and outputs of the
// call operation.
iree_status_t iree_trace_replay_event_call(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node, const iree_trace_replay_call_hooks_t* hooks);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_TRACE_REPLAY_H_
