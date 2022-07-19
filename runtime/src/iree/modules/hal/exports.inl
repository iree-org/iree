// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//         ██     ██  █████  ██████  ███    ██ ██ ███    ██  ██████
//         ██     ██ ██   ██ ██   ██ ████   ██ ██ ████   ██ ██
//         ██  █  ██ ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███
//         ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
//          ███ ███  ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████
//
//===----------------------------------------------------------------------===//
//
// This file will be auto generated from hal.imports.mlir in the future; for
// now it's modified by hand but with strict alphabetical sorting required.
// The order of these functions must be sorted ascending by name in a way
// compatible with iree_string_view_compare.
//
// Users are meant to `#define EXPORT_FN` to be able to access the information.
// #define EXPORT_FN(name, arg_type, ret_type, target_fn)

// clang-format off

EXPORT_FN("allocator.allocate", iree_hal_module_allocator_allocate, riiI, r)
EXPORT_FN("allocator.allocate.initialized", iree_hal_module_allocator_allocate_initialized, riirII, r)
EXPORT_FN("allocator.map.byte_buffer", iree_hal_module_allocator_map_byte_buffer, riiirII, r)

EXPORT_FN("buffer.assert", iree_hal_module_buffer_assert, rrrIii, v)
EXPORT_FN("buffer.length", iree_hal_module_buffer_length, r, I)
EXPORT_FN("buffer.load", iree_hal_module_buffer_load, rIi, i)
EXPORT_FN("buffer.store", iree_hal_module_buffer_store, irIi, v)
EXPORT_FN("buffer.subspan", iree_hal_module_buffer_subspan, rII, r)

EXPORT_FN("buffer_view.assert", iree_hal_module_buffer_view_assert, rriiCID, v)
EXPORT_FN("buffer_view.buffer", iree_hal_module_buffer_view_buffer, r, r)
EXPORT_FN("buffer_view.create", iree_hal_module_buffer_view_create, riiCID, r)
EXPORT_FN("buffer_view.dim", iree_hal_module_buffer_view_dim, ri, I)
EXPORT_FN("buffer_view.element_type", iree_hal_module_buffer_view_element_type, r, i)
EXPORT_FN("buffer_view.encoding_type", iree_hal_module_buffer_view_encoding_type, r, i)
EXPORT_FN("buffer_view.rank", iree_hal_module_buffer_view_rank, r, i)
EXPORT_FN("buffer_view.trace", iree_hal_module_buffer_view_trace, rCrD, v)

EXPORT_FN("command_buffer.begin_debug_group", iree_hal_module_command_buffer_begin_debug_group, rr, v)
EXPORT_FN("command_buffer.bind_descriptor_set", iree_hal_module_command_buffer_bind_descriptor_set, rrirCID, v)
EXPORT_FN("command_buffer.copy_buffer", iree_hal_module_command_buffer_copy_buffer, rrIrII, v)
EXPORT_FN("command_buffer.create", iree_hal_module_command_buffer_create, rii, r)
EXPORT_FN("command_buffer.dispatch", iree_hal_module_command_buffer_dispatch, rriiii, v)
EXPORT_FN("command_buffer.dispatch.indirect", iree_hal_module_command_buffer_dispatch_indirect, rrirI, v)
EXPORT_FN("command_buffer.end_debug_group", iree_hal_module_command_buffer_end_debug_group, r, v)
EXPORT_FN("command_buffer.execution_barrier", iree_hal_module_command_buffer_execution_barrier, riii, v)
EXPORT_FN("command_buffer.fill_buffer", iree_hal_module_command_buffer_fill_buffer, rrIIii, v)
EXPORT_FN("command_buffer.finalize", iree_hal_module_command_buffer_finalize, r, v)
EXPORT_FN("command_buffer.push_constants", iree_hal_module_command_buffer_push_constants, rriCiD, v)
EXPORT_FN("command_buffer.push_descriptor_set", iree_hal_module_command_buffer_push_descriptor_set, rriCirIID, v)

EXPORT_FN("descriptor_set.create", iree_hal_module_descriptor_set_create, rrCirIID, r)

EXPORT_FN("descriptor_set_layout.create", iree_hal_module_descriptor_set_layout_create, riCiiD, r)

EXPORT_FN("device.allocator", iree_hal_module_device_allocator, r, r)
EXPORT_FN("device.query.i64", iree_hal_module_device_query_i64, rrr, iI)
EXPORT_FN("device.queue.alloca", iree_hal_module_device_queue_alloca, rIrriiiI, r)
EXPORT_FN("device.queue.dealloca", iree_hal_module_device_queue_dealloca, rIrrr, v)
EXPORT_FN("device.queue.execute", iree_hal_module_device_queue_execute, rIrrCrD, v)
EXPORT_FN("device.queue.flush", iree_hal_module_device_queue_flush, rI, v)

EXPORT_FN("ex.shared_device", iree_hal_module_ex_shared_device, v, r)
EXPORT_FN("ex.submit_and_wait", iree_hal_module_ex_submit_and_wait, rr, v)

EXPORT_FN("executable.create", iree_hal_module_executable_create, rrrrCrD, r)

EXPORT_FN("executable_layout.create", iree_hal_module_executable_layout_create, riCrD, r)

EXPORT_FN("fence.await", iree_hal_module_fence_await, iCrD, i)
EXPORT_FN("fence.create", iree_hal_module_fence_create, CrID, r)
EXPORT_FN("fence.fail", iree_hal_module_fence_signal, ri, v)
EXPORT_FN("fence.join", iree_hal_module_fence_join, CrD, r)
EXPORT_FN("fence.signal", iree_hal_module_fence_signal, r, v)

EXPORT_FN("semaphore.await", iree_hal_module_semaphore_await, rI, i)
EXPORT_FN("semaphore.create", iree_hal_module_semaphore_create, rI, r)
EXPORT_FN("semaphore.fail", iree_hal_module_semaphore_fail, r, i)
EXPORT_FN("semaphore.query", iree_hal_module_semaphore_query, r, iI)
EXPORT_FN("semaphore.signal", iree_hal_module_semaphore_signal, rI, v)

// clang-format on
