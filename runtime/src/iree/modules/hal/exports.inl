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

EXPORT_FN("allocator.allocate", iree_hal_module_allocator_allocate, rIiiI, r)
EXPORT_FN("allocator.import", iree_hal_module_allocator_import, riIiirII, r)

EXPORT_FN("buffer.allocation.discard", iree_hal_module_buffer_allocation_discard, r, i)
EXPORT_FN("buffer.allocation.is_terminal", iree_hal_module_buffer_allocation_is_terminal, r, i)
EXPORT_FN("buffer.allocation.preserve", iree_hal_module_buffer_allocation_preserve, r, v)
EXPORT_FN("buffer.assert", iree_hal_module_buffer_assert, rrrIii, v)
EXPORT_FN("buffer.length", iree_hal_module_buffer_length, r, I)
EXPORT_FN("buffer.load", iree_hal_module_buffer_load, rIi, i)
EXPORT_FN("buffer.store", iree_hal_module_buffer_store, irIi, v)
EXPORT_FN("buffer.subspan", iree_hal_module_buffer_subspan, rII, r)

EXPORT_FN("buffer_view.assert", iree_hal_module_buffer_view_assert, rriiCID, v)
EXPORT_FN("buffer_view.buffer", iree_hal_module_buffer_view_buffer, r, r)
EXPORT_FN("buffer_view.create", iree_hal_module_buffer_view_create, rIIiiCID, r)
EXPORT_FN("buffer_view.dim", iree_hal_module_buffer_view_dim, ri, I)
EXPORT_FN("buffer_view.element_type", iree_hal_module_buffer_view_element_type, r, i)
EXPORT_FN("buffer_view.encoding_type", iree_hal_module_buffer_view_encoding_type, r, i)
EXPORT_FN("buffer_view.rank", iree_hal_module_buffer_view_rank, r, i)
EXPORT_FN("buffer_view.trace", iree_hal_module_buffer_view_trace, rCrD, v)

EXPORT_FN("channel.create", iree_hal_module_channel_create, rIIrrii, r)
EXPORT_FN("channel.rank_and_count", iree_hal_module_channel_rank_and_count, r, ii)
EXPORT_FN("channel.split", iree_hal_module_channel_split, riiI, r)

EXPORT_FN("command_buffer.advise_buffer", iree_hal_module_command_buffer_advise_buffer, rrIIIi, v)
EXPORT_FN("command_buffer.begin_debug_group", iree_hal_module_command_buffer_begin_debug_group, rr, v)
EXPORT_FN("command_buffer.collective", iree_hal_module_command_buffer_collective, rriiiirrIIIII, v)
EXPORT_FN("command_buffer.copy_buffer", iree_hal_module_command_buffer_copy_buffer, riirIrIII, v)
EXPORT_FN("command_buffer.create", iree_hal_module_command_buffer_create, riiIi, r)
EXPORT_FN_CUSTOM("command_buffer.dispatch", iree_hal_module_command_buffer_dispatch, rriiiiICiDCiirIID, v)
EXPORT_FN_CUSTOM("command_buffer.dispatch.indirect", iree_hal_module_command_buffer_dispatch_indirect, rriirIICiDCiirIID, v)
EXPORT_FN("command_buffer.end_debug_group", iree_hal_module_command_buffer_end_debug_group, r, v)
EXPORT_FN("command_buffer.execution_barrier", iree_hal_module_command_buffer_execution_barrier, riiI, v)
EXPORT_FN("command_buffer.fill_buffer", iree_hal_module_command_buffer_fill_buffer, rrIIiIiI, v)
EXPORT_FN("command_buffer.finalize", iree_hal_module_command_buffer_finalize, r, v)
EXPORT_FN("command_buffer.update_buffer", iree_hal_module_command_buffer_update_buffer, rrIrIIiI, v)

EXPORT_FN("device.allocator", iree_hal_module_device_allocator, r, r)
EXPORT_FN("device.query.i64", iree_hal_module_device_query_i64, rrr, iI)
EXPORT_FN("device.queue.alloca", iree_hal_module_device_queue_alloca, rIrrIiiII, r)
EXPORT_FN("device.queue.barrier", iree_hal_module_device_queue_barrier, rIrrI, v)
EXPORT_FN("device.queue.copy", iree_hal_module_device_queue_copy, rIrrrIrIII, v)
EXPORT_FN("device.queue.dealloca", iree_hal_module_device_queue_dealloca, rIrrrI, v)
EXPORT_FN("device.queue.execute", iree_hal_module_device_queue_execute, rIrrrI, v)
EXPORT_FN("device.queue.execute.indirect", iree_hal_module_device_queue_execute_indirect, rIrrrICrIID, v)
EXPORT_FN("device.queue.fill", iree_hal_module_device_queue_fill, rIrrrIIIiI, v)
EXPORT_FN("device.queue.flush", iree_hal_module_device_queue_flush, rI, v)
EXPORT_FN("device.queue.read", iree_hal_module_device_queue_read, rIrrrIrIII, v)
EXPORT_FN("device.queue.update", iree_hal_module_device_queue_update, rIrrrIrIII, v)
EXPORT_FN("device.queue.write", iree_hal_module_device_queue_write, rIrrrIrIII, v)

EXPORT_FN("devices.count", iree_hal_module_devices_count, v, i)
EXPORT_FN("devices.get", iree_hal_module_devices_get, i, r)

EXPORT_FN("ex.file.from_memory", iree_hal_module_ex_file_from_memory, rIirIIi, r)

EXPORT_FN("executable.create", iree_hal_module_executable_create, rIrrr, r)

EXPORT_FN("fence.await", iree_hal_module_fence_await, iICrD, i)
EXPORT_FN("fence.create", iree_hal_module_fence_create, rI, r)
EXPORT_FN("fence.fail", iree_hal_module_fence_fail, ri, v)
EXPORT_FN("fence.join", iree_hal_module_fence_join, ICrD, r)
EXPORT_FN("fence.query", iree_hal_module_fence_query, r, i)
EXPORT_FN("fence.signal", iree_hal_module_fence_signal, r, v)

// clang-format on
