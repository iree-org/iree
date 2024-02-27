// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/testing/benchmark.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/function_io.h"
#include "iree/vm/api.h"

IREE_FLAG(
    int32_t, batch_size, 64,
    "Number of dispatches to perform per command buffer submission.\n"
    "Higher numbers will reduce the effect of submission overheads on the\n"
    "final timings but too high a value may result in hangs.");

IREE_FLAG(string, executable_format, "",
          "Format of the executable file being loaded.");
IREE_FLAG(string, executable_file, "", "Path to the executable file to load.");

IREE_FLAG(int32_t, entry_point, 0, "Entry point ordinal to run.");

IREE_FLAG_LIST(
    string, workgroup_count,
    "`x,y,z` dimensions of the workgroup count defining the number of\n"
    "workgroup invocations that will be run per benchmark iteration.\n"
    "Each occurrence of the flag will run a benchmark with that set of\n"
    "workgroup count values.");

// Total number of executable-level constants we (currently) allow; this is only
// a limitation of how much memory we allocate and we could make this
// dynamically growable.
#define IREE_HAL_MAX_EXECUTABLE_CONSTANT_COUNT 512
// Total number of push constants we (currently) allow any executable to have.
#define IREE_HAL_MAX_PUSH_CONSTANT_COUNT 64
// Maximum number of descriptor sets in an pipeline layout.
#define IREE_HAL_MAX_DESCRIPTOR_SET_COUNT 2
// Total number of bindings we (currently) allow any executable to have.
#define IREE_HAL_MAX_TOTAL_BINDING_COUNT \
  (IREE_HAL_MAX_DESCRIPTOR_SET_COUNT * 32)

// Parsed dispatch parameters from flags.
// Used to construct the dispatch parameters for the benchmark invocation.
struct {
  int32_t set_count;
  struct {
    // For now we only track the binding counts and assume they are all storage
    // buffers. When we support more types we'll need an encoding.
    int32_t binding_count;
  } sets[IREE_HAL_MAX_DESCRIPTOR_SET_COUNT];

  int32_t executable_constant_count;
  union {
    uint32_t ui32;
  } executable_constants[IREE_HAL_MAX_EXECUTABLE_CONSTANT_COUNT];

  int32_t push_constant_count;
  union {
    uint32_t ui32;
  } push_constants[IREE_HAL_MAX_PUSH_CONSTANT_COUNT];

  int32_t binding_count;
  iree_string_view_t binding_specs[IREE_HAL_MAX_TOTAL_BINDING_COUNT];
  char binding_cconv[IREE_HAL_MAX_TOTAL_BINDING_COUNT];
  iree_hal_descriptor_set_layout_binding_t
      binding_layouts[IREE_HAL_MAX_TOTAL_BINDING_COUNT];
} parsed_params = {
    .executable_constant_count = 0,
    .push_constant_count = 0,
    .binding_count = 0,
};

static iree_status_t parse_executable_constant(iree_string_view_t flag_name,
                                               void* storage,
                                               iree_string_view_t value) {
  IREE_ASSERT_LE(parsed_params.executable_constant_count + 1,
                 IREE_ARRAYSIZE(parsed_params.executable_constants),
                 "too many executable constants");
  uint32_t value_ui32 = 0;
  if (!iree_string_view_atoi_uint32(value, &value_ui32)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid executable constant value `%.*s`; expects uint32_t",
        (int)value.size, value.data);
  }
  parsed_params.executable_constants[parsed_params.executable_constant_count++]
      .ui32 = value_ui32;
  return iree_ok_status();
}
static void print_executable_constant(iree_string_view_t flag_name,
                                      void* storage, FILE* file) {
  if (parsed_params.executable_constant_count == 0) {
    fprintf(file, "# --%.*s=[integer value]\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (int32_t i = 0; i < parsed_params.executable_constant_count; ++i) {
    fprintf(file, "--%.*s=%u", (int)flag_name.size, flag_name.data,
            parsed_params.executable_constants[i].ui32);
    if (i < parsed_params.executable_constant_count - 1) {
      fprintf(file, "\n");
    }
  }
}
IREE_FLAG_CALLBACK(parse_executable_constant, print_executable_constant,
                   &parsed_params, executable_constant,
                   "Appends a uint32_t executable constant value.\n");

static iree_status_t parse_push_constant(iree_string_view_t flag_name,
                                         void* storage,
                                         iree_string_view_t value) {
  IREE_ASSERT_LE(parsed_params.push_constant_count + 1,
                 IREE_ARRAYSIZE(parsed_params.push_constants),
                 "too many push constants");
  uint32_t value_ui32 = 0;
  if (!iree_string_view_atoi_uint32(value, &value_ui32)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid push constant value `%.*s`; expects uint32_t", (int)value.size,
        value.data);
  }
  parsed_params.push_constants[parsed_params.push_constant_count++].ui32 =
      value_ui32;
  return iree_ok_status();
}
static void print_push_constant(iree_string_view_t flag_name, void* storage,
                                FILE* file) {
  if (parsed_params.push_constant_count == 0) {
    fprintf(file, "# --%.*s=[integer value]\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (int32_t i = 0; i < parsed_params.push_constant_count; ++i) {
    fprintf(file, "--%.*s=%u", (int)flag_name.size, flag_name.data,
            parsed_params.push_constants[i].ui32);
    if (i < parsed_params.push_constant_count - 1) {
      fprintf(file, "\n");
    }
  }
}
IREE_FLAG_CALLBACK(parse_push_constant, print_push_constant, &parsed_params,
                   push_constant, "Appends a uint32_t push constant value.\n");

static iree_status_t parse_binding(iree_string_view_t flag_name, void* storage,
                                   iree_string_view_t value) {
  IREE_ASSERT_LE(parsed_params.binding_count + 1,
                 IREE_ARRAYSIZE(parsed_params.binding_specs),
                 "too many bindings");
  int32_t i = parsed_params.binding_count++;
  parsed_params.binding_specs[i] = value;
  parsed_params.binding_cconv[i] = 'r';
  // TODO(benvanik): allow for a specification of type/immutability.
  parsed_params.binding_layouts[i] = (iree_hal_descriptor_set_layout_binding_t){
      .binding = (uint32_t)i,
      .type = IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .flags = IREE_HAL_DESCRIPTOR_FLAG_NONE,
  };
  return iree_ok_status();
}
static void print_binding(iree_string_view_t flag_name, void* storage,
                          FILE* file) {
  if (parsed_params.binding_count == 0) {
    fprintf(file, "# --%.*s=\"shapextype[=values]\"\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (int32_t i = 0; i < parsed_params.binding_count; ++i) {
    const iree_string_view_t binding_spec = parsed_params.binding_specs[i];
    fprintf(file, "--%.*s=\"%.*s\"\n", (int)flag_name.size, flag_name.data,
            (int)binding_spec.size, binding_spec.data);
  }
}
IREE_FLAG_CALLBACK(
    parse_binding, print_binding, &parsed_params, binding,
    "Appends a binding to the dispatch parameters.\n"
    "Bindings are defined by their shape, element type, and their data.\n"
    "There must be one binding for every declared layout binding.\n"
    "Examples:\n"
    "  # 16 4-byte elements zero-initialized:\n"
    "  --binding=2x8xi32\n"
    "  # 10000 bytes all initialized to 123:\n"
    "  --binding=10000xi8=123\n"
    "  # 2 4-byte floating-point values with contents [[1.4], [2.1]]:\n"
    "  --binding=2x1xf32=1.4,2.1\n"
    "  # First array from a numpy file followed by the second:\n"
    "  --binding=@file.npy\n"
    "  --binding=+file.npy\n"
    "  # All arrays from a numpy file\n"
    "  --binding=*file.npy\n"
    "  # Binary tensor<2x2xf32> and tensor<4xf32> read from a single file\n"
    "  --binding=2x2xf32=@file.ext\n"
    "  --binding=4xf32=+file.ext");

typedef struct iree_benchmark_executable_args_t {
  iree_hal_device_t* device;
  iree_hal_executable_t* executable;
  iree_hal_pipeline_layout_t* pipeline_layout;
  const iree_hal_descriptor_set_binding_t* bindings;
  uint32_t workgroup_count[3];
} iree_benchmark_executable_args_t;

// NOTE: error handling is here just for better diagnostics: it is not tracking
// allocations correctly and will leak. Don't use this as an example for how to
// write robust code.
static iree_status_t iree_benchmark_executable_run(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_benchmark_executable_args_t* args =
      (iree_benchmark_executable_args_t*)benchmark_def->user_data;

  iree_hal_semaphore_t* fence_semaphore = NULL;
  uint64_t fence_value = 0ull;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_create(args->device, fence_value, &fence_semaphore));
  iree_hal_semaphore_list_t wait_semaphore_list =
      iree_hal_semaphore_list_empty();
  iree_hal_semaphore_list_t signal_semaphore_list = {
      .count = 1,
      .semaphores = &fence_semaphore,
      .payload_values = &fence_value,
  };

  // Start profiling now - all subsequent device operations will be what the
  // user wants to measure.
  IREE_RETURN_IF_ERROR(iree_hal_begin_profiling_from_flags(args->device));

  // Submit the command buffer and wait for it to complete.
  // Note that each iteration runs through the whole grid as it's important that
  // we are testing the memory access patterns: if we just ran the same single
  // workgroup processing the same exact region of memory over and over we are
  // not testing cache effects. This means we need to account for the total
  // number of workgroups executed.
  int64_t dispatch_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, FLAG_batch_size)) {
    // TODO(benvanik): record a secondary command buffer and just replay it
    // here. This should fix the overhead at just primary command buffer
    // creation. Most backends don't support reusable command buffers, yet, and
    // some only support inline execution so we are conservatively doing that.
    // In the future we should have an option (possibly based on device query)
    // as to which path to use.

    // Record a command buffer with the dispatches.
    // Note that today we are doing this inside of the benchmark loop so that
    // we can use inline execution. This is a boost to devices that support it
    // like CUDA streams and synchronous CPU executors but a pessimization to
    // devices that benefit from reusable command buffers like CUDA graphs.
    // In the future we can add a flag that switches the mode between
    // reusable and one-shot.
    iree_hal_command_buffer_t* command_buffer = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        args->device,
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
            IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, &command_buffer));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_push_constants(
        command_buffer, args->pipeline_layout, /*offset=*/0,
        &parsed_params.push_constants[0].ui32,
        parsed_params.push_constant_count *
            sizeof(parsed_params.push_constants[0])));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_push_descriptor_set(
        command_buffer, args->pipeline_layout, /*set=*/0,
        parsed_params.binding_count, args->bindings));
    for (int32_t i = 0; i < FLAG_batch_size; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch(
          command_buffer, args->executable, FLAG_entry_point,
          args->workgroup_count[0], args->workgroup_count[1],
          args->workgroup_count[2]));
      IREE_RETURN_IF_ERROR(iree_hal_command_buffer_execution_barrier(
          command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
          IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
          IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, NULL, 0, NULL));
    }
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer));

    // Submit the command buffer; if the device could not start executing while
    // we were recording then this will kick off the execution.
    ++fence_value;
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
        args->device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
        signal_semaphore_list, 1, &command_buffer));

    // Block and wait for the submission to complete.
    // Note that this will include round-trip overhead and if the dispatch or
    // batch size is small then the final time may end up being mostly overhead.
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(fence_semaphore, fence_value,
                                                 iree_infinite_timeout()));

    iree_benchmark_pause_timing(benchmark_state);

    // Don't count cleanup time in the benchmark.
    iree_hal_command_buffer_release(command_buffer);

    // Accumulate the total number of dispatches executed.
    dispatch_count += FLAG_batch_size;

    // Flush profiling if recording. Note that we don't want to include the
    // profiling time in the benchmark result.
    IREE_RETURN_IF_ERROR(iree_hal_device_profiling_flush(args->device));

    iree_benchmark_resume_timing(benchmark_state);
  }

  // End profiling before cleaning up so tooling doesn't capture it.
  IREE_RETURN_IF_ERROR(iree_hal_end_profiling_from_flags(args->device));

  // To get a total time per invocation we set the item count to the total
  // invocations dispatched. That gives us both total dispatch and single
  // invocation times in the reporter output.
  int64_t total_invocations = dispatch_count * args->workgroup_count[0] *
                              args->workgroup_count[1] *
                              args->workgroup_count[2];
  iree_benchmark_set_items_processed(benchmark_state, total_invocations);

  iree_hal_semaphore_release(fence_semaphore);

  return iree_ok_status();
}

// Parses an `x,y,z` workgroup count.
static iree_status_t iree_parse_workgroup_count(
    iree_string_view_t workgroup_count_str, uint32_t* out_workgroup_count) {
  iree_string_view_t str = workgroup_count_str;
  iree_string_view_t str_x;
  iree_string_view_split(str, ',', &str_x, &str);
  iree_string_view_t str_y;
  iree_string_view_split(str, ',', &str_y, &str);
  iree_string_view_t str_z = str;
  if (!iree_string_view_atoi_uint32(str_x, &out_workgroup_count[0]) ||
      !iree_string_view_atoi_uint32(str_y, &out_workgroup_count[1]) ||
      !iree_string_view_atoi_uint32(str_z, &out_workgroup_count[2])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid workgroup count string `%.*s`; expects `X,Y,Z`",
        (int)workgroup_count_str.size, workgroup_count_str.data);
  }
  return iree_ok_status();
}

// Runs one benchmark per workgroup count specified using the same device
// and input/output buffers.
static iree_status_t iree_benchmark_executable_from_flags(
    iree_allocator_t host_allocator) {
  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                               host_allocator, &instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_inline_types(instance));

  // Create the HAL device we'll be using during execution.
  // Devices can be very expensive to create and we want to avoid doing it
  // multiple times throughout the benchmark execution.
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_create_device_from_flags(
      iree_hal_available_driver_registry(), iree_hal_default_device_uri(),
      host_allocator, &device));

  // We'll reuse the same executable cache so that once we load the executable
  // we'll be able to reuse any driver-side optimizations.
  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_status_t loop_status = iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_hal_executable_cache_create(
      device, iree_make_cstring_view("cache"), iree_loop_inline(&loop_status),
      &executable_cache));
  IREE_RETURN_IF_ERROR(loop_status);

  // Allocate storage for buffers and populate them.
  // They only need to remain valid for the duration of the invocation and all
  // memory accessed by the invocation will come from here.
  // Note that we do this parsing first so that we can reflect on the I/O to
  // infer the pipeline layout.
  iree_hal_allocator_t* device_allocator = iree_hal_device_allocator(device);
  iree_vm_list_t* binding_list = NULL;
  IREE_RETURN_IF_ERROR(iree_tooling_parse_variants(
      iree_make_string_view(parsed_params.binding_cconv,
                            parsed_params.binding_count),
      (iree_string_view_list_t){parsed_params.binding_count,
                                parsed_params.binding_specs},
      device, device_allocator, host_allocator, &binding_list));
  iree_hal_descriptor_set_binding_t bindings[IREE_HAL_MAX_TOTAL_BINDING_COUNT];
  for (iree_host_size_t i = 0; i < parsed_params.binding_count; ++i) {
    iree_vm_ref_t value = iree_vm_ref_null();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(binding_list, i, &value));
    iree_hal_buffer_t* buffer = NULL;
    if (iree_hal_buffer_isa(value)) {
      buffer = iree_hal_buffer_deref(value);
    } else if (iree_hal_buffer_view_isa(value)) {
      buffer = iree_hal_buffer_view_buffer(iree_hal_buffer_view_deref(value));
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "bindings must be shaped types (4xf32, etc), binding %" PRIhsz
          " is not",
          i);
    }
    bindings[i] = (iree_hal_descriptor_set_binding_t){
        .binding = i,
        .buffer_slot = 0,
        .buffer = buffer,
        .offset = 0,
        .length = IREE_WHOLE_BUFFER,
    };
  }

  // Setup the specification used to perform the executable load.
  // This information is normally used to select the appropriate loader but in
  // this benchmark we only have a single one.
  // TODO(benvanik): expose the flags once they are implemented anywhere.
  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION |
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;

  // Load the executable data into memory.
  // In normal usage this would be mapped from the containing module file (which
  // itself may be mapped from disk).
  iree_file_contents_t* file_contents = NULL;
  if (strcmp(FLAG_executable_file, "-") == 0) {
    IREE_RETURN_IF_ERROR(
        iree_stdin_read_contents(host_allocator, &file_contents));
  } else {
    IREE_RETURN_IF_ERROR(iree_file_read_contents(
        FLAG_executable_file, IREE_FILE_READ_FLAG_DEFAULT, host_allocator,
        &file_contents));
  }
  executable_params.executable_format =
      iree_make_cstring_view(FLAG_executable_format);
  executable_params.executable_data = file_contents->const_buffer;

  // Setup the layouts defining how each entry point is interpreted.
  iree_hal_pipeline_layout_t* pipeline_layout = NULL;
  iree_hal_descriptor_set_layout_t* descriptor_set_layout = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_descriptor_set_layout_create(
      device, IREE_HAL_DESCRIPTOR_SET_LAYOUT_FLAG_NONE,
      parsed_params.binding_count, parsed_params.binding_layouts,
      &descriptor_set_layout));
  IREE_RETURN_IF_ERROR(iree_hal_pipeline_layout_create(
      device, parsed_params.push_constant_count,
      /*set_layout_count=*/1, &descriptor_set_layout, &pipeline_layout));
  executable_params.pipeline_layout_count = 1;
  executable_params.pipeline_layouts = &pipeline_layout;

  // Executable-level constants allow us to perform some basic load-time value
  // propagation - usually dependent on device features or tuning parameters.
  executable_params.constant_count = parsed_params.executable_constant_count;
  executable_params.constants = &parsed_params.executable_constants[0].ui32;

  // Perform the load, which will fail if the executable cannot be loaded or
  // there was an issue with the layouts.
  iree_hal_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, &executable));

  // Register one benchmark per workgroup count specified.
  iree_benchmark_executable_args_t* args = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*args) * FLAG_workgroup_count_list().count,
      (void**)&args));
  for (iree_host_size_t i = 0; i < FLAG_workgroup_count_list().count; ++i) {
    args[i] = (iree_benchmark_executable_args_t){
        .device = device,
        .executable = executable,
        .pipeline_layout = pipeline_layout,
        .bindings = bindings,
        .workgroup_count = {1, 1, 1},
    };
    IREE_RETURN_IF_ERROR(iree_parse_workgroup_count(
        FLAG_workgroup_count_list().values[i], args[i].workgroup_count));
    iree_benchmark_def_t benchmark_def = {
        .flags = IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME |
                 IREE_BENCHMARK_FLAG_USE_REAL_TIME,
        .time_unit = IREE_BENCHMARK_UNIT_NANOSECOND,
        .minimum_duration_ns = 0,
        .iteration_count = 0,
        .run = iree_benchmark_executable_run,
        .user_data = &args[i],
    };
    char benchmark_name[512];
    snprintf(benchmark_name, sizeof(benchmark_name) - 1, "dispatch_%ux%ux%u",
             args[i].workgroup_count[0], args[i].workgroup_count[1],
             args[i].workgroup_count[2]);
    iree_benchmark_register(iree_make_cstring_view(benchmark_name),
                            &benchmark_def);
  }
  iree_benchmark_run_specified();
  iree_allocator_free(host_allocator, args);

  iree_vm_list_release(binding_list);
  iree_hal_executable_release(executable);
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
  iree_hal_pipeline_layout_release(pipeline_layout);
  iree_file_contents_free(file_contents);
  iree_hal_executable_cache_release(executable_cache);
  iree_hal_device_release(device);
  iree_vm_instance_release(instance);

  return iree_ok_status();
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  iree_flags_set_usage(
      "iree-benchmark-executable",
      "Benchmarks a single entry point within an executable library.\n"
      "The parameters used can be inferred from the entry point "
      "`hal.interface` and dispatches to it in the source program.\n"
      "\n"
      "Executables can be extracted from VMFB files using `unzip` or dumped\n"
      "during compilation using --iree-hal-dump-executable-binaries-to=path/.\n"
      "\n"
      "The compiler can directly compile `hal.executable.source` and\n"
      "`hal.executable` ops to the appropriate binaries by using the\n"
      "`iree-compile --compile-mode=hal-executable` mode.\n"
      "\n"
      "Example flags for various compilation backends:\n"
      "  --iree-hal-target-backends=vmvx\n"
      "    --device=local-sync or --device=local-task\n"
      "    --executable_format=vmvx-bytecode-fb\n"
      "  --iree-hal-target-backends=llvm-cpu\n"
      "    --device=local-sync or --device=local-task\n"
      "    --executable_format=embedded-elf-x86_64\n"
      "    --executable_format=system-dll-x86_64\n"
      "  --iree-hal-target-backends=cuda\n"
      "    --device=cuda\n"
      "    --executable_format=cuda-nvptx-fb\n"
      "  --iree-hal-target-backends=vulkan-spirv\n"
      "    --device=vulkan\n"
      "    --executable_format=vulkan-spirv-fb\n"
      "\n"
      "Note that this tool is intentionally low level: you must specify all\n"
      "of the push constant/binding parameters precisely as they are expected\n"
      "by the executable. `iree-benchmark-module` is the user-friendly\n"
      "benchmarking tool while this one favors direct access to the\n"
      "executables (bypassing all of the IREE VM, HAL APIs, task system,\n"
      "etc).\n"
      "\n"
      "Example --flagfile:\n"
      "  --device=local-sync\n"
      "  --executable_format=embedded-elf-x86_64\n"
      "  --executable_file=runtime/src/iree/hal/local/elf/testdata/"
      "elementwise_mul_x86_64.so\n"
      "  --entry_point=0\n"
      "  --binding=4xf32=1,2,3,4\n"
      "  --binding=4xf32=100,200,300,400\n"
      "  --binding=4xf32=0,0,0,0\n"
      "  --workgroup_count=1,1,1\n"
      "\n");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_benchmark_initialize(&argc, argv);

  iree_status_t status = iree_benchmark_executable_from_flags(host_allocator);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
