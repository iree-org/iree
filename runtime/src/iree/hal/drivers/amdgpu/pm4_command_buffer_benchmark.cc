// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/command_buffer.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/aql_program_builder.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"
#include "iree/hal/drivers/amdgpu/util/benchmark_flags.h"
#include "iree/hal/drivers/amdgpu/util/benchmark_profile.h"
#include "runtime/src/iree/hal/drivers/amdgpu/testdata_amdgpu_pm4_command_buffer_benchmark.h"

IREE_FLAG(
    bool, pm4_collect_finalize_timings, false,
    "Enables a metadata-only HAL profiling session while recording PM4 "
    "command buffers so PM4 finalize phase timing counters are populated. "
    "Default benchmark runs leave this off to avoid iree_time_now() overhead "
    "in the recorder.");
IREE_FLAG(
    string, pm4_publication_mode, "host-copy",
    "PM4 command-buffer resident publication mode: 'host-copy' writes host "
    "staging builders and publishes populated segments with hsa_memory_copy; "
    "'host-async-copy' publishes one contiguous staging image with "
    "hsa_amd_memory_async_copy and waits in end(); "
    "'host-async-copy-nonblocking' publishes the same staging image and makes "
    "queue execution wait on publication completion.");
IREE_FLAG(
    bool, command_buffer_unretained, false,
    "Records command buffers with IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED "
    "to measure recorder cost when the caller owns resource lifetimes.");
IREE_FLAG(bool, command_buffer_unvalidated, false,
          "Records command buffers with "
          "IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED to measure recorder cost "
          "with cold validation removed.");

namespace {

constexpr iree_host_size_t kFrontierAxisTableCapacity = 256;
constexpr iree_host_size_t kMaximumBindingTableCount = 100;
constexpr iree_device_size_t kPayloadBufferLength = 16;
constexpr iree_device_size_t kPayloadBufferAlignment = 16;
constexpr uint32_t kDefaultUploadCapacity = 64 * 1024;

enum class CommandBufferPath : int64_t {
  kAqlDynamic = 0,
  kAqlStatic = 1,
  kPm4Static = 2,
  kPm4Fixup = 3,
};

struct BenchmarkSpec {
  // Number of dispatch commands recorded in the command buffer.
  int64_t operation_count = 0;
  // Number of entries published in dynamic queue_execute binding tables.
  int64_t binding_table_count = 0;
  // Approximate percent of dispatches placed in no-barrier spans.
  int64_t overlap_percent = 0;
};

struct SubmittedCompletion {
  // Semaphore whose payload value marks iteration completion.
  iree_hal_semaphore_t* semaphore = nullptr;
  // Payload value to wait for on |semaphore|.
  uint64_t payload_value = 0;
};

bool HandleStatus(benchmark::State& state, iree_status_t status,
                  const char* message) {
  if (iree_status_is_ok(status)) return true;
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  state.SkipWithError(message);
  return false;
}

iree_const_byte_span_t FindExecutableData(iree_string_view_t file_name) {
  const iree_file_toc_t* toc =
      iree_pm4_command_buffer_benchmark_testdata_amdgpu_create();
  for (iree_host_size_t i = 0; toc[i].name != nullptr; ++i) {
    if (iree_string_view_equal(file_name,
                               iree_make_cstring_view(toc[i].name))) {
      return iree_make_const_byte_span(
          reinterpret_cast<const uint8_t*>(toc[i].data), toc[i].size);
    }
  }
  return iree_const_byte_span_empty();
}

bool IsDynamicPath(CommandBufferPath path) {
  return path == CommandBufferPath::kAqlDynamic ||
         path == CommandBufferPath::kPm4Fixup;
}

bool IsPm4Path(CommandBufferPath path) {
  return path == CommandBufferPath::kPm4Static ||
         path == CommandBufferPath::kPm4Fixup;
}

int64_t NextDispatchGroupSize(int64_t dispatch_index, int64_t operation_count,
                              int64_t overlap_percent) {
  if (overlap_percent == 0) return 1;
  if (overlap_percent != 20) return 1;
  const int64_t pattern_position = dispatch_index % 25;
  int64_t group_size = 1;
  if (pattern_position == 0) {
    group_size = 2;
  } else if (pattern_position == 10) {
    group_size = 3;
  }
  return std::min<int64_t>(group_size, operation_count - dispatch_index);
}

iree_status_t EmitDispatchBarrier(iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, /*memory_barrier_count=*/0,
      /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
}

struct DeviceBundle {
  // HAL device configured for one command-buffer path family.
  iree_hal_device_t* device = nullptr;
  // Device group assigning topology/frontier metadata to |device|.
  iree_hal_device_group_t* device_group = nullptr;
  // Executable cache owning the benchmark executable.
  iree_hal_executable_cache_t* executable_cache = nullptr;
  // Two-entrypoint benchmark executable with identical binding layout.
  iree_hal_executable_t* executable = nullptr;
  // Device-local buffers used as static refs and dynamic binding-table values.
  std::array<iree_hal_buffer_t*, kMaximumBindingTableCount> buffers = {};
  // Host-side binding table passed to dynamic queue_execute paths.
  std::array<iree_hal_buffer_binding_t, kMaximumBindingTableCount>
      binding_table = {};
  // Export ordinal for the first benchmark entrypoint.
  iree_hal_executable_export_ordinal_t model_a = 0;
  // Export ordinal for the second benchmark entrypoint.
  iree_hal_executable_export_ordinal_t model_b = 0;
  // Completion semaphore reused across iterations for this device.
  iree_hal_semaphore_t* completion_semaphore = nullptr;
  // Next completion payload value for |completion_semaphore|.
  uint64_t completion_payload_value = 0;

  void Release() {
    iree_hal_semaphore_release(completion_semaphore);
    completion_semaphore = nullptr;
    for (iree_hal_buffer_t*& buffer : buffers) {
      iree_hal_buffer_release(buffer);
      buffer = nullptr;
    }
    iree_hal_executable_release(executable);
    executable = nullptr;
    iree_hal_executable_cache_release(executable_cache);
    executable_cache = nullptr;
    iree_hal_device_release(device);
    device = nullptr;
    iree_hal_device_group_release(device_group);
    device_group = nullptr;
    completion_payload_value = 0;
  }
};

class Pm4CommandBufferBenchmark : public benchmark::Fixture {
 public:
  static void InitializeOnce() {
    if (initialized_) return;
    initialized_ = true;

    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa_,
                                                                 &topology_);
    }
    if (iree_status_is_ok(status) && topology_.gpu_agent_count == 0) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "no AMDGPU HSA GPU agents are visible");
    }
    if (iree_status_is_ok(status)) {
      status = iree_async_proactor_pool_create(
          iree_numa_node_count(), /*node_ids=*/nullptr,
          iree_async_proactor_pool_options_default(), host_allocator_,
          &proactor_pool_);
    }
    if (iree_status_is_ok(status)) {
      status = CreateDevice(IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL,
                            /*upload_capacity=*/0, &aql_);
    }
    if (iree_status_is_ok(status)) {
      status = CreateDevice(IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4,
                            kDefaultUploadCapacity, &pm4_);
    }
    if (iree_status_is_ok(status)) {
      status = BeginPm4FinalizeTimingProfile();
    }
    if (iree_status_is_ok(status)) {
      available_ = true;
      return;
    }

    iree_status_fprint(stderr, status);
    iree_status_free(status);
    DeinitializeOnce();
  }

  static void DeinitializeOnce() {
    EndPm4FinalizeTimingProfile();
    pm4_.Release();
    aql_.Release();
    iree_async_proactor_pool_release(proactor_pool_);
    proactor_pool_ = nullptr;
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
    available_ = false;
  }

  void SetUp(benchmark::State& state) override {
    InitializeOnce();
    if (!available_) {
      state.SkipWithError("AMDGPU benchmark device is unavailable");
    }
  }

 protected:
  static iree_status_t CreateDevice(
      iree_hal_amdgpu_command_buffer_mode_t command_buffer_mode,
      uint32_t upload_capacity, DeviceBundle* out_bundle) {
    iree_hal_amdgpu_logical_device_options_t options;
    iree_hal_amdgpu_logical_device_options_initialize(&options);
    options.command_buffer_mode = command_buffer_mode;
    options.host_queues.upload_capacity = upload_capacity;
    if (command_buffer_mode == IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4) {
      if (strcmp(FLAG_pm4_publication_mode, "host-copy") == 0) {
        options.pm4_command_buffer_publication_mode =
            IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE_HOST_COPY;
      } else if (strcmp(FLAG_pm4_publication_mode, "host-async-copy") == 0) {
        options.pm4_command_buffer_publication_mode =
            IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE_HOST_ASYNC_COPY;
      } else if (strcmp(FLAG_pm4_publication_mode,
                        "host-async-copy-nonblocking") == 0) {
        options.pm4_command_buffer_publication_mode =
            IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE_HOST_ASYNC_COPY_NONBLOCKING;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unrecognized PM4 publication mode: '%s'",
                                FLAG_pm4_publication_mode);
      }
    }

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool_;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), &options, &libhsa_, &topology_, &create_params,
        host_allocator_, &out_bundle->device));

    iree_async_frontier_tracker_t* frontier_tracker = nullptr;
    iree_async_frontier_tracker_options_t frontier_options =
        iree_async_frontier_tracker_options_default();
    frontier_options.axis_table_capacity = kFrontierAxisTableCapacity;
    iree_status_t status = iree_async_frontier_tracker_create(
        frontier_options, host_allocator_, &frontier_tracker);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_create_from_device(
          out_bundle->device, frontier_tracker, host_allocator_,
          &out_bundle->device_group);
    }
    iree_async_frontier_tracker_release(frontier_tracker);

    if (iree_status_is_ok(status)) {
      status = LoadExecutable(out_bundle);
    }
    if (iree_status_is_ok(status)) {
      status = AllocateBuffers(out_bundle);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_create(
          out_bundle->device, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &out_bundle->completion_semaphore);
    }
    if (!iree_status_is_ok(status)) {
      out_bundle->Release();
    }
    return status;
  }

  static iree_status_t BeginPm4FinalizeTimingProfile() {
    if (!FLAG_pm4_collect_finalize_timings) return iree_ok_status();
    if (pm4_finalize_timing_profile_active_) return iree_ok_status();
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_benchmark_discard_profile_sink_create(
        host_allocator_, &pm4_finalize_timing_profile_sink_));
    iree_hal_device_profiling_options_t options = {0};
    options.data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
    options.sink = pm4_finalize_timing_profile_sink_;
    iree_status_t status =
        iree_hal_device_profiling_begin(pm4_.device, &options);
    if (iree_status_is_ok(status)) {
      pm4_finalize_timing_profile_active_ = true;
    } else {
      iree_hal_profile_sink_release(pm4_finalize_timing_profile_sink_);
      pm4_finalize_timing_profile_sink_ = nullptr;
    }
    return status;
  }

  static void EndPm4FinalizeTimingProfile() {
    if (pm4_finalize_timing_profile_active_) {
      iree_status_t status = iree_hal_device_profiling_end(pm4_.device);
      if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
      }
      pm4_finalize_timing_profile_active_ = false;
    }
    iree_hal_profile_sink_release(pm4_finalize_timing_profile_sink_);
    pm4_finalize_timing_profile_sink_ = nullptr;
  }

  static iree_status_t LoadExecutable(DeviceBundle* bundle) {
    iree_const_byte_span_t executable_data = FindExecutableData(
        iree_make_cstring_view("pm4_command_buffer_benchmark_testdata.bin"));
    if (executable_data.data_length == 0) {
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "AMDGPU PM4 command-buffer benchmark executable not found");
    }

    iree_status_t status = iree_hal_executable_cache_create(
        bundle->device, iree_make_cstring_view("pm4_command_buffer_benchmark"),
        &bundle->executable_cache);

    char executable_format[128] = {0};
    iree_host_size_t inferred_size = 0;
    if (iree_status_is_ok(status)) {
      status = iree_hal_executable_cache_infer_format(
          bundle->executable_cache,
          IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA, executable_data,
          IREE_ARRAYSIZE(executable_format), executable_format, &inferred_size);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_executable_params_t executable_params;
      iree_hal_executable_params_initialize(&executable_params);
      executable_params.caching_mode =
          IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
      executable_params.executable_format =
          iree_make_cstring_view(executable_format);
      executable_params.executable_data = executable_data;
      status = iree_hal_executable_cache_prepare_executable(
          bundle->executable_cache, &executable_params, &bundle->executable);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_executable_lookup_export_by_name(
          bundle->executable, IREE_SV("model_a"), &bundle->model_a);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_executable_lookup_export_by_name(
          bundle->executable, IREE_SV("model_b"), &bundle->model_b);
    }
    return status;
  }

  static iree_status_t AllocateBuffers(DeviceBundle* bundle) {
    iree_hal_allocator_t* allocator = iree_hal_device_allocator(bundle->device);
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.min_alignment = kPayloadBufferAlignment;
    for (iree_host_size_t i = 0; i < kMaximumBindingTableCount; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
          allocator, params, kPayloadBufferLength, &bundle->buffers[i]));
      bundle->binding_table[i] = iree_hal_buffer_binding_t{
          /*buffer=*/bundle->buffers[i],
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      };
    }
    return iree_ok_status();
  }

  DeviceBundle& BundleForPath(CommandBufferPath path) {
    return IsPm4Path(path) ? pm4_ : aql_;
  }

  iree_status_t BeginAbabaCommandBuffer(
      CommandBufferPath path, BenchmarkSpec spec,
      iree_hal_command_buffer_t** out_command_buffer) {
    *out_command_buffer = nullptr;
    if (spec.operation_count <= 0) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "operation count must be positive");
    }
    if (spec.binding_table_count <= 1 ||
        spec.binding_table_count > (int64_t)kMaximumBindingTableCount) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "binding table count must be in [2, %" PRIhsz "]",
                              kMaximumBindingTableCount);
    }

    DeviceBundle& bundle = BundleForPath(path);
    const bool dynamic_path = IsDynamicPath(path);
    const iree_host_size_t binding_capacity =
        dynamic_path ? (iree_host_size_t)spec.binding_table_count : 0;

    iree_hal_command_buffer_mode_t command_buffer_mode =
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
    if (FLAG_command_buffer_unretained) {
      command_buffer_mode |= IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED;
    }
    if (FLAG_command_buffer_unvalidated) {
      command_buffer_mode |= IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
    }
    iree_hal_command_buffer_t* command_buffer = nullptr;
    iree_status_t status = iree_hal_command_buffer_create(
        bundle.device, command_buffer_mode, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
        IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity, &command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_begin(command_buffer);
    }

    int64_t dispatch_index = 0;
    while (dispatch_index < spec.operation_count && iree_status_is_ok(status)) {
      const int64_t group_size = NextDispatchGroupSize(
          dispatch_index, spec.operation_count, spec.overlap_percent);
      for (int64_t i = 0; i < group_size && iree_status_is_ok(status); ++i) {
        const int64_t current_dispatch = dispatch_index + i;
        const iree_hal_executable_export_ordinal_t export_ordinal =
            (current_dispatch & 1) ? bundle.model_b : bundle.model_a;
        const uint32_t slot0 =
            (uint32_t)(current_dispatch % spec.binding_table_count);
        uint32_t slot1 =
            (uint32_t)((current_dispatch * 7 + 1) % spec.binding_table_count);
        if (slot0 == slot1) {
          slot1 = (slot1 + 1) % (uint32_t)spec.binding_table_count;
        }
        std::array<iree_hal_buffer_ref_t, 2> binding_refs;
        if (dynamic_path) {
          binding_refs[0] = iree_hal_make_indirect_buffer_ref(
              slot0, /*offset=*/0, kPayloadBufferLength);
          binding_refs[1] = iree_hal_make_indirect_buffer_ref(
              slot1, /*offset=*/0, kPayloadBufferLength);
        } else {
          binding_refs[0] = iree_hal_make_buffer_ref(
              bundle.buffers[slot0], /*offset=*/0, kPayloadBufferLength);
          binding_refs[1] = iree_hal_make_buffer_ref(
              bundle.buffers[slot1], /*offset=*/0, kPayloadBufferLength);
        }
        const iree_hal_buffer_ref_list_t bindings = {
            /*count=*/binding_refs.size(),
            /*values=*/binding_refs.data(),
        };
        status = iree_hal_command_buffer_dispatch(
            command_buffer, bundle.executable, export_ordinal,
            iree_hal_make_static_dispatch_config(1, 1, 1),
            iree_const_byte_span_empty(), bindings,
            IREE_HAL_DISPATCH_FLAG_NONE);
      }
      dispatch_index += group_size;
      if (dispatch_index < spec.operation_count && iree_status_is_ok(status)) {
        status = EmitDispatchBarrier(command_buffer);
      }
    }
    if (iree_status_is_ok(status)) {
      *out_command_buffer = command_buffer;
    } else {
      iree_hal_command_buffer_release(command_buffer);
    }
    return status;
  }

  iree_status_t RecordAbabaCommandBuffer(
      CommandBufferPath path, BenchmarkSpec spec,
      iree_hal_command_buffer_t** out_command_buffer) {
    IREE_RETURN_IF_ERROR(
        BeginAbabaCommandBuffer(path, spec, out_command_buffer));
    iree_status_t status = iree_hal_command_buffer_end(*out_command_buffer);
    if (!iree_status_is_ok(status)) {
      iree_hal_command_buffer_release(*out_command_buffer);
      *out_command_buffer = nullptr;
    }
    return status;
  }

  iree_status_t SubmitCommandBuffer(CommandBufferPath path, BenchmarkSpec spec,
                                    iree_hal_command_buffer_t* command_buffer,
                                    SubmittedCompletion* out_completion) {
    DeviceBundle& bundle = BundleForPath(path);
    uint64_t payload_value = ++bundle.completion_payload_value;
    iree_hal_semaphore_t* signal_semaphore = bundle.completion_semaphore;
    const iree_hal_semaphore_list_t signal_semaphore_list = {
        /*count=*/1,
        /*semaphores=*/&signal_semaphore,
        /*payload_values=*/&payload_value,
    };
    const iree_hal_buffer_binding_table_t binding_table =
        IsDynamicPath(path)
            ? iree_hal_buffer_binding_table_t{
                  /*count=*/(iree_host_size_t)spec.binding_table_count,
                  /*bindings=*/bundle.binding_table.data(),
              }
            : iree_hal_buffer_binding_table_empty();
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
        bundle.device, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), signal_semaphore_list, command_buffer,
        binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
    out_completion->semaphore = signal_semaphore;
    out_completion->payload_value = payload_value;
    return iree_ok_status();
  }

  iree_status_t Wait(SubmittedCompletion completion) {
    return iree_hal_semaphore_wait(
        completion.semaphore, completion.payload_value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
  }

  void SetAqlCommandBufferCounters(benchmark::State& state,
                                   iree_hal_command_buffer_t* command_buffer) {
    const iree_hal_amdgpu_aql_program_t* program =
        iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
    uint64_t payload_block_count = 0;
    uint64_t total_aql_packet_count = 0;
    uint64_t total_block_bytes = 0;
    uint64_t total_used_bytes = 0;
    struct {
      uint64_t dispatch_count = 0;
      uint64_t payload_bytes = 0;
      uint64_t storage_span_bytes = 0;
    } prepublished_kernarg;
    struct {
      uint64_t dispatch_count = 0;
      uint64_t payload_bytes = 0;
      uint64_t reserved_bytes = 0;
    } queue_kernarg;

    for (const iree_hal_amdgpu_command_buffer_block_header_t* block =
             program->first_block;
         block; block = iree_hal_amdgpu_aql_program_block_next(
                    program->block_pool, block)) {
      const uint64_t binding_source_length =
          (uint64_t)block->binding_source_count *
          sizeof(iree_hal_amdgpu_command_buffer_binding_source_t);
      const uint64_t used_bytes = block->header_length + block->command_length +
                                  binding_source_length + block->rodata_length;
      if (block->aql_packet_count > 0) ++payload_block_count;
      total_aql_packet_count += block->aql_packet_count;
      total_block_bytes += block->block_length;
      total_used_bytes += used_bytes;

      const uint8_t* command_end =
          (const uint8_t*)block + block->command_offset + block->command_length;
      for (const iree_hal_amdgpu_command_buffer_command_header_t* command =
               iree_hal_amdgpu_command_buffer_block_commands_const(block);
           (const uint8_t*)command < command_end;
           command =
               iree_hal_amdgpu_command_buffer_command_next_const(command)) {
        if (command->opcode != IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH) {
          continue;
        }
        const iree_hal_amdgpu_command_buffer_dispatch_command_t*
            dispatch_command =
                (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)
                    command;
        const uint64_t kernarg_bytes =
            (uint64_t)dispatch_command->kernarg_length_qwords * 8u;
        if (dispatch_command->kernarg_strategy ==
            IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED) {
          ++prepublished_kernarg.dispatch_count;
          prepublished_kernarg.payload_bytes += kernarg_bytes;
          const uint64_t storage_end =
              (uint64_t)dispatch_command->payload_reference + kernarg_bytes;
          prepublished_kernarg.storage_span_bytes =
              std::max(prepublished_kernarg.storage_span_bytes, storage_end);
        } else {
          ++queue_kernarg.dispatch_count;
          queue_kernarg.payload_bytes += kernarg_bytes;
          queue_kernarg.reserved_bytes +=
              std::max<uint64_t>(
                  1, (kernarg_bytes + sizeof(iree_hal_amdgpu_kernarg_block_t) -
                      1) /
                         sizeof(iree_hal_amdgpu_kernarg_block_t)) *
              sizeof(iree_hal_amdgpu_kernarg_block_t);
        }
      }
    }

    state.counters["aql_packets_per_sync"] =
        static_cast<double>(total_aql_packet_count);
    state.counters["aql_program_blocks"] =
        static_cast<double>(program->block_count);
    state.counters["aql_payload_blocks"] =
        static_cast<double>(payload_block_count);
    state.counters["aql_program_occupancy_pct"] =
        total_block_bytes
            ? 100.0 * (double)total_used_bytes / (double)total_block_bytes
            : 0.0;
    state.counters["aql_prepublished_dispatches"] =
        static_cast<double>(prepublished_kernarg.dispatch_count);
    state.counters["aql_prepublished_kernarg_bytes"] =
        static_cast<double>(prepublished_kernarg.payload_bytes);
    state.counters["aql_prepublished_storage_span_bytes"] =
        static_cast<double>(prepublished_kernarg.storage_span_bytes);
    state.counters["aql_queue_kernarg_dispatches"] =
        static_cast<double>(queue_kernarg.dispatch_count);
    state.counters["aql_queue_kernarg_payload_bytes"] =
        static_cast<double>(queue_kernarg.payload_bytes);
    state.counters["aql_queue_kernarg_reserved_bytes"] =
        static_cast<double>(queue_kernarg.reserved_bytes);
  }

  void SetPm4CommandBufferCounters(benchmark::State& state,
                                   iree_hal_command_buffer_t* command_buffer) {
    const iree_hal_amdgpu_pm4_program_t* program =
        iree_hal_amdgpu_pm4_command_buffer_program(command_buffer);
    const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t* fixup_plan =
        iree_hal_amdgpu_pm4_command_buffer_fixup_plan(command_buffer);
    const iree_hal_amdgpu_pm4_command_buffer_publish_stats_t* publish_stats =
        iree_hal_amdgpu_pm4_command_buffer_publish_stats(command_buffer);
    state.counters["pm4_ib_dwords"] = static_cast<double>(program->dword_count);
    state.counters["pm4_ib_bytes"] =
        static_cast<double>(program->dword_count * sizeof(uint32_t));
    state.counters["pm4_fixup_entries"] =
        static_cast<double>(fixup_plan->entry_count);
    state.counters["pm4_fixup_entry_bytes"] = static_cast<double>(
        fixup_plan->entry_count *
        sizeof(iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t));
    state.counters["pm4_template_bytes"] =
        static_cast<double>(fixup_plan->target_byte_length);
    state.counters["pm4_publication_mode"] =
        strcmp(FLAG_pm4_publication_mode, "host-copy") == 0         ? 1.0
        : strcmp(FLAG_pm4_publication_mode, "host-async-copy") == 0 ? 2.0
        : strcmp(FLAG_pm4_publication_mode, "host-async-copy-nonblocking") == 0
            ? 3.0
            : 0.0;
    state.counters["pm4_finalize_timing_profile"] =
        FLAG_pm4_collect_finalize_timings ? 1.0 : 0.0;
    state.counters["pm4_finalize_us"] =
        static_cast<double>(publish_stats->total_finalize_ns) / 1000.0;
    state.counters["pm4_materialize_us"] =
        static_cast<double>(publish_stats->materialize_ns) / 1000.0;
    state.counters["pm4_resident_allocate_us"] =
        static_cast<double>(publish_stats->resident_allocate_ns) / 1000.0;
    state.counters["pm4_resident_allow_access_us"] =
        static_cast<double>(publish_stats->resident_allow_access_ns) / 1000.0;
    state.counters["pm4_resident_copy_us"] =
        static_cast<double>(publish_stats->resident_copy_ns) / 1000.0;
    state.counters["pm4_host_staging_allocate_us"] =
        static_cast<double>(publish_stats->host_staging_allocate_ns) / 1000.0;
    state.counters["pm4_host_staging_allow_access_us"] =
        static_cast<double>(publish_stats->host_staging_allow_access_ns) /
        1000.0;
    state.counters["pm4_host_record_bytes"] =
        static_cast<double>(publish_stats->host_record_bytes);
    state.counters["pm4_host_staging_bytes"] =
        static_cast<double>(publish_stats->host_staging_bytes);
    state.counters["pm4_resident_bytes"] =
        static_cast<double>(publish_stats->resident_bytes);
    state.counters["pm4_resident_copy_bytes"] =
        static_cast<double>(publish_stats->resident_copy_bytes);
    state.counters["pm4_resident_allocations"] =
        static_cast<double>(publish_stats->resident_allocation_count);
    state.counters["pm4_resident_allow_access_agents"] =
        static_cast<double>(publish_stats->resident_allow_access_agent_count);
    state.counters["pm4_host_staging_allocations"] =
        static_cast<double>(publish_stats->host_staging_allocation_count);
    state.counters["pm4_host_staging_allow_access_agents"] =
        static_cast<double>(
            publish_stats->host_staging_allow_access_agent_count);
    state.counters["pm4_execution_barrier_dwords"] =
        static_cast<double>(publish_stats->execution_barrier_dwords);
    state.counters["pm4_fixup_barrier_dwords"] =
        static_cast<double>(publish_stats->fixup_barrier_dwords);
    state.counters["pm4_dispatch_setup_dwords"] =
        static_cast<double>(publish_stats->dispatch_setup_dwords);
    state.counters["pm4_dispatch_user_data_dwords"] =
        static_cast<double>(publish_stats->dispatch_user_data_dwords);
    state.counters["pm4_dispatch_direct_dwords"] =
        static_cast<double>(publish_stats->dispatch_direct_dwords);
    state.counters["pm4_terminal_barrier_dwords"] =
        static_cast<double>(publish_stats->terminal_barrier_dwords);
  }

  void SetCounters(benchmark::State& state, CommandBufferPath path,
                   BenchmarkSpec spec,
                   iree_hal_command_buffer_t* command_buffer) {
    state.counters["dispatches_per_sync"] =
        static_cast<double>(spec.operation_count);
    state.counters["binding_table_entries"] =
        static_cast<double>(spec.binding_table_count);
    state.counters["overlap_dispatch_percent"] =
        static_cast<double>(spec.overlap_percent);
    state.counters["queue_submissions_per_sync"] = 1.0;
    state.counters["command_buffer_unretained"] =
        FLAG_command_buffer_unretained ? 1.0 : 0.0;
    state.counters["command_buffer_unvalidated"] =
        FLAG_command_buffer_unvalidated ? 1.0 : 0.0;
    state.counters["dynamic_binding_table_bytes_per_sync"] =
        IsDynamicPath(path)
            ? static_cast<double>(spec.binding_table_count * sizeof(uint64_t))
            : 0.0;
    if (iree_hal_amdgpu_aql_command_buffer_isa(command_buffer)) {
      SetAqlCommandBufferCounters(state, command_buffer);
    } else if (iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer)) {
      SetPm4CommandBufferCounters(state, command_buffer);
    }
    iree_hal_amdgpu_benchmark_set_completion_wait_counters(state);
  }

  void RunRecordFinalize(benchmark::State& state, CommandBufferPath path) {
    BenchmarkSpec spec = {
        /*operation_count=*/state.range(0),
        /*binding_table_count=*/state.range(1),
        /*overlap_percent=*/state.range(2),
    };
    iree_hal_command_buffer_t* last_command_buffer = nullptr;
    for (auto _ : state) {
      iree_hal_command_buffer_t* command_buffer = nullptr;
      if (!HandleStatus(state,
                        RecordAbabaCommandBuffer(path, spec, &command_buffer),
                        "failed to record ABABA command buffer")) {
        break;
      }
      state.PauseTiming();
      iree_hal_command_buffer_release(last_command_buffer);
      last_command_buffer = command_buffer;
      state.ResumeTiming();
    }
    if (last_command_buffer) {
      SetCounters(state, path, spec, last_command_buffer);
    }
    state.SetItemsProcessed(state.iterations() * spec.operation_count);
    iree_hal_command_buffer_release(last_command_buffer);
  }

  void RunRecordOnly(benchmark::State& state, CommandBufferPath path) {
    BenchmarkSpec spec = {
        /*operation_count=*/state.range(0),
        /*binding_table_count=*/state.range(1),
        /*overlap_percent=*/state.range(2),
    };
    iree_hal_command_buffer_t* last_command_buffer = nullptr;
    for (auto _ : state) {
      iree_hal_command_buffer_t* command_buffer = nullptr;
      if (!HandleStatus(state,
                        BeginAbabaCommandBuffer(path, spec, &command_buffer),
                        "failed to record ABABA command buffer")) {
        break;
      }
      state.PauseTiming();
      const bool finalize_ok =
          HandleStatus(state, iree_hal_command_buffer_end(command_buffer),
                       "failed to finalize ABABA command buffer");
      if (finalize_ok) {
        iree_hal_command_buffer_release(last_command_buffer);
        last_command_buffer = command_buffer;
      } else {
        iree_hal_command_buffer_release(command_buffer);
      }
      state.ResumeTiming();
      if (!finalize_ok) break;
    }
    if (last_command_buffer) {
      SetCounters(state, path, spec, last_command_buffer);
    }
    state.SetItemsProcessed(state.iterations() * spec.operation_count);
    iree_hal_command_buffer_release(last_command_buffer);
  }

  void RunFinalizeOnly(benchmark::State& state, CommandBufferPath path) {
    BenchmarkSpec spec = {
        /*operation_count=*/state.range(0),
        /*binding_table_count=*/state.range(1),
        /*overlap_percent=*/state.range(2),
    };
    iree_hal_command_buffer_t* last_command_buffer = nullptr;
    for (auto _ : state) {
      iree_hal_command_buffer_t* command_buffer = nullptr;
      state.PauseTiming();
      const bool record_ok = HandleStatus(
          state, BeginAbabaCommandBuffer(path, spec, &command_buffer),
          "failed to record ABABA command buffer");
      state.ResumeTiming();
      if (!record_ok) break;

      const bool finalize_ok =
          HandleStatus(state, iree_hal_command_buffer_end(command_buffer),
                       "failed to finalize ABABA command buffer");
      state.PauseTiming();
      if (finalize_ok) {
        iree_hal_command_buffer_release(last_command_buffer);
        last_command_buffer = command_buffer;
      } else {
        iree_hal_command_buffer_release(command_buffer);
      }
      state.ResumeTiming();
      if (!finalize_ok) break;
    }
    if (last_command_buffer) {
      SetCounters(state, path, spec, last_command_buffer);
    }
    state.SetItemsProcessed(state.iterations() * spec.operation_count);
    iree_hal_command_buffer_release(last_command_buffer);
  }

  void RunSubmitOnly(benchmark::State& state, CommandBufferPath path) {
    BenchmarkSpec spec = {
        /*operation_count=*/state.range(0),
        /*binding_table_count=*/state.range(1),
        /*overlap_percent=*/state.range(2),
    };
    iree_hal_command_buffer_t* command_buffer = nullptr;
    if (!HandleStatus(state,
                      RecordAbabaCommandBuffer(path, spec, &command_buffer),
                      "failed to record ABABA command buffer")) {
      return;
    }
    for (auto _ : state) {
      SubmittedCompletion completion;
      if (!HandleStatus(
              state,
              SubmitCommandBuffer(path, spec, command_buffer, &completion),
              "ABABA command-buffer submit failed")) {
        break;
      }
      state.PauseTiming();
      const bool wait_ok = HandleStatus(state, Wait(completion),
                                        "ABABA command-buffer wait failed");
      state.ResumeTiming();
      if (!wait_ok) break;
    }
    SetCounters(state, path, spec, command_buffer);
    state.SetItemsProcessed(state.iterations() * spec.operation_count);
    iree_hal_command_buffer_release(command_buffer);
  }

  void RunSubmitWait(benchmark::State& state, CommandBufferPath path) {
    BenchmarkSpec spec = {
        /*operation_count=*/state.range(0),
        /*binding_table_count=*/state.range(1),
        /*overlap_percent=*/state.range(2),
    };
    iree_hal_command_buffer_t* command_buffer = nullptr;
    if (!HandleStatus(state,
                      RecordAbabaCommandBuffer(path, spec, &command_buffer),
                      "failed to record ABABA command buffer")) {
      return;
    }
    for (auto _ : state) {
      SubmittedCompletion completion;
      if (!HandleStatus(
              state,
              SubmitCommandBuffer(path, spec, command_buffer, &completion),
              "ABABA command-buffer submit failed")) {
        break;
      }
      if (!HandleStatus(state, Wait(completion),
                        "ABABA command-buffer wait failed")) {
        break;
      }
    }
    SetCounters(state, path, spec, command_buffer);
    state.SetItemsProcessed(state.iterations() * spec.operation_count);
    iree_hal_command_buffer_release(command_buffer);
  }

  void RunEndToEnd(benchmark::State& state, CommandBufferPath path) {
    BenchmarkSpec spec = {
        /*operation_count=*/state.range(0),
        /*binding_table_count=*/state.range(1),
        /*overlap_percent=*/state.range(2),
    };
    iree_hal_command_buffer_t* last_command_buffer = nullptr;
    for (auto _ : state) {
      iree_hal_command_buffer_t* command_buffer = nullptr;
      if (!HandleStatus(state,
                        RecordAbabaCommandBuffer(path, spec, &command_buffer),
                        "failed to record ABABA command buffer")) {
        break;
      }
      SubmittedCompletion completion;
      if (!HandleStatus(
              state,
              SubmitCommandBuffer(path, spec, command_buffer, &completion),
              "ABABA command-buffer submit failed")) {
        iree_hal_command_buffer_release(command_buffer);
        break;
      }
      if (!HandleStatus(state, Wait(completion),
                        "ABABA command-buffer wait failed")) {
        iree_hal_command_buffer_release(command_buffer);
        break;
      }
      state.PauseTiming();
      iree_hal_command_buffer_release(last_command_buffer);
      last_command_buffer = command_buffer;
      state.ResumeTiming();
    }
    if (last_command_buffer) {
      SetCounters(state, path, spec, last_command_buffer);
    }
    state.SetItemsProcessed(state.iterations() * spec.operation_count);
    iree_hal_command_buffer_release(last_command_buffer);
  }

  static iree_allocator_t host_allocator_;
  static bool initialized_;
  static bool available_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
  static iree_async_proactor_pool_t* proactor_pool_;
  static bool pm4_finalize_timing_profile_active_;
  static iree_hal_profile_sink_t* pm4_finalize_timing_profile_sink_;
  static DeviceBundle aql_;
  static DeviceBundle pm4_;
};

iree_allocator_t Pm4CommandBufferBenchmark::host_allocator_ =
    iree_allocator_system();
bool Pm4CommandBufferBenchmark::initialized_ = false;
bool Pm4CommandBufferBenchmark::available_ = false;
iree_hal_amdgpu_libhsa_t Pm4CommandBufferBenchmark::libhsa_;
iree_hal_amdgpu_topology_t Pm4CommandBufferBenchmark::topology_;
iree_async_proactor_pool_t* Pm4CommandBufferBenchmark::proactor_pool_ = nullptr;
bool Pm4CommandBufferBenchmark::pm4_finalize_timing_profile_active_ = false;
iree_hal_profile_sink_t*
    Pm4CommandBufferBenchmark::pm4_finalize_timing_profile_sink_ = nullptr;
DeviceBundle Pm4CommandBufferBenchmark::aql_;
DeviceBundle Pm4CommandBufferBenchmark::pm4_;

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlDynamicAbabaRecordFinalize)
(benchmark::State& state) {
  RunRecordFinalize(state, CommandBufferPath::kAqlDynamic);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlStaticAbabaRecordFinalize)
(benchmark::State& state) {
  RunRecordFinalize(state, CommandBufferPath::kAqlStatic);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4StaticAbabaRecordFinalize)
(benchmark::State& state) {
  RunRecordFinalize(state, CommandBufferPath::kPm4Static);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4FixupAbabaRecordFinalize)
(benchmark::State& state) {
  RunRecordFinalize(state, CommandBufferPath::kPm4Fixup);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlDynamicAbabaRecordOnly)
(benchmark::State& state) {
  RunRecordOnly(state, CommandBufferPath::kAqlDynamic);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlStaticAbabaRecordOnly)
(benchmark::State& state) {
  RunRecordOnly(state, CommandBufferPath::kAqlStatic);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4StaticAbabaRecordOnly)
(benchmark::State& state) {
  RunRecordOnly(state, CommandBufferPath::kPm4Static);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4FixupAbabaRecordOnly)
(benchmark::State& state) {
  RunRecordOnly(state, CommandBufferPath::kPm4Fixup);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlDynamicAbabaFinalizeOnly)
(benchmark::State& state) {
  RunFinalizeOnly(state, CommandBufferPath::kAqlDynamic);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlStaticAbabaFinalizeOnly)
(benchmark::State& state) {
  RunFinalizeOnly(state, CommandBufferPath::kAqlStatic);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4StaticAbabaFinalizeOnly)
(benchmark::State& state) {
  RunFinalizeOnly(state, CommandBufferPath::kPm4Static);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4FixupAbabaFinalizeOnly)
(benchmark::State& state) {
  RunFinalizeOnly(state, CommandBufferPath::kPm4Fixup);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlDynamicAbabaSubmitOnly)
(benchmark::State& state) {
  RunSubmitOnly(state, CommandBufferPath::kAqlDynamic);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlStaticAbabaSubmitOnly)
(benchmark::State& state) {
  RunSubmitOnly(state, CommandBufferPath::kAqlStatic);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4StaticAbabaSubmitOnly)
(benchmark::State& state) {
  RunSubmitOnly(state, CommandBufferPath::kPm4Static);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4FixupAbabaSubmitOnly)
(benchmark::State& state) {
  RunSubmitOnly(state, CommandBufferPath::kPm4Fixup);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlDynamicAbabaSubmitWait)
(benchmark::State& state) {
  RunSubmitWait(state, CommandBufferPath::kAqlDynamic);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlStaticAbabaSubmitWait)
(benchmark::State& state) {
  RunSubmitWait(state, CommandBufferPath::kAqlStatic);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4StaticAbabaSubmitWait)
(benchmark::State& state) {
  RunSubmitWait(state, CommandBufferPath::kPm4Static);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4FixupAbabaSubmitWait)
(benchmark::State& state) {
  RunSubmitWait(state, CommandBufferPath::kPm4Fixup);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlDynamicAbabaEndToEnd)
(benchmark::State& state) {
  RunEndToEnd(state, CommandBufferPath::kAqlDynamic);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, AqlStaticAbabaEndToEnd)
(benchmark::State& state) { RunEndToEnd(state, CommandBufferPath::kAqlStatic); }

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4StaticAbabaEndToEnd)
(benchmark::State& state) { RunEndToEnd(state, CommandBufferPath::kPm4Static); }

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4FixupAbabaEndToEnd)
(benchmark::State& state) { RunEndToEnd(state, CommandBufferPath::kPm4Fixup); }

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4StaticSizePolicyRecordFinalize)
(benchmark::State& state) {
  RunRecordFinalize(state, CommandBufferPath::kPm4Static);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4FixupSizePolicyRecordFinalize)
(benchmark::State& state) {
  RunRecordFinalize(state, CommandBufferPath::kPm4Fixup);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4StaticSizePolicyFinalizeOnly)
(benchmark::State& state) {
  RunFinalizeOnly(state, CommandBufferPath::kPm4Static);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4FixupSizePolicyFinalizeOnly)
(benchmark::State& state) {
  RunFinalizeOnly(state, CommandBufferPath::kPm4Fixup);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4StaticSizePolicySubmitWait)
(benchmark::State& state) {
  RunSubmitWait(state, CommandBufferPath::kPm4Static);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4FixupSizePolicySubmitWait)
(benchmark::State& state) {
  RunSubmitWait(state, CommandBufferPath::kPm4Fixup);
}

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4StaticSizePolicyEndToEnd)
(benchmark::State& state) { RunEndToEnd(state, CommandBufferPath::kPm4Static); }

BENCHMARK_DEFINE_F(Pm4CommandBufferBenchmark, Pm4FixupSizePolicyEndToEnd)
(benchmark::State& state) { RunEndToEnd(state, CommandBufferPath::kPm4Fixup); }

void ApplyAbabaArguments(benchmark::Benchmark* benchmark) {
  benchmark->ArgsProduct({{5000, 10000}, {10, 32, 100}, {0, 20}});
  benchmark->ArgNames(
      {"dispatch_count", "binding_table_entries", "overlap_percent"});
}

void ApplySizePolicyArguments(benchmark::Benchmark* benchmark) {
  benchmark->ArgsProduct(
      {{16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000},
       {100},
       {20}});
  benchmark->ArgNames(
      {"dispatch_count", "binding_table_entries", "overlap_percent"});
}

#define IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(name) \
  BENCHMARK_REGISTER_F(Pm4CommandBufferBenchmark, name)         \
      ->Apply(ApplyAbabaArguments)                              \
      ->UseRealTime()                                           \
      ->Unit(benchmark::kMicrosecond)

IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(
    AqlDynamicAbabaRecordFinalize);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(AqlStaticAbabaRecordFinalize);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4StaticAbabaRecordFinalize);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4FixupAbabaRecordFinalize);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(AqlDynamicAbabaRecordOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(AqlStaticAbabaRecordOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4StaticAbabaRecordOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4FixupAbabaRecordOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(AqlDynamicAbabaFinalizeOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(AqlStaticAbabaFinalizeOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4StaticAbabaFinalizeOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4FixupAbabaFinalizeOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(AqlDynamicAbabaSubmitOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(AqlStaticAbabaSubmitOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4StaticAbabaSubmitOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4FixupAbabaSubmitOnly);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(AqlDynamicAbabaSubmitWait);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(AqlStaticAbabaSubmitWait);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4StaticAbabaSubmitWait);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4FixupAbabaSubmitWait);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(AqlDynamicAbabaEndToEnd);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(AqlStaticAbabaEndToEnd);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4StaticAbabaEndToEnd);
IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK(Pm4FixupAbabaEndToEnd);

#undef IREE_AMDGPU_REGISTER_PM4_COMMAND_BUFFER_BENCHMARK

#define IREE_AMDGPU_REGISTER_PM4_SIZE_POLICY_BENCHMARK(name) \
  BENCHMARK_REGISTER_F(Pm4CommandBufferBenchmark, name)      \
      ->Apply(ApplySizePolicyArguments)                      \
      ->UseRealTime()                                        \
      ->Unit(benchmark::kMicrosecond)

IREE_AMDGPU_REGISTER_PM4_SIZE_POLICY_BENCHMARK(
    Pm4StaticSizePolicyRecordFinalize);
IREE_AMDGPU_REGISTER_PM4_SIZE_POLICY_BENCHMARK(
    Pm4FixupSizePolicyRecordFinalize);
IREE_AMDGPU_REGISTER_PM4_SIZE_POLICY_BENCHMARK(Pm4StaticSizePolicyFinalizeOnly);
IREE_AMDGPU_REGISTER_PM4_SIZE_POLICY_BENCHMARK(Pm4FixupSizePolicyFinalizeOnly);
IREE_AMDGPU_REGISTER_PM4_SIZE_POLICY_BENCHMARK(Pm4StaticSizePolicySubmitWait);
IREE_AMDGPU_REGISTER_PM4_SIZE_POLICY_BENCHMARK(Pm4FixupSizePolicySubmitWait);
IREE_AMDGPU_REGISTER_PM4_SIZE_POLICY_BENCHMARK(Pm4StaticSizePolicyEndToEnd);
IREE_AMDGPU_REGISTER_PM4_SIZE_POLICY_BENCHMARK(Pm4FixupSizePolicyEndToEnd);

#undef IREE_AMDGPU_REGISTER_PM4_SIZE_POLICY_BENCHMARK

}  // namespace

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  Pm4CommandBufferBenchmark::DeinitializeOnce();
  return 0;
}
