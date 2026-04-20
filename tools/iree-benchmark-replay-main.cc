// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "benchmark/benchmark.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/replay/execute.h"
#include "iree/io/file_contents.h"
#include "iree/tooling/device_util.h"

namespace {

void BenchmarkReplay(iree_const_byte_span_t file_contents,
                     iree_hal_device_group_t* device_group,
                     iree_hal_profiling_from_flags_t* profiling,
                     benchmark::State& state) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  for (auto _ : state) {
    (void)_;
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "BenchmarkIteration");
    IREE_TRACE_FRAME_MARK_NAMED("ReplayIteration");
    IREE_CHECK_OK(iree_hal_replay_execute_file(file_contents, device_group,
                                               &options, host_allocator));
    IREE_TRACE_ZONE_END(z0);
    state.PauseTiming();
    IREE_CHECK_OK(iree_hal_flush_profiling_from_flags(profiling));
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations());
}

iree_status_t CreateDeviceGroupFromList(
    iree_hal_device_list_t* device_list, iree_allocator_t host_allocator,
    iree_hal_device_group_t** out_device_group) {
  IREE_ASSERT_ARGUMENT(device_list);
  IREE_ASSERT_ARGUMENT(out_device_group);
  *out_device_group = nullptr;

  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), host_allocator,
      &frontier_tracker));

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
  iree_async_frontier_tracker_release(frontier_tracker);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < device_list->count && iree_status_is_ok(status); ++i) {
    status = iree_hal_device_group_builder_add_device(&builder,
                                                      device_list->devices[i]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_builder_finalize(&builder, host_allocator,
                                                    out_device_group);
  } else {
    iree_hal_device_group_builder_deinitialize(&builder);
  }
  return status;
}

}  // namespace

static int runMain(int argc, char** argv) {
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree-benchmark-replay");

  iree_flags_set_usage(
      "iree-benchmark-replay",
      "Benchmarks deterministic execution of an IREE HAL replay file.\n"
      "\n"
      "Use --device= to select the target HAL device, matching\n"
      "iree-benchmark-module. Benchmark flags are forwarded to Google\n"
      "Benchmark. HAL-native profiling flags capture the timed replay work\n"
      "and profiling flushes are excluded from benchmark timing.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  ::benchmark::Initialize(&argc, argv);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();
  if (argc != 2) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected one replay file path argument");
  }

  iree_io_file_contents_t* file_contents = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_map(iree_make_cstring_view(argv[1]),
                                       IREE_IO_FILE_ACCESS_READ, host_allocator,
                                       &file_contents);
  }

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), host_allocator,
        &proactor_pool);
  }

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  iree_hal_device_list_t* device_list = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_devices_from_flags(
        iree_hal_available_driver_registry(), iree_hal_default_device_uri(),
        &create_params, host_allocator, &device_list);
  }

  iree_hal_device_group_t* device_group = nullptr;
  if (iree_status_is_ok(status)) {
    status =
        CreateDeviceGroupFromList(device_list, host_allocator, &device_group);
  }

  // Start profiling after tool setup. The replay payload itself may contain
  // setup operations captured from the original application; those remain part
  // of the benchmark because they are user HAL traffic.
  iree_hal_profiling_from_flags_t* profiling = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        device_group, host_allocator, &profiling);
  }

  if (iree_status_is_ok(status)) {
    benchmark::RegisterBenchmark("BM_replay",
                                 [=](benchmark::State& state) -> void {
                                   BenchmarkReplay(file_contents->const_buffer,
                                                   device_group, profiling,
                                                   state);
                                 })
        ->MeasureProcessCPUTime()
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond);
    ::benchmark::RunSpecifiedBenchmarks();
  }

  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  iree_hal_device_group_release(device_group);
  iree_hal_device_list_free(device_list);
  iree_async_proactor_pool_release(proactor_pool);
  iree_io_file_contents_free(file_contents);

  int exit_code = EXIT_SUCCESS;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  IREE_TRACE_ZONE_END(z0);
  return exit_code;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  int exit_code = runMain(argc, argv);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
