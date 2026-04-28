// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>

#include <cstring>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/replay/execute.h"
#include "iree/io/file_contents.h"
#include "iree/tooling/device_util.h"

IREE_FLAG_LIST(
    string, replay_file_remap,
    "Remaps captured external file path prefixes before replay opens them. "
    "Repeat as --replay_file_remap=captured_prefix=replay_prefix.");
IREE_FLAG_LIST(
    string, replay_executable_substitution,
    "Substitutes a captured executable payload. Repeat as "
    "--replay_executable_substitution=EXECUTABLE_ID=PATH to infer the format, "
    "or --replay_executable_substitution=EXECUTABLE_ID@FORMAT=PATH when the "
    "format must be explicit. Use all=PATH or all@FORMAT=PATH to apply one "
    "replacement to every captured executable.");
IREE_FLAG(
    string, replay_scope, "",
    "Times only replay operations inside matching named scope markers. The "
    "full replay still executes each iteration so setup and teardown remain "
    "faithful, but Google Benchmark reports manually accumulated time from "
    "matching scope regions.");
IREE_FLAG(bool, agents_md, false,
          "Prints AGENTS.md guidance for iree-benchmark-replay and exits.");

namespace {

static const char kIreeBenchmarkReplayUsage[] =
    "Benchmarks deterministic execution of an IREE HAL replay file.\n"
    "\n"
    "Each Google Benchmark iteration replays the complete captured HAL stream\n"
    "against the selected device group. Profiling flushes are performed "
    "outside\n"
    "the timed region, so --device_profiling_output captures the useful "
    "replay\n"
    "work without charging the benchmark iteration for profile serialization.\n"
    "\n"
    "Usage:\n"
    "  iree-benchmark-replay [benchmark options] [replay options]\n"
    "      <capture.ireereplay>\n"
    "\n"
    "Important flags:\n"
    "  --device=URI\n"
    "      Target HAL device URI, matching iree-benchmark-module.\n"
    "  --replay_file_remap=captured_prefix=replay_prefix\n"
    "      Remaps referenced external file paths before strict identity\n"
    "      validation. Repeat the flag for multiple roots.\n"
    "  --replay_executable_substitution=EXECUTABLE_ID=PATH\n"
    "  --replay_executable_substitution=EXECUTABLE_ID@FORMAT=PATH\n"
    "  --replay_executable_substitution=all=PATH\n"
    "  --replay_executable_substitution=all@FORMAT=PATH\n"
    "      Replaces captured executable payloads for kernel iteration. The "
    "same\n"
    "      syntax and ABI validation as iree-run-replay applies.\n"
    "  --replay_scope=name\n"
    "      Times only regions enclosed by matching replay scope markers. The\n"
    "      complete capture still executes on every iteration.\n"
    "  --benchmark_min_time=20x\n"
    "      Useful for fixed-iteration replay benchmarking.\n"
    "  --device_profiling_mode=queue-events,host-execution\n"
    "  --device_profiling_output=path.ireeprof\n"
    "      Captures HAL profiling for benchmarked replay iterations.\n"
    "  --agents_md\n"
    "      Prints AGENTS.md guidance specific to iree-benchmark-replay. Use\n"
    "      `iree-run-replay --agents_md` for the full replay tool playbook.\n"
    "\n"
    "Examples:\n"
    "  iree-benchmark-replay --device=local-sync --benchmark_min_time=20x \\\n"
    "      /tmp/model.ireereplay\n"
    "  iree-benchmark-replay --device=local-sync --benchmark_min_time=20x \\\n"
    "      --replay_scope=execute /tmp/model.ireereplay\n";

static void PrintBenchmarkReplayAgentMarkdown(FILE* file) {
  fputs(
      "# iree-benchmark-replay\n"
      "\n"
      "`iree-benchmark-replay` executes a prepared `.ireereplay` plan inside\n"
      "Google Benchmark. It creates the replay plan once, then executes fresh\n"
      "per-run replay object state for each iteration.\n"
      "\n"
      "Use fixed iteration counts for small replay workloads:\n"
      "\n"
      "```bash\n"
      "iree-benchmark-replay --device=local-sync --benchmark_min_time=20x \\\n"
      "  /tmp/model.ireereplay\n"
      "```\n"
      "\n"
      "`--replay_scope=name` reports manual time accumulated inside matching\n"
      "scope markers while still executing the complete captured stream every\n"
      "iteration. Use this when a capture includes setup/teardown and you "
      "want\n"
      "numbers for the VM invocation or another named phase:\n"
      "\n"
      "```bash\n"
      "iree-benchmark-replay --device=local-sync --benchmark_min_time=20x \\\n"
      "  --replay_scope=execute /tmp/model.ireereplay\n"
      "```\n"
      "\n"
      "Profiling flushes happen outside unscoped benchmark timing. Scoped "
      "manual\n"
      "timing records only the selected scope and still flushes profiling "
      "after\n"
      "the replay iteration completes:\n"
      "\n"
      "```bash\n"
      "iree-benchmark-replay --device=amdgpu --benchmark_min_time=50x \\\n"
      "  --device_profiling_mode=queue-events,device-queue-events \\\n"
      "  --device_profiling_output=/tmp/model-replay.ireeprof \\\n"
      "  /tmp/model.ireereplay\n"
      "```\n"
      "\n"
      "For capture, single-run replay, file remapping, executable "
      "substitution,\n"
      "dump JSONL, and the shared replay failure contract, pipe "
      "`iree-run-replay --agents_md` into your AGENTS.md.\n",
      file);
}

typedef struct ReplayExecutableSubstitution {
  // True when this replacement applies to every captured executable.
  bool match_all;
  // Captured executable object id to replace.
  iree_hal_replay_object_id_t executable_id;
  // Optional replacement executable format.
  iree_string_view_t executable_format;
  // Replacement executable file path.
  iree_string_view_t source_path;
  // Mapped replacement executable file contents.
  iree_io_file_contents_t* file_contents;
} ReplayExecutableSubstitution;

typedef struct ReplayExecutableSubstitutionState {
  // Replacement entries parsed from --replay_executable_substitution.
  ReplayExecutableSubstitution* entries;
  // Number of entries in |entries|.
  iree_host_size_t entry_count;
} ReplayExecutableSubstitutionState;

typedef struct ReplayScopeTimingState {
  // Selected scope name borrowed from the flag storage.
  iree_string_view_t selected_scope;
  // State for matching selected scope intervals.
  struct {
    // Nesting depth of matching selected scopes.
    iree_host_size_t depth;
    // True once at least one matching begin marker was observed.
    bool observed_begin;
    // Monotonic start timestamp for the active outermost matching scope.
    iree_time_t start_time_ns;
    // Accumulated wall-clock time spent inside matching scope regions.
    iree_duration_t elapsed_time_ns;
  } match;
} ReplayScopeTimingState;

void ReleaseReplayExecutableSubstitutions(
    iree_allocator_t host_allocator, ReplayExecutableSubstitutionState* state) {
  for (iree_host_size_t i = 0; i < state->entry_count; ++i) {
    iree_io_file_contents_free(state->entries[i].file_contents);
  }
  iree_allocator_free(host_allocator, state->entries);
  memset(state, 0, sizeof(*state));
}

iree_status_t ParseReplayExecutableSubstitutions(
    iree_allocator_t host_allocator,
    ReplayExecutableSubstitutionState* out_state) {
  IREE_ASSERT_ARGUMENT(out_state);
  memset(out_state, 0, sizeof(*out_state));

  iree_flag_string_list_t flag_list =
      FLAG_replay_executable_substitution_list();
  if (flag_list.count == 0) return iree_ok_status();

  iree_host_size_t entry_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          flag_list.count, sizeof(*out_state->entries), &entry_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay executable substitution list is too large");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, entry_size,
                                             (void**)&out_state->entries));
  memset(out_state->entries, 0, entry_size);
  out_state->entry_count = flag_list.count;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < flag_list.count && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t selector;
    iree_string_view_t path;
    if (iree_string_view_split(flag_list.values[i], '=', &selector, &path) <
            0 ||
        iree_string_view_is_empty(selector) ||
        iree_string_view_is_empty(path)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "--replay_executable_substitution values must be "
                           "EXECUTABLE_ID=PATH, EXECUTABLE_ID@FORMAT=PATH, "
                           "all=PATH, or all@FORMAT=PATH");
      break;
    }

    iree_string_view_t id_string = selector;
    iree_string_view_t executable_format = iree_string_view_empty();
    iree_string_view_t maybe_format;
    if (iree_string_view_split(selector, '@', &id_string, &maybe_format) >= 0) {
      if (iree_string_view_is_empty(id_string) ||
          iree_string_view_is_empty(maybe_format)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--replay_executable_substitution selector must be "
            "EXECUTABLE_ID, EXECUTABLE_ID@FORMAT, all, or all@FORMAT");
        break;
      }
      executable_format = maybe_format;
    }

    uint64_t executable_id = 0;
    const bool match_all = iree_string_view_equal(id_string, IREE_SV("all"));
    if (match_all) {
      if (flag_list.count != 1) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--replay_executable_substitution all selector cannot be combined "
            "with other executable substitutions");
        break;
      }
    } else {
      if (!iree_string_view_atoi_uint64(id_string, &executable_id) ||
          executable_id == IREE_HAL_REPLAY_OBJECT_ID_NONE) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--replay_executable_substitution executable selector must be "
            "a non-zero integer id or all");
        break;
      }
      for (iree_host_size_t j = 0; j < i; ++j) {
        if (out_state->entries[j].executable_id == executable_id) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "--replay_executable_substitution repeats executable id %" PRIu64,
              executable_id);
          break;
        }
      }
    }
    if (!iree_status_is_ok(status)) break;

    out_state->entries[i].match_all = match_all;
    out_state->entries[i].executable_id =
        (iree_hal_replay_object_id_t)executable_id;
    out_state->entries[i].executable_format = executable_format;
    out_state->entries[i].source_path = path;
    status = iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ,
                                       host_allocator,
                                       &out_state->entries[i].file_contents);
  }
  if (!iree_status_is_ok(status)) {
    ReleaseReplayExecutableSubstitutions(host_allocator, out_state);
  }
  return status;
}

iree_status_t ReplayExecutableSubstitutionCallback(
    void* user_data,
    const iree_hal_replay_executable_substitution_request_t* request,
    iree_hal_replay_executable_substitution_t* out_substitution) {
  ReplayExecutableSubstitutionState* state =
      (ReplayExecutableSubstitutionState*)user_data;
  memset(out_substitution, 0, sizeof(*out_substitution));
  for (iree_host_size_t i = 0; i < state->entry_count; ++i) {
    const ReplayExecutableSubstitution* entry = &state->entries[i];
    if (!entry->match_all && entry->executable_id != request->executable_id) {
      continue;
    }
    out_substitution->substitute = true;
    out_substitution->source = entry->source_path;
    out_substitution->executable_format = entry->executable_format;
    out_substitution->executable_data = entry->file_contents->const_buffer;
    return iree_ok_status();
  }
  return iree_ok_status();
}

iree_status_t ReplayScopeTimingCallback(
    void* user_data, const iree_hal_replay_scope_event_t* event) {
  ReplayScopeTimingState* state = (ReplayScopeTimingState*)user_data;
  if (!iree_string_view_equal(event->name, state->selected_scope)) {
    return iree_ok_status();
  }

  switch (event->type) {
    case IREE_HAL_REPLAY_SCOPE_EVENT_TYPE_BEGIN:
      state->match.observed_begin = true;
      if (state->match.depth == 0) {
        state->match.start_time_ns = iree_time_now();
      }
      ++state->match.depth;
      return iree_ok_status();
    case IREE_HAL_REPLAY_SCOPE_EVENT_TYPE_END:
      if (state->match.depth == 0) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "replay scope '%.*s' ended before it began",
                                (int)state->selected_scope.size,
                                state->selected_scope.data);
      }
      --state->match.depth;
      if (state->match.depth == 0) {
        state->match.elapsed_time_ns +=
            iree_time_now() - state->match.start_time_ns;
        state->match.start_time_ns = 0;
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "unknown replay scope event type %" PRIu32,
                              event->type);
  }
}

void BenchmarkReplay(const iree_hal_replay_plan_t* replay_plan,
                     iree_hal_device_group_t* device_group,
                     iree_hal_profiling_from_flags_t* profiling,
                     iree_hal_replay_execute_options_t options,
                     benchmark::State& state) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_string_view_t selected_scope = iree_make_cstring_view(FLAG_replay_scope);
  const bool scoped_timing = !iree_string_view_is_empty(selected_scope);
  for (auto _ : state) {
    (void)_;
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "BenchmarkIteration");
    IREE_TRACE_FRAME_MARK_NAMED("ReplayIteration");
    ReplayScopeTimingState scope_state = {
        /*.selected_scope=*/selected_scope,
        /*.match=*/{},
    };
    if (scoped_timing) {
      options.scope_event_callback.fn = ReplayScopeTimingCallback;
      options.scope_event_callback.user_data = &scope_state;
    }
    iree_status_t status = iree_hal_replay_plan_execute(
        replay_plan, device_group, &options, host_allocator);
    IREE_CHECK_OK(status);
    if (scoped_timing) {
      if (!scope_state.match.observed_begin) {
        IREE_CHECK_OK(iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "--replay_scope='%.*s' did not match any replay scope marker",
            (int)selected_scope.size, selected_scope.data));
      }
      if (scope_state.match.depth != 0) {
        IREE_CHECK_OK(iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "--replay_scope='%.*s' matched a scope that did not end",
            (int)selected_scope.size, selected_scope.data));
      }
      state.SetIterationTime((double)scope_state.match.elapsed_time_ns /
                             1000000000.0);
    }
    IREE_TRACE_ZONE_END(z0);
    if (!scoped_timing) {
      state.PauseTiming();
      IREE_CHECK_OK(iree_hal_flush_profiling_from_flags(profiling));
      state.ResumeTiming();
    } else {
      IREE_CHECK_OK(iree_hal_flush_profiling_from_flags(profiling));
    }
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

iree_status_t ParseReplayFileRemaps(
    iree_allocator_t host_allocator,
    iree_hal_replay_file_path_remap_t** out_file_path_remaps,
    iree_host_size_t* out_file_path_remap_count) {
  IREE_ASSERT_ARGUMENT(out_file_path_remaps);
  IREE_ASSERT_ARGUMENT(out_file_path_remap_count);
  *out_file_path_remaps = nullptr;
  *out_file_path_remap_count = 0;

  iree_flag_string_list_t flag_list = FLAG_replay_file_remap_list();
  if (flag_list.count == 0) return iree_ok_status();

  iree_host_size_t remap_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          flag_list.count, sizeof(**out_file_path_remaps), &remap_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay file remap list is too large");
  }
  iree_hal_replay_file_path_remap_t* file_path_remaps = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, remap_size,
                                             (void**)&file_path_remaps));
  memset(file_path_remaps, 0, remap_size);
  for (iree_host_size_t i = 0; i < flag_list.count; ++i) {
    iree_string_view_t captured_prefix;
    iree_string_view_t replay_prefix;
    if (iree_string_view_split(flag_list.values[i], '=', &captured_prefix,
                               &replay_prefix) < 0 ||
        iree_string_view_is_empty(captured_prefix)) {
      iree_allocator_free(host_allocator, file_path_remaps);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--replay_file_remap values must be captured_prefix=replay_prefix");
    }
    file_path_remaps[i].captured_prefix = captured_prefix;
    file_path_remaps[i].replay_prefix = replay_prefix;
  }
  *out_file_path_remaps = file_path_remaps;
  *out_file_path_remap_count = flag_list.count;
  return iree_ok_status();
}

}  // namespace

static int runMain(int argc, char** argv) {
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree-benchmark-replay");

  iree_flags_set_usage("iree-benchmark-replay", kIreeBenchmarkReplayUsage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  if (FLAG_agents_md) {
    PrintBenchmarkReplayAgentMarkdown(stdout);
    fflush(stdout);
    IREE_TRACE_ZONE_END(z0);
    return EXIT_SUCCESS;
  }
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

  iree_hal_replay_file_path_remap_t* file_path_remaps = nullptr;
  iree_host_size_t file_path_remap_count = 0;
  if (iree_status_is_ok(status)) {
    status = ParseReplayFileRemaps(host_allocator, &file_path_remaps,
                                   &file_path_remap_count);
  }

  ReplayExecutableSubstitutionState executable_substitutions;
  memset(&executable_substitutions, 0, sizeof(executable_substitutions));
  if (iree_status_is_ok(status)) {
    status = ParseReplayExecutableSubstitutions(host_allocator,
                                                &executable_substitutions);
  }

  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.file_path_remap_count = file_path_remap_count;
  options.file_path_remaps = file_path_remaps;
  if (executable_substitutions.entry_count != 0) {
    options.executable_substitution_callback.fn =
        ReplayExecutableSubstitutionCallback;
    options.executable_substitution_callback.user_data =
        &executable_substitutions;
  }

  iree_hal_replay_plan_t* replay_plan = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_plan_create(file_contents->const_buffer,
                                         host_allocator, &replay_plan);
  }

  if (iree_status_is_ok(status)) {
    iree_string_view_t replay_scope = iree_make_cstring_view(FLAG_replay_scope);
    std::string benchmark_name = "BM_replay";
    if (!iree_string_view_is_empty(replay_scope)) {
      benchmark_name.append("/scope:");
      benchmark_name.append(replay_scope.data, replay_scope.size);
    }
    benchmark::Benchmark* replay_benchmark =
        benchmark::RegisterBenchmark(benchmark_name,
                                     [=](benchmark::State& state) -> void {
                                       BenchmarkReplay(replay_plan,
                                                       device_group, profiling,
                                                       options, state);
                                     })
            ->MeasureProcessCPUTime()
            ->Unit(benchmark::kMillisecond);
    if (iree_string_view_is_empty(replay_scope)) {
      replay_benchmark->UseRealTime();
    } else {
      replay_benchmark->UseManualTime();
    }
    ::benchmark::RunSpecifiedBenchmarks();
  }

  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  iree_hal_replay_plan_destroy(replay_plan);
  ReleaseReplayExecutableSubstitutions(host_allocator,
                                       &executable_substitutions);
  iree_allocator_free(host_allocator, file_path_remaps);
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
