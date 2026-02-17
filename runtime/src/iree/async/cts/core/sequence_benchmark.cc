// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS benchmarks for SEQUENCE operation overhead.
//
// Measures the per-step overhead of the LINK and emulation execution paths
// relative to raw linked operations. Key comparisons:
//
//   - LINK path vs raw LINKED batch: overhead of the link trampoline's
//     completion counting (SAW_ERROR check, current_step increment, final
//     status selection).
//   - Emulation path vs LINK path: overhead of step_fn dispatch and per-step
//     re-submission through the proactor vtable.
//   - step_fn overhead: cost of the step_fn call itself (isolated from
//     re-submission overhead by comparing emulation-with-step_fn vs
//     emulation-without, but since emulation requires step_fn, we compare
//     against the LINK path).

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/cts/util/benchmark_base.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Trivial step_fn that returns OK. Used to measure the overhead of the
// step_fn dispatch mechanism itself.
static iree_status_t TrivialStepFn(void* user_data,
                                   iree_async_operation_t* completed_step,
                                   iree_async_operation_t* next_step) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// BM_SequenceLinkNops
//===----------------------------------------------------------------------===//

// N NOP steps submitted as a SEQUENCE via the LINK path (step_fn == NULL).
// Measures the per-step overhead of the link trampoline's completion counting:
// error capture, CANCEL_REQUESTED check, current_step increment, and final
// status determination.
//
// Expected: O(1) per step, total time linear in N.
static void BM_SequenceLinkNops(::benchmark::State& state,
                                const ProactorFactory& factory) {
  size_t step_count = static_cast<size_t>(state.range(0));
  auto* context = CreateBenchmarkContext(factory, state);
  if (!context) return;

  // Allocate NOP operations and step pointer array.
  std::vector<iree_async_nop_operation_t> nops(step_count);
  std::vector<iree_async_operation_t*> steps(step_count);

  for (auto _ : state) {
    // Re-initialize NOPs each iteration (trampolines overwrite callbacks).
    for (size_t i = 0; i < step_count; ++i) {
      memset(&nops[i], 0, sizeof(nops[i]));
      nops[i].base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
      steps[i] = &nops[i].base;
    }

    iree_async_sequence_operation_t sequence;
    memset(&sequence, 0, sizeof(sequence));
    sequence.base.type = IREE_ASYNC_OPERATION_TYPE_SEQUENCE;
    sequence.base.completion_fn = BenchmarkContext::Callback;
    sequence.base.user_data = context;
    sequence.steps = steps.data();
    sequence.step_count = step_count;
    sequence.step_fn = nullptr;  // LINK path.

    context->Reset();
    int expected = context->completions.load(std::memory_order_acquire) + 1;

    auto start = std::chrono::high_resolution_clock::now();
    iree_status_t status =
        iree_async_proactor_submit_one(context->proactor, &sequence.base);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Submit failed");
      iree_status_ignore(status);
      break;
    }
    if (!context->SpinPollUntilComplete(expected)) {
      state.SkipWithError("Poll timed out");
      break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(step_count));
  DestroyBenchmarkContext(context);
}

//===----------------------------------------------------------------------===//
// BM_SequenceEmulationNops
//===----------------------------------------------------------------------===//

// N NOP steps submitted as a SEQUENCE via the emulation path (step_fn set).
// Measures the per-step overhead of the emulation trampoline: step_fn call,
// CANCEL_REQUESTED checks, and per-step re-submission through the proactor.
//
// Expected: higher per-step overhead than LINK path due to re-submission and
// step_fn dispatch. Each step requires a poll round-trip.
static void BM_SequenceEmulationNops(::benchmark::State& state,
                                     const ProactorFactory& factory) {
  size_t step_count = static_cast<size_t>(state.range(0));
  auto* context = CreateBenchmarkContext(factory, state);
  if (!context) return;

  std::vector<iree_async_nop_operation_t> nops(step_count);
  std::vector<iree_async_operation_t*> steps(step_count);

  for (auto _ : state) {
    for (size_t i = 0; i < step_count; ++i) {
      memset(&nops[i], 0, sizeof(nops[i]));
      nops[i].base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
      steps[i] = &nops[i].base;
    }

    iree_async_sequence_operation_t sequence;
    memset(&sequence, 0, sizeof(sequence));
    sequence.base.type = IREE_ASYNC_OPERATION_TYPE_SEQUENCE;
    sequence.base.completion_fn = BenchmarkContext::Callback;
    sequence.base.user_data = context;
    sequence.steps = steps.data();
    sequence.step_count = step_count;
    sequence.step_fn = TrivialStepFn;  // Emulation path.

    context->Reset();
    int expected = context->completions.load(std::memory_order_acquire) + 1;

    auto start = std::chrono::high_resolution_clock::now();
    iree_status_t status =
        iree_async_proactor_submit_one(context->proactor, &sequence.base);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Submit failed");
      iree_status_ignore(status);
      break;
    }
    if (!context->SpinPollUntilComplete(expected)) {
      state.SkipWithError("Poll timed out");
      break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(step_count));
  DestroyBenchmarkContext(context);
}

//===----------------------------------------------------------------------===//
// BM_SequenceVsRawLinked
//===----------------------------------------------------------------------===//

// N NOP steps submitted as a raw LINKED batch (no SEQUENCE wrapper).
// Provides the baseline for measuring the overhead of the SEQUENCE abstraction.
// Compare against BM_SequenceLinkNops to see the cost of the link trampolines.
static void BM_SequenceVsRawLinked(::benchmark::State& state,
                                   const ProactorFactory& factory) {
  size_t step_count = static_cast<size_t>(state.range(0));
  auto* context = CreateBenchmarkContext(factory, state);
  if (!context) return;

  if (!(context->capabilities &
        IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS)) {
    state.SkipWithError("Backend lacks LINKED_OPERATIONS capability");
    DestroyBenchmarkContext(context);
    return;
  }

  std::vector<iree_async_nop_operation_t> nops(step_count);
  std::vector<iree_async_operation_t*> steps(step_count);

  for (auto _ : state) {
    for (size_t i = 0; i < step_count; ++i) {
      memset(&nops[i], 0, sizeof(nops[i]));
      nops[i].base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
      nops[i].base.completion_fn = BenchmarkContext::Callback;
      nops[i].base.user_data = context;
      if (i + 1 < step_count) {
        nops[i].base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
      }
      steps[i] = &nops[i].base;
    }

    context->Reset();
    int expected = static_cast<int>(step_count);

    auto start = std::chrono::high_resolution_clock::now();
    iree_async_operation_list_t list = {steps.data(), step_count};
    iree_status_t status = iree_async_proactor_submit(context->proactor, list);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Submit failed");
      iree_status_ignore(status);
      break;
    }
    if (!context->SpinPollUntilComplete(expected)) {
      state.SkipWithError("Poll timed out");
      break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(step_count));
  DestroyBenchmarkContext(context);
}

//===----------------------------------------------------------------------===//
// BM_StepFnOverhead
//===----------------------------------------------------------------------===//

// Measures step_fn dispatch overhead by comparing emulation path (with trivial
// step_fn) to LINK path (no step_fn). This isolates the cost of the step_fn
// mechanism: the function pointer call, the two CANCEL_REQUESTED checks, and
// the per-step re-submission through the proactor vtable.
//
// Since these use different code paths, the comparison is approximate — the
// difference includes both step_fn call overhead and the re-submission cost.
// BM_SequenceLinkNops and BM_SequenceEmulationNops show the raw numbers;
// this benchmark presents them side-by-side.
static void BM_StepFnOverhead(::benchmark::State& state,
                              const ProactorFactory& factory) {
  // This is the emulation path (same as BM_SequenceEmulationNops).
  // Register alongside BM_SequenceLinkNops for side-by-side comparison.
  BM_SequenceEmulationNops(state, factory);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

class SequenceBenchmarks {
 public:
  static void RegisterBenchmarks(const char* prefix,
                                 const ProactorFactory& factory) {
    // LINK path: N NOP steps via submit_as_linked.
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/SequenceLinkNops").c_str(),
        [factory](::benchmark::State& state) {
          BM_SequenceLinkNops(state, factory);
        })
        ->Range(1, 64)
        ->Unit(::benchmark::kMicrosecond)
        ->UseManualTime();

    // Emulation path: N NOP steps via emulation_begin with trivial step_fn.
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/SequenceEmulationNops").c_str(),
        [factory](::benchmark::State& state) {
          BM_SequenceEmulationNops(state, factory);
        })
        ->Range(1, 64)
        ->Unit(::benchmark::kMicrosecond)
        ->UseManualTime();

    // Raw LINKED baseline: N NOPs as raw LINKED batch (no SEQUENCE wrapper).
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/SequenceVsRawLinked").c_str(),
        [factory](::benchmark::State& state) {
          BM_SequenceVsRawLinked(state, factory);
        })
        ->Range(1, 64)
        ->Unit(::benchmark::kMicrosecond)
        ->UseManualTime();

    // Step_fn overhead: emulation path for side-by-side with LINK path.
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/StepFnOverhead").c_str(),
        [factory](::benchmark::State& state) {
          BM_StepFnOverhead(state, factory);
        })
        ->Range(1, 64)
        ->Unit(::benchmark::kMicrosecond)
        ->UseManualTime();
  }
};

CTS_REGISTER_BENCHMARK_SUITE(SequenceBenchmarks);

}  // namespace iree::async::cts
