// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Representative workload benchmarks for the task system.
//
// These benchmarks exercise realistic work patterns — compute-bound,
// memory-bound, imbalanced, and model-like dispatch chains — to complement
// the noop/trivial overhead benchmarks in dispatch_benchmark.cc.
//
// The noop benchmarks measure pure scheduling overhead. These benchmarks
// measure how scheduling decisions interact with real work: Does spinning
// help when tiles do 100us of compute? Does work-stealing help when tiles
// are imbalanced? Does the scheduler behave well with model-like dispatch
// patterns?
//
// Benchmarks:
//
//   Workload/Compute/{dispatches}x{tiles}/{workers}w/work{work_us}us:
//     Dispatch chain where each tile does configurable compute work
//     (PRNG iterations). Calibrated so work_us ≈ actual microseconds of
//     ALU-bound work per tile. This is the "pleasant case" where tiles
//     are uniform and compute-bound — scheduling overhead should be
//     fully hidden.
//
//   Workload/Memory/{dispatches}x{tiles}/{workers}w/{buffer_kb}KB:
//     Dispatch chain where each tile reads through a shared buffer with
//     cache-line stride. Buffer sizes span L1→DRAM to exercise different
//     memory hierarchy levels. Memory-bound tiles have less consistent
//     duration, testing how the scheduler handles variable tile completion.
//
//   Workload/Imbalanced/{dispatches}x{tiles}/{workers}w/skew{ratio}:
//     Dispatch chain where 1 in every 8 tiles gets ratio× the work of
//     other tiles. Creates straggler tiles that hold up barrier resolution
//     while most workers sit idle. This is the critical case for adaptive
//     spinning: idle workers must decide whether to spin (hoping for the
//     next dispatch) or park.
//
//   Workload/ModelLike/{pattern}/{workers}w:
//     Dispatch chains that mimic real model execution patterns — mix of
//     small dispatches (activation functions, norms) and large dispatches
//     (matmuls, attention). Shows whether scheduling optimizations help
//     real models or only synthetic benchmarks.
//
//   Workload/ComputeWithSpin/{dispatches}x{tiles}/{workers}w/work{work_us}us/spin{spin_us}us:
//     Compute chain with varying spin durations. The key experiment: at
//     what point does spinning stop helping as tile work increases? With
//     noop tiles, spinning hurts (baseline results). With real work, the
//     MiniLM data shows 30% improvement. Where's the crossover?

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/task/benchmarks/benchmark_base.h"

namespace iree::task::benchmarks {

//===----------------------------------------------------------------------===//
// Calibration
//===----------------------------------------------------------------------===//

// Approximate PRNG iterations per microsecond of ALU work on modern x86.
// PRNG chain: multiply + add with data dependency → ~4 cycles/iteration.
// At 4.5 GHz: ~1.1B iterations/second ≈ 1100 iterations/us.
// We round to 1000 for readable benchmark names. The absolute value doesn't
// matter — what matters is consistency across configurations.
static constexpr uintptr_t kIterationsPerMicrosecond = 1000;

//===----------------------------------------------------------------------===//
// Memory workload closure
//===----------------------------------------------------------------------===//

// Shared buffer for memory-bound workloads. Allocated once per benchmark
// function, accessed by all tiles across all dispatches.
struct MemoryWorkloadContext {
  uint8_t* buffer;
  size_t buffer_size;
  size_t accesses_per_tile;
};

// Dispatch closure that reads through a shared buffer with cache-line stride.
// Each tile starts at a different offset (based on workgroup index) to
// distribute pressure across the buffer rather than all tiles hitting the
// same cache lines.
static iree_status_t dispatch_closure_memory(
    void* user_context, const iree_task_tile_context_t* tile_context,
    iree_task_submission_t* pending_submission) {
  (void)pending_submission;
  auto* context = static_cast<MemoryWorkloadContext*>(user_context);
  // Each tile starts at a different offset to spread accesses.
  // Multiply by a prime to avoid aliasing patterns in the cache.
  constexpr size_t kCacheLineSize = 64;
  size_t offset = (tile_context->workgroup_xyz[0] * kCacheLineSize * 7919) %
                  context->buffer_size;
  volatile uint64_t accumulator = 0;
  for (size_t i = 0; i < context->accesses_per_tile; ++i) {
    accumulator += context->buffer[offset];
    offset = (offset + kCacheLineSize) % context->buffer_size;
  }
  benchmark_sink += accumulator;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Imbalanced workload closure
//===----------------------------------------------------------------------===//

// Every 8th tile gets ratio× the base work.
struct ImbalancedWorkloadContext {
  uintptr_t base_iterations;
  uintptr_t heavy_ratio;
};

//===----------------------------------------------------------------------===//
// Jittered workload closure
//===----------------------------------------------------------------------===//

// Each tile gets a random duration drawn from [base × (1-jitter), base ×
// (1+jitter)]. This models the natural variability in real workloads: memory
// access patterns, cache behavior, and tensor shape irregularity all create
// tile-to-tile variation.
//
// The jitter is deterministic per tile index (uses tile index as PRNG seed) so
// results are reproducible across iterations.
struct JitteredWorkloadContext {
  uintptr_t base_iterations;
  // Jitter magnitude: 0 = uniform, 50 = ±50%, 100 = 0-200%.
  // Stored as percentage (0-100).
  uintptr_t jitter_percent;
};

static iree_status_t dispatch_closure_jittered(
    void* user_context, const iree_task_tile_context_t* tile_context,
    iree_task_submission_t* pending_submission) {
  (void)pending_submission;
  auto* context = static_cast<JitteredWorkloadContext*>(user_context);
  // Deterministic per-tile jitter: hash the tile index to get a [-1, +1]
  // factor.
  uint32_t tile_index = tile_context->workgroup_xyz[0];
  // Simple hash to distribute across range.
  uint32_t hash = tile_index * 2654435761u;  // Knuth multiplicative hash.
  // Map hash to jitter factor: [0, 2^32) → [-jitter_percent, +jitter_percent].
  int32_t signed_hash = static_cast<int32_t>(hash);
  double jitter_factor =
      static_cast<double>(signed_hash) / static_cast<double>(INT32_MAX);
  double scale = 1.0 + (jitter_factor *
                        static_cast<double>(context->jitter_percent) / 100.0);
  if (scale < 0.05) scale = 0.05;  // Floor at 5% to avoid zero-work tiles.

  uintptr_t iterations = static_cast<uintptr_t>(
      static_cast<double>(context->base_iterations) * scale);

  volatile uint64_t accumulator = tile_index;
  for (uintptr_t i = 0; i < iterations; ++i) {
    accumulator = accumulator * 6364136223846793005ULL + 1442695040888963407ULL;
  }
  benchmark_sink += accumulator;
  return iree_ok_status();
}

static iree_status_t dispatch_closure_imbalanced(
    void* user_context, const iree_task_tile_context_t* tile_context,
    iree_task_submission_t* pending_submission) {
  (void)pending_submission;
  auto* context = static_cast<ImbalancedWorkloadContext*>(user_context);
  // Every 8th tile is a "straggler" that takes ratio× longer.
  uintptr_t iterations = context->base_iterations;
  if (tile_context->workgroup_xyz[0] % 8 == 0) {
    iterations *= context->heavy_ratio;
  }
  volatile uint64_t accumulator = tile_context->workgroup_xyz[0];
  for (uintptr_t i = 0; i < iterations; ++i) {
    accumulator = accumulator * 6364136223846793005ULL + 1442695040888963407ULL;
  }
  benchmark_sink += accumulator;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Compute-bound dispatch chain
//===----------------------------------------------------------------------===//

// Measures dispatch chain latency with compute-bound tiles.
//
// Args: [0] = dispatch_count, [1] = tile_count, [2] = worker_count,
//       [3] = work_us (approximate microseconds of work per tile)
static void BM_WorkloadCompute(::benchmark::State& state) {
  const int dispatch_count = static_cast<int>(state.range(0));
  const uint32_t tile_count = static_cast<uint32_t>(state.range(1));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(2));
  const uintptr_t work_us = static_cast<uintptr_t>(state.range(3));

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, state)) return;

  const uintptr_t iterations = work_us * kIterationsPerMicrosecond;
  iree_task_dispatch_closure_t closure = iree_task_make_dispatch_closure(
      dispatch_closure_work, reinterpret_cast<void*>(iterations));

  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);

  for (auto _ : state) {
    iree_task_t* head = nullptr;
    iree_task_t* tail = nullptr;
    BuildDispatchChain(&context.scope, dispatch_count, tile_count, closure,
                       dispatches.data(), &head, &tail);

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["tiles_per_dispatch"] = tile_count;
  state.counters["workers"] = static_cast<double>(worker_count);
  state.counters["work_us"] = static_cast<double>(work_us);
}

//===----------------------------------------------------------------------===//
// Memory-bound dispatch chain
//===----------------------------------------------------------------------===//

// Measures dispatch chain latency with memory-bound tiles.
//
// Each tile reads through a shared buffer with cache-line stride. Buffer
// sizes are chosen to exercise different memory hierarchy levels:
//   32KB  → fits in L1 (per-core), ~1ns/access
//   256KB → fits in L2 (per-core), ~3-5ns/access
//   4MB   → fits in L3 (shared), ~10-15ns/access
//   64MB  → exceeds L3, hits DRAM, ~50-100ns/access
//
// Args: [0] = dispatch_count, [1] = tile_count, [2] = worker_count,
//       [3] = buffer_kb
static void BM_WorkloadMemory(::benchmark::State& state) {
  const int dispatch_count = static_cast<int>(state.range(0));
  const uint32_t tile_count = static_cast<uint32_t>(state.range(1));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(2));
  const size_t buffer_kb = static_cast<size_t>(state.range(3));

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, state)) return;

  // Allocate and touch the buffer to ensure it's physically backed.
  MemoryWorkloadContext workload_context;
  workload_context.buffer_size = buffer_kb * 1024;
  workload_context.buffer =
      static_cast<uint8_t*>(malloc(workload_context.buffer_size));
  if (!workload_context.buffer) {
    state.SkipWithError("Buffer allocation failed");
    return;
  }
  memset(workload_context.buffer, 0xAB, workload_context.buffer_size);

  // ~500 cache line accesses per tile, giving ~50us/tile for DRAM-bound
  // and ~0.5us/tile for L1-bound.
  workload_context.accesses_per_tile = 500;

  iree_task_dispatch_closure_t closure = iree_task_make_dispatch_closure(
      dispatch_closure_memory, &workload_context);

  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);

  for (auto _ : state) {
    iree_task_t* head = nullptr;
    iree_task_t* tail = nullptr;
    BuildDispatchChain(&context.scope, dispatch_count, tile_count, closure,
                       dispatches.data(), &head, &tail);

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  free(workload_context.buffer);

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["tiles_per_dispatch"] = tile_count;
  state.counters["workers"] = static_cast<double>(worker_count);
  state.counters["buffer_kb"] = static_cast<double>(buffer_kb);
}

//===----------------------------------------------------------------------===//
// Imbalanced dispatch chain
//===----------------------------------------------------------------------===//

// Measures dispatch chain latency with non-uniform tile work.
//
// Every 8th tile (the "stragglers") does ratio× the work of other tiles.
// With 128 tiles and ratio=10, there are 16 heavy tiles and 112 light tiles.
// If each light tile is 1us and each heavy tile is 10us, the critical path
// through a dispatch is 10us — but 7/8 of workers finish in 1us and sit idle
// for 9us. This is exactly the scenario where adaptive spinning should shine:
// those idle workers could spin for the ~10us until the next dispatch arrives.
//
// Args: [0] = dispatch_count, [1] = tile_count, [2] = worker_count,
//       [3] = skew_ratio
static void BM_WorkloadImbalanced(::benchmark::State& state) {
  const int dispatch_count = static_cast<int>(state.range(0));
  const uint32_t tile_count = static_cast<uint32_t>(state.range(1));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(2));
  const uintptr_t skew_ratio = static_cast<uintptr_t>(state.range(3));

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, state)) return;

  // Base work: ~1us per light tile.
  ImbalancedWorkloadContext workload_context;
  workload_context.base_iterations = 1 * kIterationsPerMicrosecond;
  workload_context.heavy_ratio = skew_ratio;

  iree_task_dispatch_closure_t closure = iree_task_make_dispatch_closure(
      dispatch_closure_imbalanced, &workload_context);

  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);

  for (auto _ : state) {
    iree_task_t* head = nullptr;
    iree_task_t* tail = nullptr;
    BuildDispatchChain(&context.scope, dispatch_count, tile_count, closure,
                       dispatches.data(), &head, &tail);

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["tiles_per_dispatch"] = tile_count;
  state.counters["workers"] = static_cast<double>(worker_count);
  state.counters["skew_ratio"] = static_cast<double>(skew_ratio);
}

//===----------------------------------------------------------------------===//
// Model-like dispatch chain
//===----------------------------------------------------------------------===//

// A dispatch in the model-like benchmark pattern.
struct ModelDispatchSpec {
  uint32_t tile_count;
  uintptr_t work_us;  // Approximate work per tile in microseconds.
};

// Simplified transformer block dispatch pattern.
// A single transformer block has roughly this structure:
//   - LayerNorm: few tiles, light work (~1us/tile)
//   - QKV projection: many tiles, heavy compute (~50us/tile)
//   - Attention: moderate tiles, moderate work (~20us/tile)
//   - Output projection: many tiles, heavy compute (~50us/tile)
//   - Residual add: few tiles, light work (~1us/tile)
//   - LayerNorm: few tiles, light work (~1us/tile)
//   - FFN up-projection: many tiles, heavy compute (~50us/tile)
//   - Activation (GELU): moderate tiles, light work (~5us/tile)
//   - FFN down-projection: many tiles, heavy compute (~50us/tile)
//   - Residual add: few tiles, light work (~1us/tile)
//
// We repeat this for multiple layers, varying tile counts slightly to avoid
// perfect caching behavior.
static const ModelDispatchSpec kTransformerBlock[] = {
    {16, 1},    // LayerNorm
    {128, 50},  // QKV projection
    {64, 20},   // Attention scores
    {64, 20},   // Attention softmax + values
    {128, 50},  // Output projection
    {16, 1},    // Residual add
    {16, 1},    // LayerNorm
    {128, 50},  // FFN up
    {64, 5},    // GELU activation
    {128, 50},  // FFN down
    {16, 1},    // Residual add
};
static constexpr int kTransformerBlockSize =
    sizeof(kTransformerBlock) / sizeof(kTransformerBlock[0]);

// Small transformer: 4 layers = 44 dispatches.
// Approximate total: 4 × (~11ms) = ~44ms single-threaded.
// With 8 workers: ~5-6ms expected.
static const ModelDispatchSpec kSmallTransformer[] = {
    // Embedding lookup.
    {32, 5},
    // 4 transformer blocks (with slight tile count variation per layer).
    {16, 1},
    {128, 50},
    {64, 20},
    {64, 20},
    {128, 50},
    {16, 1},
    {16, 1},
    {128, 50},
    {64, 5},
    {128, 50},
    {16, 1},
    {16, 1},
    {96, 50},
    {48, 20},
    {48, 20},
    {96, 50},
    {16, 1},
    {16, 1},
    {96, 50},
    {48, 5},
    {96, 50},
    {16, 1},
    {16, 1},
    {128, 45},
    {64, 18},
    {64, 18},
    {128, 45},
    {16, 1},
    {16, 1},
    {128, 45},
    {64, 5},
    {128, 45},
    {16, 1},
    {16, 1},
    {96, 45},
    {48, 18},
    {48, 18},
    {96, 45},
    {16, 1},
    {16, 1},
    {96, 45},
    {48, 5},
    {96, 45},
    {16, 1},
    // Final LayerNorm + classification head.
    {16, 1},
    {32, 10},
};
static constexpr int kSmallTransformerSize =
    sizeof(kSmallTransformer) / sizeof(kSmallTransformer[0]);

// Measures a model-like dispatch chain with heterogeneous dispatch sizes.
//
// Args: [0] = pattern (0 = single transformer block, 1 = small transformer),
//       [1] = worker_count
static void BM_WorkloadModelLike(::benchmark::State& state) {
  const int pattern = static_cast<int>(state.range(0));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(1));

  const ModelDispatchSpec* specs = nullptr;
  int dispatch_count = 0;
  const char* pattern_name = nullptr;
  switch (pattern) {
    case 0:
      specs = kTransformerBlock;
      dispatch_count = kTransformerBlockSize;
      pattern_name = "TransformerBlock";
      break;
    case 1:
      specs = kSmallTransformer;
      dispatch_count = kSmallTransformerSize;
      pattern_name = "SmallTransformer";
      break;
    default:
      state.SkipWithError("Unknown pattern");
      return;
  }

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, state)) return;

  // Build dispatch chain with per-dispatch closures.
  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);
  const uint32_t workgroup_size[3] = {1, 1, 1};

  for (auto _ : state) {
    // Initialize each dispatch with its specific tile count and work closure.
    for (int i = 0; i < dispatch_count; ++i) {
      uintptr_t iterations = specs[i].work_us * kIterationsPerMicrosecond;
      iree_task_dispatch_closure_t closure = iree_task_make_dispatch_closure(
          dispatch_closure_work, reinterpret_cast<void*>(iterations));
      const uint32_t workgroup_count[3] = {specs[i].tile_count, 1, 1};
      iree_task_dispatch_initialize(&context.scope, closure, workgroup_size,
                                    workgroup_count, &dispatches[i]);
    }

    // Chain dispatches directly.
    for (int i = 0; i < dispatch_count - 1; ++i) {
      iree_task_set_completion_task(&dispatches[i].header,
                                    &dispatches[i + 1].header);
    }

    iree_task_t* head = &dispatches[0].header;
    iree_task_t* tail = &dispatches[dispatch_count - 1].header;

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["workers"] = static_cast<double>(worker_count);
  state.SetLabel(pattern_name);
}

//===----------------------------------------------------------------------===//
// Compute chain with spin (crossover experiment)
//===----------------------------------------------------------------------===//

// The key experiment: finding the crossover point where spinning helps.
//
// With noop tiles (baseline results), spinning hurts — inter-dispatch gaps
// are ~10us, so any spin budget > 10us is wasted CPU. With MiniLM (sweep.md),
// spinning saves 30% because inter-dispatch gaps are ~200-400us and tiles do
// real work.
//
// This benchmark fills in the middle: at what tile work duration does spinning
// start to help? This directly informs the adaptive spinning algorithm's
// break-even analysis.
//
// Args: [0] = dispatch_count, [1] = tile_count, [2] = worker_count,
//       [3] = work_us, [4] = spin_us
static void BM_WorkloadComputeWithSpin(::benchmark::State& state) {
  const int dispatch_count = static_cast<int>(state.range(0));
  const uint32_t tile_count = static_cast<uint32_t>(state.range(1));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(2));
  const uintptr_t work_us = static_cast<uintptr_t>(state.range(3));
  const iree_duration_t spin_ns = state.range(4) * 1000;  // us → ns

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, spin_ns, state)) return;

  const uintptr_t iterations = work_us * kIterationsPerMicrosecond;
  iree_task_dispatch_closure_t closure = iree_task_make_dispatch_closure(
      dispatch_closure_work, reinterpret_cast<void*>(iterations));

  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);

  for (auto _ : state) {
    iree_task_t* head = nullptr;
    iree_task_t* tail = nullptr;
    BuildDispatchChain(&context.scope, dispatch_count, tile_count, closure,
                       dispatches.data(), &head, &tail);

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["tiles_per_dispatch"] = tile_count;
  state.counters["workers"] = static_cast<double>(worker_count);
  state.counters["work_us"] = static_cast<double>(work_us);
  state.counters["spin_us"] = state.range(4);
}

//===----------------------------------------------------------------------===//
// Imbalanced chain with spin
//===----------------------------------------------------------------------===//

// Tests spinning with imbalanced workloads — the case that most benefits
// from adaptive spinning. Workers that finish early can either spin (hoping
// for next dispatch) or park. The right answer depends on the straggler
// duration: if the trailing tile takes 10us, spinning for 5us catches it.
// If it takes 100us, spinning for 5us wastes CPU.
//
// Args: [0] = dispatch_count, [1] = tile_count, [2] = worker_count,
//       [3] = skew_ratio, [4] = spin_us
static void BM_WorkloadImbalancedWithSpin(::benchmark::State& state) {
  const int dispatch_count = static_cast<int>(state.range(0));
  const uint32_t tile_count = static_cast<uint32_t>(state.range(1));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(2));
  const uintptr_t skew_ratio = static_cast<uintptr_t>(state.range(3));
  const iree_duration_t spin_ns = state.range(4) * 1000;

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, spin_ns, state)) return;

  ImbalancedWorkloadContext workload_context;
  workload_context.base_iterations = 1 * kIterationsPerMicrosecond;
  workload_context.heavy_ratio = skew_ratio;

  iree_task_dispatch_closure_t closure = iree_task_make_dispatch_closure(
      dispatch_closure_imbalanced, &workload_context);

  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);

  for (auto _ : state) {
    iree_task_t* head = nullptr;
    iree_task_t* tail = nullptr;
    BuildDispatchChain(&context.scope, dispatch_count, tile_count, closure,
                       dispatches.data(), &head, &tail);

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["tiles_per_dispatch"] = tile_count;
  state.counters["workers"] = static_cast<double>(worker_count);
  state.counters["skew_ratio"] = static_cast<double>(skew_ratio);
  state.counters["spin_us"] = state.range(4);
}

//===----------------------------------------------------------------------===//
// Jittered dispatch chain
//===----------------------------------------------------------------------===//

// The key experiment for reproducing MiniLM's 30% spin benefit.
//
// Real model tiles have naturally variable durations due to cache behavior,
// memory access patterns, and tensor shape irregularity. Uniform synthetic
// tiles miss this entirely — all workers finish simultaneously, no gap for
// spinning to help.
//
// This benchmark adds controlled jitter: each tile's work varies by ±jitter%
// around the base duration. With 50% jitter and 50us base, tiles range from
// 25-75us. The trailing tile creates a barrier gap that matches real model
// behavior.
//
// Args: [0] = dispatch_count, [1] = tile_count, [2] = worker_count,
//       [3] = work_us, [4] = jitter_percent, [5] = spin_us
static void BM_WorkloadJitteredWithSpin(::benchmark::State& state) {
  const int dispatch_count = static_cast<int>(state.range(0));
  const uint32_t tile_count = static_cast<uint32_t>(state.range(1));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(2));
  const uintptr_t work_us = static_cast<uintptr_t>(state.range(3));
  const uintptr_t jitter_percent = static_cast<uintptr_t>(state.range(4));
  const iree_duration_t spin_ns = state.range(5) * 1000;

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, spin_ns, state)) return;

  JitteredWorkloadContext workload_context;
  workload_context.base_iterations = work_us * kIterationsPerMicrosecond;
  workload_context.jitter_percent = jitter_percent;

  iree_task_dispatch_closure_t closure = iree_task_make_dispatch_closure(
      dispatch_closure_jittered, &workload_context);

  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);

  for (auto _ : state) {
    iree_task_t* head = nullptr;
    iree_task_t* tail = nullptr;
    BuildDispatchChain(&context.scope, dispatch_count, tile_count, closure,
                       dispatches.data(), &head, &tail);

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["tiles_per_dispatch"] = tile_count;
  state.counters["workers"] = static_cast<double>(worker_count);
  state.counters["work_us"] = static_cast<double>(work_us);
  state.counters["jitter_pct"] = static_cast<double>(jitter_percent);
  state.counters["spin_us"] = state.range(5);
}

//===----------------------------------------------------------------------===//
// Model-like chain with spin
//===----------------------------------------------------------------------===//

// The ultimate experiment: does spinning help a model-like workload?
// This should reproduce the MiniLM sweep.md results synthetically.
//
// Args: [0] = pattern, [1] = worker_count, [2] = spin_us
static void BM_WorkloadModelLikeWithSpin(::benchmark::State& state) {
  const int pattern = static_cast<int>(state.range(0));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(1));
  const iree_duration_t spin_ns = state.range(2) * 1000;

  const ModelDispatchSpec* specs = nullptr;
  int dispatch_count = 0;
  const char* pattern_name = nullptr;
  switch (pattern) {
    case 0:
      specs = kTransformerBlock;
      dispatch_count = kTransformerBlockSize;
      pattern_name = "TransformerBlock";
      break;
    case 1:
      specs = kSmallTransformer;
      dispatch_count = kSmallTransformerSize;
      pattern_name = "SmallTransformer";
      break;
    default:
      state.SkipWithError("Unknown pattern");
      return;
  }

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, spin_ns, state)) return;

  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);
  const uint32_t workgroup_size[3] = {1, 1, 1};

  for (auto _ : state) {
    for (int i = 0; i < dispatch_count; ++i) {
      uintptr_t iterations = specs[i].work_us * kIterationsPerMicrosecond;
      iree_task_dispatch_closure_t closure = iree_task_make_dispatch_closure(
          dispatch_closure_work, reinterpret_cast<void*>(iterations));
      const uint32_t workgroup_count[3] = {specs[i].tile_count, 1, 1};
      iree_task_dispatch_initialize(&context.scope, closure, workgroup_size,
                                    workgroup_count, &dispatches[i]);
    }

    for (int i = 0; i < dispatch_count - 1; ++i) {
      iree_task_set_completion_task(&dispatches[i].header,
                                    &dispatches[i + 1].header);
    }

    iree_task_t* head = &dispatches[0].header;
    iree_task_t* tail = &dispatches[dispatch_count - 1].header;

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["workers"] = static_cast<double>(worker_count);
  state.counters["spin_us"] = state.range(2);
  state.SetLabel(pattern_name);
}

//===----------------------------------------------------------------------===//
// Benchmark registration
//===----------------------------------------------------------------------===//

static void RegisterWorkloadBenchmarks() {
  // --- Compute-bound dispatch chains ---
  // Sweep work_us to understand scheduling overhead relative to work.
  for (int work_us : {1, 10, 50, 100}) {
    for (int dispatches : {16, 64}) {
      for (int tiles : {32, 128}) {
        for (int workers : {1, 4, 8}) {
          std::string name = "Workload/Compute/" + std::to_string(dispatches) +
                             "x" + std::to_string(tiles) + "/" +
                             std::to_string(workers) + "w/work" +
                             std::to_string(work_us) + "us";
          ::benchmark::RegisterBenchmark(
              name.c_str(),
              [](::benchmark::State& state) { BM_WorkloadCompute(state); })
              ->Args({dispatches, tiles, workers, work_us})
              ->Unit(::benchmark::kMicrosecond)
              ->MeasureProcessCPUTime()
              ->UseRealTime();
        }
      }
    }
  }

  // --- Memory-bound dispatch chains ---
  // Sweep buffer size from L1 to DRAM.
  for (int buffer_kb : {32, 256, 4096, 65536}) {
    for (int workers : {1, 4, 8}) {
      std::string name = "Workload/Memory/16x128/" + std::to_string(workers) +
                         "w/" + std::to_string(buffer_kb) + "KB";
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [](::benchmark::State& state) { BM_WorkloadMemory(state); })
          ->Args({16, 128, workers, buffer_kb})
          ->Unit(::benchmark::kMicrosecond)
          ->MeasureProcessCPUTime()
          ->UseRealTime();
    }
  }

  // --- Imbalanced dispatch chains ---
  // Sweep skew ratio to find work-stealing effectiveness boundaries.
  for (int skew_ratio : {2, 5, 10, 50}) {
    for (int workers : {1, 4, 8}) {
      std::string name = "Workload/Imbalanced/16x128/" +
                         std::to_string(workers) + "w/skew" +
                         std::to_string(skew_ratio);
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [](::benchmark::State& state) { BM_WorkloadImbalanced(state); })
          ->Args({16, 128, workers, skew_ratio})
          ->Unit(::benchmark::kMicrosecond)
          ->MeasureProcessCPUTime()
          ->UseRealTime();
    }
  }

  // --- Model-like dispatch chains ---
  for (int pattern : {0, 1}) {
    for (int workers : {1, 4, 8, 16, 32}) {
      const char* pattern_name =
          pattern == 0 ? "TransformerBlock" : "SmallTransformer";
      std::string name = "Workload/ModelLike/" + std::string(pattern_name) +
                         "/" + std::to_string(workers) + "w";
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [](::benchmark::State& state) { BM_WorkloadModelLike(state); })
          ->Args({pattern, workers})
          ->Unit(::benchmark::kMicrosecond)
          ->MeasureProcessCPUTime()
          ->UseRealTime();
    }
  }

  // --- Compute chain with spin (crossover experiment) ---
  // Fixed: 16 dispatches × 128 tiles. Vary work_us and spin_us.
  // Looking for the crossover: at what work_us does spinning start helping?
  for (int work_us : {1, 5, 10, 50, 100}) {
    for (int spin_us : {0, 10, 50, 100, 500}) {
      for (int workers : {4, 8}) {
        std::string name = "Workload/ComputeWithSpin/16x128/" +
                           std::to_string(workers) + "w/work" +
                           std::to_string(work_us) + "us/spin" +
                           std::to_string(spin_us) + "us";
        ::benchmark::RegisterBenchmark(name.c_str(),
                                       [](::benchmark::State& state) {
                                         BM_WorkloadComputeWithSpin(state);
                                       })
            ->Args({16, 128, workers, work_us, spin_us})
            ->Unit(::benchmark::kMicrosecond)
            ->MeasureProcessCPUTime()
            ->UseRealTime();
      }
    }
  }

  // --- Imbalanced chain with spin ---
  // Fixed: 16 dispatches × 128 tiles, skew=10.
  // Shows whether spinning helps when work is unevenly distributed.
  for (int spin_us : {0, 10, 50, 100, 500}) {
    for (int workers : {4, 8}) {
      std::string name = "Workload/ImbalancedWithSpin/16x128/" +
                         std::to_string(workers) + "w/skew10/spin" +
                         std::to_string(spin_us) + "us";
      ::benchmark::RegisterBenchmark(name.c_str(),
                                     [](::benchmark::State& state) {
                                       BM_WorkloadImbalancedWithSpin(state);
                                     })
          ->Args({16, 128, workers, 10, spin_us})
          ->Unit(::benchmark::kMicrosecond)
          ->MeasureProcessCPUTime()
          ->UseRealTime();
    }
  }

  // --- Jittered chain with spin (the MiniLM reproduction experiment) ---
  // This is the key experiment: does tile duration variance create the same
  // spin benefit seen in MiniLM? If jitter=50% at 50us base reproduces a
  // 20-30% spin benefit, we've identified the mechanism.
  for (int jitter_pct : {0, 25, 50, 75}) {
    for (int spin_us : {0, 50, 100, 500}) {
      for (int workers : {4, 8, 32}) {
        std::string name = "Workload/JitteredWithSpin/16x128/" +
                           std::to_string(workers) + "w/work50us/jitter" +
                           std::to_string(jitter_pct) + "pct/spin" +
                           std::to_string(spin_us) + "us";
        ::benchmark::RegisterBenchmark(name.c_str(),
                                       [](::benchmark::State& state) {
                                         BM_WorkloadJitteredWithSpin(state);
                                       })
            ->Args({16, 128, workers, 50, jitter_pct, spin_us})
            ->Unit(::benchmark::kMicrosecond)
            ->MeasureProcessCPUTime()
            ->UseRealTime();
      }
    }
  }

  // --- Model-like chain with spin (the MiniLM reproduction experiment) ---
  // This should show similar trends to sweep.md: spinning saves 20-30%
  // for model-like workloads. If it doesn't, our synthetic pattern doesn't
  // match real model behavior and needs adjustment.
  for (int pattern : {0, 1}) {
    for (int spin_us : {0, 50, 100, 500, 1000}) {
      for (int workers : {4, 8, 16, 32}) {
        const char* pattern_name =
            pattern == 0 ? "TransformerBlock" : "SmallTransformer";
        std::string name =
            "Workload/ModelLikeWithSpin/" + std::string(pattern_name) + "/" +
            std::to_string(workers) + "w/spin" + std::to_string(spin_us) + "us";
        ::benchmark::RegisterBenchmark(name.c_str(),
                                       [](::benchmark::State& state) {
                                         BM_WorkloadModelLikeWithSpin(state);
                                       })
            ->Args({pattern, workers, spin_us})
            ->Unit(::benchmark::kMicrosecond)
            ->MeasureProcessCPUTime()
            ->UseRealTime();
      }
    }
  }
}

static bool workload_benchmarks_registered_ =
    (RegisterWorkloadBenchmarks(), true);

}  // namespace iree::task::benchmarks
