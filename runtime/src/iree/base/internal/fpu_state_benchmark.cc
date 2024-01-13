// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if !IREE_HAS_NOP_BENCHMARK_LIB

#include <cstddef>

#include "iree/base/api.h"
#include "iree/base/internal/fpu_state.h"
#include "iree/testing/benchmark_lib.h"

namespace {

constexpr size_t kElementBufferSize = 2048;

// Scales a buffer of floats by |scale| and disables autovectorization.
// Will generally be normal scalar floating point math and indicate whether the
// FPU has issues with denormals.
static float UnvectorizedScaleBufferByValue(float scale) {
  float buffer[kElementBufferSize];
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffer); ++i) {
    buffer[i] = 1.0f;
  }
  benchmark::DoNotOptimize(*buffer);
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffer); ++i) {
    buffer[i] *= scale;
    benchmark::DoNotOptimize(buffer[i]);
  }
  benchmark::DoNotOptimize(*buffer);
  float sum = 0.0f;
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffer); ++i) {
    sum += buffer[i];
  }
  return sum;
}

// Scales a buffer of floats by |scale| and allows autovectorization.
// Will generally be SIMD floating point math and indicate whether the vector
// units (NEON, AVX, etc) have issues with denormals.
static float VectorizedScaleBufferByValue(float scale) {
  float buffer[kElementBufferSize];
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffer); ++i) {
    buffer[i] = 1.0f;
  }
  benchmark::DoNotOptimize(*buffer);
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffer); ++i) {
    buffer[i] *= scale;
  }
  benchmark::DoNotOptimize(*buffer);
  float sum = 0.0f;
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffer); ++i) {
    sum += buffer[i];
  }
  return sum;
}

void BM_UnvectorizedNormals(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(UnvectorizedScaleBufferByValue(1.0f));
  }
}
BENCHMARK(BM_UnvectorizedNormals);

void BM_UnvectorizedDenormals(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(UnvectorizedScaleBufferByValue(1e-39f));
  }
}
BENCHMARK(BM_UnvectorizedDenormals);

void BM_UnvectorizedDenormalsFlushedToZero(benchmark::State& state) {
  iree_fpu_state_t fpu_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);
  for (auto _ : state) {
    benchmark::DoNotOptimize(UnvectorizedScaleBufferByValue(1e-39f));
  }
  iree_fpu_state_pop(fpu_state);
}
BENCHMARK(BM_UnvectorizedDenormalsFlushedToZero);

void BM_UnvectorizedDenormalsNotFlushedToZero(benchmark::State& state) {
  iree_fpu_state_t fpu_state = iree_fpu_state_push(IREE_FPU_STATE_DEFAULT);
  for (auto _ : state) {
    benchmark::DoNotOptimize(UnvectorizedScaleBufferByValue(1e-39f));
  }
  iree_fpu_state_pop(fpu_state);
}
BENCHMARK(BM_UnvectorizedDenormalsNotFlushedToZero);

void BM_VectorizedNormals(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(VectorizedScaleBufferByValue(1.0f));
  }
}
BENCHMARK(BM_VectorizedNormals);

void BM_VectorizedDenormals(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(VectorizedScaleBufferByValue(1e-39f));
  }
}
BENCHMARK(BM_VectorizedDenormals);

void BM_VectorizedDenormalsFlushedToZero(benchmark::State& state) {
  iree_fpu_state_t fpu_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);
  for (auto _ : state) {
    benchmark::DoNotOptimize(VectorizedScaleBufferByValue(1e-39f));
  }
  iree_fpu_state_pop(fpu_state);
}
BENCHMARK(BM_VectorizedDenormalsFlushedToZero);

void BM_VectorizedDenormalsNotFlushedToZero(benchmark::State& state) {
  iree_fpu_state_t fpu_state = iree_fpu_state_push(IREE_FPU_STATE_DEFAULT);
  for (auto _ : state) {
    benchmark::DoNotOptimize(VectorizedScaleBufferByValue(1e-39f));
  }
  iree_fpu_state_pop(fpu_state);
}
BENCHMARK(BM_VectorizedDenormalsNotFlushedToZero);

}  // namespace

#endif  // !IREE_HAS_NOP_BENCHMARK_LIB
