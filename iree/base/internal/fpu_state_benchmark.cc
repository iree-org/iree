// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "benchmark/benchmark.h"
#include "iree/base/internal/fpu_state.h"

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
