// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/fpu_state.h"
#include "iree/testing/benchmark.h"

#define ELEMENT_BUFFER_SIZE 2048

// Scales a buffer of floats by |scale| and disables autovectorization.
// Will generally be normal scalar floating point math and indicate whether the
// FPU has issues with denormals.
static float UnvectorizedScaleBufferByValue(float scale) {
  float buffer[ELEMENT_BUFFER_SIZE];
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffer); ++i) {
    buffer[i] = 1.0f;
  }
  iree_optimization_barrier(*buffer);
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffer); ++i) {
    buffer[i] *= scale;
    iree_optimization_barrier(buffer[i]);
  }
  iree_optimization_barrier(*buffer);
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
  float buffer[ELEMENT_BUFFER_SIZE];
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffer); ++i) {
    buffer[i] = 1.0f;
  }
  iree_optimization_barrier(*buffer);
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffer); ++i) {
    buffer[i] *= scale;
  }
  iree_optimization_barrier(*buffer);
  float sum = 0.0f;
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffer); ++i) {
    sum += buffer[i];
  }
  return sum;
}

IREE_BENCHMARK_FN(BM_UnvectorizedNormals) {
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_optimization_barrier(UnvectorizedScaleBufferByValue(1.0f));
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_UnvectorizedNormals);

IREE_BENCHMARK_FN(BM_UnvectorizedDenormals) {
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_optimization_barrier(UnvectorizedScaleBufferByValue(1e-39f));
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_UnvectorizedDenormals);

IREE_BENCHMARK_FN(BM_UnvectorizedDenormalsFlushedToZero) {
  iree_fpu_state_t fpu_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_optimization_barrier(UnvectorizedScaleBufferByValue(1e-39f));
  }
  iree_fpu_state_pop(fpu_state);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_UnvectorizedDenormalsFlushedToZero);

IREE_BENCHMARK_FN(BM_UnvectorizedDenormalsNotFlushedToZero) {
  iree_fpu_state_t fpu_state = iree_fpu_state_push(IREE_FPU_STATE_DEFAULT);
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_optimization_barrier(UnvectorizedScaleBufferByValue(1e-39f));
  }
  iree_fpu_state_pop(fpu_state);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_UnvectorizedDenormalsNotFlushedToZero);

IREE_BENCHMARK_FN(BM_VectorizedNormals) {
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_optimization_barrier(VectorizedScaleBufferByValue(1.0f));
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_VectorizedNormals);

IREE_BENCHMARK_FN(BM_VectorizedDenormals) {
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_optimization_barrier(VectorizedScaleBufferByValue(1e-39f));
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_VectorizedDenormals);

IREE_BENCHMARK_FN(BM_VectorizedDenormalsFlushedToZero) {
  iree_fpu_state_t fpu_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_optimization_barrier(VectorizedScaleBufferByValue(1e-39f));
  }
  iree_fpu_state_pop(fpu_state);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_VectorizedDenormalsFlushedToZero);

IREE_BENCHMARK_FN(BM_VectorizedDenormalsNotFlushedToZero) {
  iree_fpu_state_t fpu_state = iree_fpu_state_push(IREE_FPU_STATE_DEFAULT);
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_optimization_barrier(VectorizedScaleBufferByValue(1e-39f));
  }
  iree_fpu_state_pop(fpu_state);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_VectorizedDenormalsNotFlushedToZero);
