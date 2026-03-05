// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>
#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/base64.h"
#include "iree/testing/benchmark.h"

//===----------------------------------------------------------------------===//
// Benchmark Data Generation
//===----------------------------------------------------------------------===//

// Fills a buffer with deterministic pseudo-random bytes for reproducible
// benchmarks. Uses a simple LCG that produces a good mix of byte values.
static void FillTestData(uint8_t* data, iree_host_size_t length) {
  uint32_t state = 0xDEADBEEF;
  for (iree_host_size_t i = 0; i < length; ++i) {
    state = state * 1103515245 + 12345;
    data[i] = (uint8_t)(state >> 16);
  }
}

//===----------------------------------------------------------------------===//
// Encode Benchmarks
//===----------------------------------------------------------------------===//

// Benchmarks base64 encoding for a given input size. The encoded data is
// consumed via iree_optimization_barrier to prevent dead code elimination.
static iree_status_t BenchmarkEncode(iree_benchmark_state_t* benchmark_state,
                                     iree_host_size_t data_length) {
  std::vector<uint8_t> input(data_length);
  FillTestData(input.data(), data_length);

  iree_const_byte_span_t input_span =
      iree_make_const_byte_span(input.data(), input.size());
  iree_host_size_t encoded_size = iree_base64_encoded_size(data_length);
  std::vector<char> output(encoded_size);

  while (iree_benchmark_keep_running(benchmark_state, data_length)) {
    iree_host_size_t actual_length = 0;
    iree_status_t status = iree_base64_encode(
        input_span, iree_make_mutable_string_view(output.data(), output.size()),
        &actual_length);
    if (!iree_status_is_ok(status)) return status;
    iree_optimization_barrier(output[0]);
  }
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Encode_4B) { return BenchmarkEncode(benchmark_state, 4); }
IREE_BENCHMARK_REGISTER(BM_Encode_4B);

IREE_BENCHMARK_FN(BM_Encode_64B) {
  return BenchmarkEncode(benchmark_state, 64);
}
IREE_BENCHMARK_REGISTER(BM_Encode_64B);

IREE_BENCHMARK_FN(BM_Encode_256B) {
  return BenchmarkEncode(benchmark_state, 256);
}
IREE_BENCHMARK_REGISTER(BM_Encode_256B);

IREE_BENCHMARK_FN(BM_Encode_1KB) {
  return BenchmarkEncode(benchmark_state, 1024);
}
IREE_BENCHMARK_REGISTER(BM_Encode_1KB);

IREE_BENCHMARK_FN(BM_Encode_4KB) {
  return BenchmarkEncode(benchmark_state, 4096);
}
IREE_BENCHMARK_REGISTER(BM_Encode_4KB);

IREE_BENCHMARK_FN(BM_Encode_64KB) {
  return BenchmarkEncode(benchmark_state, 65536);
}
IREE_BENCHMARK_REGISTER(BM_Encode_64KB);

//===----------------------------------------------------------------------===//
// Decode Benchmarks
//===----------------------------------------------------------------------===//

// Benchmarks base64 decoding for a given original (pre-encoding) data size.
// Pre-encodes the data once, then benchmarks repeated decoding.
static iree_status_t BenchmarkDecode(iree_benchmark_state_t* benchmark_state,
                                     iree_host_size_t data_length) {
  // Generate and encode test data.
  std::vector<uint8_t> original(data_length);
  FillTestData(original.data(), data_length);

  iree_const_byte_span_t input =
      iree_make_const_byte_span(original.data(), original.size());
  iree_host_size_t encoded_size = iree_base64_encoded_size(data_length);
  std::vector<char> encoded(encoded_size);
  iree_host_size_t encode_length = 0;
  IREE_RETURN_IF_ERROR(iree_base64_encode(
      input, iree_make_mutable_string_view(encoded.data(), encoded.size()),
      &encode_length));

  iree_string_view_t encoded_view =
      iree_make_string_view(encoded.data(), encode_length);
  std::vector<uint8_t> output(data_length);

  while (iree_benchmark_keep_running(benchmark_state, data_length)) {
    iree_host_size_t actual_length = 0;
    iree_status_t status = iree_base64_decode(
        encoded_view, iree_make_byte_span(output.data(), output.size()),
        &actual_length);
    if (!iree_status_is_ok(status)) return status;
    iree_optimization_barrier(output[0]);
  }
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Decode_4B) { return BenchmarkDecode(benchmark_state, 4); }
IREE_BENCHMARK_REGISTER(BM_Decode_4B);

IREE_BENCHMARK_FN(BM_Decode_64B) {
  return BenchmarkDecode(benchmark_state, 64);
}
IREE_BENCHMARK_REGISTER(BM_Decode_64B);

IREE_BENCHMARK_FN(BM_Decode_256B) {
  return BenchmarkDecode(benchmark_state, 256);
}
IREE_BENCHMARK_REGISTER(BM_Decode_256B);

IREE_BENCHMARK_FN(BM_Decode_1KB) {
  return BenchmarkDecode(benchmark_state, 1024);
}
IREE_BENCHMARK_REGISTER(BM_Decode_1KB);

IREE_BENCHMARK_FN(BM_Decode_4KB) {
  return BenchmarkDecode(benchmark_state, 4096);
}
IREE_BENCHMARK_REGISTER(BM_Decode_4KB);

IREE_BENCHMARK_FN(BM_Decode_64KB) {
  return BenchmarkDecode(benchmark_state, 65536);
}
IREE_BENCHMARK_REGISTER(BM_Decode_64KB);

//===----------------------------------------------------------------------===//
// Roundtrip Benchmark
//===----------------------------------------------------------------------===//

// Benchmarks full encode + decode roundtrip to measure combined throughput.
static iree_status_t BenchmarkRoundtrip(iree_benchmark_state_t* benchmark_state,
                                        iree_host_size_t data_length) {
  std::vector<uint8_t> input(data_length);
  FillTestData(input.data(), data_length);

  iree_const_byte_span_t input_span =
      iree_make_const_byte_span(input.data(), input.size());
  iree_host_size_t encoded_size = iree_base64_encoded_size(data_length);
  std::vector<char> encoded(encoded_size);
  std::vector<uint8_t> output(data_length);

  while (iree_benchmark_keep_running(benchmark_state, data_length)) {
    iree_host_size_t encode_length = 0;
    iree_status_t status = iree_base64_encode(
        input_span,
        iree_make_mutable_string_view(encoded.data(), encoded.size()),
        &encode_length);
    if (!iree_status_is_ok(status)) return status;

    iree_string_view_t encoded_view =
        iree_make_string_view(encoded.data(), encode_length);
    iree_host_size_t decode_length = 0;
    status = iree_base64_decode(
        encoded_view, iree_make_byte_span(output.data(), output.size()),
        &decode_length);
    if (!iree_status_is_ok(status)) return status;
    iree_optimization_barrier(output[0]);
  }
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Roundtrip_4B) {
  return BenchmarkRoundtrip(benchmark_state, 4);
}
IREE_BENCHMARK_REGISTER(BM_Roundtrip_4B);

IREE_BENCHMARK_FN(BM_Roundtrip_64B) {
  return BenchmarkRoundtrip(benchmark_state, 64);
}
IREE_BENCHMARK_REGISTER(BM_Roundtrip_64B);

IREE_BENCHMARK_FN(BM_Roundtrip_1KB) {
  return BenchmarkRoundtrip(benchmark_state, 1024);
}
IREE_BENCHMARK_REGISTER(BM_Roundtrip_1KB);

IREE_BENCHMARK_FN(BM_Roundtrip_64KB) {
  return BenchmarkRoundtrip(benchmark_state, 65536);
}
IREE_BENCHMARK_REGISTER(BM_Roundtrip_64KB);
