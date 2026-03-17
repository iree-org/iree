// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <memory>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/net/codec/aead/codec.h"
#include "monocypher.h"

namespace {

// Test PSK.
static const uint8_t kTestPSK[IREE_NET_AEAD_KEY_SIZE] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
    0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
    0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
};

// RAII wrapper for codec.
struct CodecDeleter {
  void operator()(iree_net_codec_t* codec) const {
    if (codec) iree_net_codec_release(codec);
  }
};
using CodecPtr = std::unique_ptr<iree_net_codec_t, CodecDeleter>;

// Creates a codec with the test PSK.
static CodecPtr CreateCodec(iree_net_aead_role_t role) {
  iree_const_byte_span_t psk = {kTestPSK, sizeof(kTestPSK)};
  iree_net_codec_t* codec = nullptr;
  IREE_CHECK_OK(
      iree_net_aead_codec_create(psk, role, iree_allocator_system(), &codec));
  return CodecPtr(codec);
}

//===----------------------------------------------------------------------===//
// AEAD Codec Benchmarks
//===----------------------------------------------------------------------===//

static void BM_AEADEncode(benchmark::State& state) {
  const size_t payload_size = static_cast<size_t>(state.range(0));

  CodecPtr codec = CreateCodec(IREE_NET_AEAD_ROLE_CLIENT);

  // Allocate frame buffer.
  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(codec.get());
  size_t frame_size = overhead.prefix + payload_size + overhead.suffix;
  std::vector<uint8_t> frame(frame_size);
  uint8_t* payload_ptr = frame.data() + overhead.prefix;

  // Fill with pattern.
  for (size_t i = 0; i < payload_size; ++i) {
    payload_ptr[i] = static_cast<uint8_t>(i & 0xFF);
  }

  for (auto _ : state) {
    IREE_CHECK_OK(
        iree_net_codec_encode(codec.get(), payload_ptr, payload_size));
    benchmark::DoNotOptimize(frame.data());
  }

  state.SetBytesProcessed(state.iterations() * payload_size);
}
BENCHMARK(BM_AEADEncode)->Arg(64)->Arg(1024)->Arg(64 * 1024)->Arg(1024 * 1024);

static void BM_AEADDecode(benchmark::State& state) {
  const size_t payload_size = static_cast<size_t>(state.range(0));

  CodecPtr client = CreateCodec(IREE_NET_AEAD_ROLE_CLIENT);
  CodecPtr server = CreateCodec(IREE_NET_AEAD_ROLE_SERVER);

  // Allocate frame buffer.
  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  size_t frame_size = overhead.prefix + payload_size + overhead.suffix;
  std::vector<uint8_t> frame(frame_size);
  uint8_t* payload_ptr = frame.data() + overhead.prefix;

  // Fill with pattern.
  for (size_t i = 0; i < payload_size; ++i) {
    payload_ptr[i] = static_cast<uint8_t>(i & 0xFF);
  }

  for (auto _ : state) {
    state.PauseTiming();
    // Re-encode for each decode (nonce changes).
    IREE_CHECK_OK(
        iree_net_codec_encode(client.get(), payload_ptr, payload_size));
    state.ResumeTiming();

    iree_byte_span_t decoded = {nullptr, 0};
    IREE_CHECK_OK(iree_net_codec_decode(server.get(), frame.data(), frame_size,
                                        &decoded));
    benchmark::DoNotOptimize(decoded.data);
  }

  state.SetBytesProcessed(state.iterations() * payload_size);
}
BENCHMARK(BM_AEADDecode)->Arg(64)->Arg(1024)->Arg(64 * 1024)->Arg(1024 * 1024);

static void BM_AEADRoundtrip(benchmark::State& state) {
  const size_t payload_size = static_cast<size_t>(state.range(0));

  CodecPtr client = CreateCodec(IREE_NET_AEAD_ROLE_CLIENT);
  CodecPtr server = CreateCodec(IREE_NET_AEAD_ROLE_SERVER);

  // Allocate frame buffer.
  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  size_t frame_size = overhead.prefix + payload_size + overhead.suffix;
  std::vector<uint8_t> frame(frame_size);
  uint8_t* payload_ptr = frame.data() + overhead.prefix;

  // Fill with pattern.
  for (size_t i = 0; i < payload_size; ++i) {
    payload_ptr[i] = static_cast<uint8_t>(i & 0xFF);
  }

  for (auto _ : state) {
    IREE_CHECK_OK(
        iree_net_codec_encode(client.get(), payload_ptr, payload_size));
    iree_byte_span_t decoded = {nullptr, 0};
    IREE_CHECK_OK(iree_net_codec_decode(server.get(), frame.data(), frame_size,
                                        &decoded));
    benchmark::DoNotOptimize(decoded.data);
  }

  state.SetBytesProcessed(state.iterations() * payload_size *
                          2);  // encode + decode
}
BENCHMARK(BM_AEADRoundtrip)
    ->Arg(64)
    ->Arg(1024)
    ->Arg(64 * 1024)
    ->Arg(1024 * 1024);

//===----------------------------------------------------------------------===//
// Raw Monocypher Benchmarks (for comparison)
//===----------------------------------------------------------------------===//

static void BM_RawChaCha20Poly1305Lock(benchmark::State& state) {
  const size_t payload_size = static_cast<size_t>(state.range(0));

  std::vector<uint8_t> plaintext(payload_size);
  std::vector<uint8_t> ciphertext(payload_size);
  uint8_t tag[16] = {0};
  uint8_t nonce[24] = {0};

  for (size_t i = 0; i < payload_size; ++i) {
    plaintext[i] = static_cast<uint8_t>(i & 0xFF);
  }

  for (auto _ : state) {
    ++nonce[0];
    crypto_aead_lock(ciphertext.data(), tag, kTestPSK, nonce, nullptr, 0,
                     plaintext.data(), payload_size);
    benchmark::DoNotOptimize(ciphertext.data());
    benchmark::DoNotOptimize(tag);
  }

  state.SetBytesProcessed(state.iterations() * payload_size);
}
BENCHMARK(BM_RawChaCha20Poly1305Lock)
    ->Arg(64)
    ->Arg(1024)
    ->Arg(64 * 1024)
    ->Arg(1024 * 1024);

static void BM_RawChaCha20Poly1305Unlock(benchmark::State& state) {
  const size_t payload_size = static_cast<size_t>(state.range(0));

  std::vector<uint8_t> plaintext(payload_size);
  std::vector<uint8_t> ciphertext(payload_size);
  uint8_t tag[16] = {0};
  uint8_t nonce[24] = {0};

  for (size_t i = 0; i < payload_size; ++i) {
    plaintext[i] = static_cast<uint8_t>(i & 0xFF);
  }

  for (auto _ : state) {
    state.PauseTiming();
    ++nonce[0];
    crypto_aead_lock(ciphertext.data(), tag, kTestPSK, nonce, nullptr, 0,
                     plaintext.data(), payload_size);
    state.ResumeTiming();

    int result =
        crypto_aead_unlock(plaintext.data(), tag, kTestPSK, nonce, nullptr, 0,
                           ciphertext.data(), payload_size);
    benchmark::DoNotOptimize(plaintext.data());
    benchmark::DoNotOptimize(result);
  }

  state.SetBytesProcessed(state.iterations() * payload_size);
}
BENCHMARK(BM_RawChaCha20Poly1305Unlock)
    ->Arg(64)
    ->Arg(1024)
    ->Arg(64 * 1024)
    ->Arg(1024 * 1024);

}  // namespace
