// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/codec/aead/codec.h"

#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

// RAII wrapper for iree_net_codec_t using std::unique_ptr.
struct CodecDeleter {
  void operator()(iree_net_codec_t* codec) const {
    if (codec) iree_net_codec_release(codec);
  }
};
using CodecPtr = std::unique_ptr<iree_net_codec_t, CodecDeleter>;

// Test PSK (32 bytes of deterministic data for reproducible tests).
static const uint8_t kTestPSK[IREE_NET_AEAD_KEY_SIZE] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
    0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
    0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
};

// Creates a codec with the test PSK.
StatusOr<CodecPtr> CreateCodec(iree_net_aead_role_t role) {
  iree_const_byte_span_t psk = {kTestPSK, sizeof(kTestPSK)};
  iree_net_codec_t* codec = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_net_aead_codec_create(psk, role, iree_allocator_system(), &codec));
  return CodecPtr(codec);
}

// Creates a client/server codec pair with matching PSKs.
StatusOr<std::pair<CodecPtr, CodecPtr>> CreateCodecPair() {
  IREE_ASSIGN_OR_RETURN(auto client, CreateCodec(IREE_NET_AEAD_ROLE_CLIENT));
  IREE_ASSIGN_OR_RETURN(auto server, CreateCodec(IREE_NET_AEAD_ROLE_SERVER));
  return std::make_pair(std::move(client), std::move(server));
}

//===----------------------------------------------------------------------===//
// Creation and Lifecycle
//===----------------------------------------------------------------------===//

TEST(AEADCodecTest, CreateDestroy) {
  IREE_ASSERT_OK_AND_ASSIGN(auto codec, CreateCodec(IREE_NET_AEAD_ROLE_CLIENT));
  ASSERT_NE(codec.get(), nullptr);
}

TEST(AEADCodecTest, CreateRejectsBadPSKSize) {
  // Too short.
  uint8_t short_psk[16] = {0};
  iree_const_byte_span_t psk = {short_psk, sizeof(short_psk)};
  iree_net_codec_t* codec = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_aead_codec_create(psk, IREE_NET_AEAD_ROLE_CLIENT,
                                 iree_allocator_system(), &codec));
  EXPECT_EQ(codec, nullptr);

  // Too long.
  uint8_t long_psk[64] = {0};
  psk = {long_psk, sizeof(long_psk)};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_aead_codec_create(psk, IREE_NET_AEAD_ROLE_CLIENT,
                                 iree_allocator_system(), &codec));
  EXPECT_EQ(codec, nullptr);
}

TEST(AEADCodecTest, RetainRelease) {
  IREE_ASSERT_OK_AND_ASSIGN(auto codec, CreateCodec(IREE_NET_AEAD_ROLE_CLIENT));

  // Retain adds reference, so we need to manually release to balance.
  iree_net_codec_retain(codec.get());
  iree_net_codec_release(codec.get());
  // codec's destructor will call final release.
}

//===----------------------------------------------------------------------===//
// Overhead Query
//===----------------------------------------------------------------------===//

TEST(AEADCodecTest, QueryOverhead) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  // AEAD uses 16-byte nonce prefix and 16-byte auth tag suffix.
  EXPECT_EQ(overhead.prefix, 16u);
  EXPECT_EQ(overhead.suffix, 16u);
  EXPECT_EQ(iree_net_codec_overhead_total(overhead), 32u);

  // Server should have same overhead.
  overhead = iree_net_codec_query_overhead(server.get());
  EXPECT_EQ(overhead.prefix, 16u);
  EXPECT_EQ(overhead.suffix, 16u);
}

TEST(AEADCodecTest, RequiresCPUAccess) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;
  EXPECT_TRUE(iree_net_codec_requires_cpu_access(client.get()));
  EXPECT_TRUE(iree_net_codec_requires_cpu_access(server.get()));
}

//===----------------------------------------------------------------------===//
// Encode/Decode Roundtrip
//===----------------------------------------------------------------------===//

TEST(AEADCodecTest, RoundtripClientToServer) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  // Prepare test payload.
  const char* test_message = "Hello, secure world!";
  iree_host_size_t payload_length = strlen(test_message);

  // Allocate frame buffer with overhead.
  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  iree_host_size_t frame_size =
      iree_net_codec_frame_size(overhead, payload_length);
  std::vector<uint8_t> frame(frame_size);

  // Write payload at correct offset.
  uint8_t* payload_ptr = frame.data() + overhead.prefix;
  memcpy(payload_ptr, test_message, payload_length);

  // Encode (client side).
  IREE_ASSERT_OK(
      iree_net_codec_encode(client.get(), payload_ptr, payload_length));

  // Verify ciphertext is different from plaintext.
  EXPECT_NE(memcmp(payload_ptr, test_message, payload_length), 0);

  // Decode (server side).
  iree_byte_span_t decoded_payload = {nullptr, 0};
  IREE_ASSERT_OK(iree_net_codec_decode(server.get(), frame.data(), frame_size,
                                       &decoded_payload));

  // Verify decoded payload matches original.
  ASSERT_EQ(decoded_payload.data_length, payload_length);
  EXPECT_EQ(memcmp(decoded_payload.data, test_message, payload_length), 0);
}

TEST(AEADCodecTest, RoundtripServerToClient) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  const char* test_message = "Response from server";
  iree_host_size_t payload_length = strlen(test_message);

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(server.get());
  iree_host_size_t frame_size =
      iree_net_codec_frame_size(overhead, payload_length);
  std::vector<uint8_t> frame(frame_size);

  uint8_t* payload_ptr = frame.data() + overhead.prefix;
  memcpy(payload_ptr, test_message, payload_length);

  // Encode (server side).
  IREE_ASSERT_OK(
      iree_net_codec_encode(server.get(), payload_ptr, payload_length));

  // Decode (client side).
  iree_byte_span_t decoded_payload = {nullptr, 0};
  IREE_ASSERT_OK(iree_net_codec_decode(client.get(), frame.data(), frame_size,
                                       &decoded_payload));

  ASSERT_EQ(decoded_payload.data_length, payload_length);
  EXPECT_EQ(memcmp(decoded_payload.data, test_message, payload_length), 0);
}

TEST(AEADCodecTest, RoundtripEmptyPayload) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  iree_host_size_t frame_size = iree_net_codec_frame_size(overhead, 0);
  std::vector<uint8_t> frame(frame_size);

  uint8_t* payload_ptr = frame.data() + overhead.prefix;

  IREE_ASSERT_OK(iree_net_codec_encode(client.get(), payload_ptr, 0));

  iree_byte_span_t decoded_payload = {nullptr, 0};
  IREE_ASSERT_OK(iree_net_codec_decode(server.get(), frame.data(), frame_size,
                                       &decoded_payload));
  EXPECT_EQ(decoded_payload.data_length, 0u);
}

TEST(AEADCodecTest, RoundtripLargePayload) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  // 64KB payload.
  iree_host_size_t payload_length = 64 * 1024;
  std::vector<uint8_t> original_payload(payload_length);
  for (size_t i = 0; i < payload_length; ++i) {
    original_payload[i] = static_cast<uint8_t>(i & 0xFF);
  }

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  iree_host_size_t frame_size =
      iree_net_codec_frame_size(overhead, payload_length);
  std::vector<uint8_t> frame(frame_size);

  uint8_t* payload_ptr = frame.data() + overhead.prefix;
  memcpy(payload_ptr, original_payload.data(), payload_length);

  IREE_ASSERT_OK(
      iree_net_codec_encode(client.get(), payload_ptr, payload_length));

  iree_byte_span_t decoded_payload = {nullptr, 0};
  IREE_ASSERT_OK(iree_net_codec_decode(server.get(), frame.data(), frame_size,
                                       &decoded_payload));

  ASSERT_EQ(decoded_payload.data_length, payload_length);
  EXPECT_EQ(
      memcmp(decoded_payload.data, original_payload.data(), payload_length), 0);
}

//===----------------------------------------------------------------------===//
// Direction-Specific Keys
//===----------------------------------------------------------------------===//

TEST(AEADCodecTest, WrongDirectionFails) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  const char* test_message = "Test";
  iree_host_size_t payload_length = strlen(test_message);

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  iree_host_size_t frame_size =
      iree_net_codec_frame_size(overhead, payload_length);
  std::vector<uint8_t> frame(frame_size);

  uint8_t* payload_ptr = frame.data() + overhead.prefix;
  memcpy(payload_ptr, test_message, payload_length);

  // Encode with client...
  IREE_ASSERT_OK(
      iree_net_codec_encode(client.get(), payload_ptr, payload_length));

  // ...try to decode with client (wrong direction).
  iree_byte_span_t decoded_payload = {nullptr, 0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_net_codec_decode(client.get(), frame.data(),
                                              frame_size, &decoded_payload));
}

TEST(AEADCodecTest, DifferentPSKFails) {
  // Create client with test PSK.
  IREE_ASSERT_OK_AND_ASSIGN(auto client,
                            CreateCodec(IREE_NET_AEAD_ROLE_CLIENT));

  // Create server with different PSK.
  uint8_t other_psk[IREE_NET_AEAD_KEY_SIZE] = {0xFF};
  iree_const_byte_span_t psk2 = {other_psk, sizeof(other_psk)};
  iree_net_codec_t* raw_server = nullptr;
  IREE_ASSERT_OK(iree_net_aead_codec_create(
      psk2, IREE_NET_AEAD_ROLE_SERVER, iree_allocator_system(), &raw_server));
  CodecPtr server(raw_server);

  const char* test_message = "Test";
  iree_host_size_t payload_length = strlen(test_message);

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  iree_host_size_t frame_size =
      iree_net_codec_frame_size(overhead, payload_length);
  std::vector<uint8_t> frame(frame_size);

  uint8_t* payload_ptr = frame.data() + overhead.prefix;
  memcpy(payload_ptr, test_message, payload_length);

  IREE_ASSERT_OK(
      iree_net_codec_encode(client.get(), payload_ptr, payload_length));

  iree_byte_span_t decoded_payload = {nullptr, 0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_net_codec_decode(server.get(), frame.data(),
                                              frame_size, &decoded_payload));
}

//===----------------------------------------------------------------------===//
// Authentication Failure Detection
//===----------------------------------------------------------------------===//

TEST(AEADCodecTest, TamperedCiphertextFails) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  const char* test_message = "Secret data";
  iree_host_size_t payload_length = strlen(test_message);

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  iree_host_size_t frame_size =
      iree_net_codec_frame_size(overhead, payload_length);
  std::vector<uint8_t> frame(frame_size);

  uint8_t* payload_ptr = frame.data() + overhead.prefix;
  memcpy(payload_ptr, test_message, payload_length);

  IREE_ASSERT_OK(
      iree_net_codec_encode(client.get(), payload_ptr, payload_length));

  // Flip a bit in the ciphertext.
  payload_ptr[0] ^= 0x01;

  iree_byte_span_t decoded_payload = {nullptr, 0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_net_codec_decode(server.get(), frame.data(),
                                              frame_size, &decoded_payload));
}

TEST(AEADCodecTest, TamperedTagFails) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  const char* test_message = "Secret data";
  iree_host_size_t payload_length = strlen(test_message);

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  iree_host_size_t frame_size =
      iree_net_codec_frame_size(overhead, payload_length);
  std::vector<uint8_t> frame(frame_size);

  uint8_t* payload_ptr = frame.data() + overhead.prefix;
  memcpy(payload_ptr, test_message, payload_length);

  IREE_ASSERT_OK(
      iree_net_codec_encode(client.get(), payload_ptr, payload_length));

  // Flip a bit in the auth tag (at the end).
  frame[frame_size - 1] ^= 0x01;

  iree_byte_span_t decoded_payload = {nullptr, 0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_net_codec_decode(server.get(), frame.data(),
                                              frame_size, &decoded_payload));
}

TEST(AEADCodecTest, TamperedNonceFails) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  const char* test_message = "Secret data";
  iree_host_size_t payload_length = strlen(test_message);

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  iree_host_size_t frame_size =
      iree_net_codec_frame_size(overhead, payload_length);
  std::vector<uint8_t> frame(frame_size);

  uint8_t* payload_ptr = frame.data() + overhead.prefix;
  memcpy(payload_ptr, test_message, payload_length);

  IREE_ASSERT_OK(
      iree_net_codec_encode(client.get(), payload_ptr, payload_length));

  // Flip a bit in the nonce (at the start).
  frame[0] ^= 0x01;

  iree_byte_span_t decoded_payload = {nullptr, 0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_net_codec_decode(server.get(), frame.data(),
                                              frame_size, &decoded_payload));
}

TEST(AEADCodecTest, TruncatedFrameFails) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  const char* test_message = "Secret data";
  iree_host_size_t payload_length = strlen(test_message);

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  iree_host_size_t frame_size =
      iree_net_codec_frame_size(overhead, payload_length);
  std::vector<uint8_t> frame(frame_size);

  uint8_t* payload_ptr = frame.data() + overhead.prefix;
  memcpy(payload_ptr, test_message, payload_length);

  IREE_ASSERT_OK(
      iree_net_codec_encode(client.get(), payload_ptr, payload_length));

  // Try to decode with truncated frame.
  iree_byte_span_t decoded_payload = {nullptr, 0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_codec_decode(server.get(), frame.data(),
                            iree_net_codec_overhead_total(overhead) - 1,
                            &decoded_payload));
}

//===----------------------------------------------------------------------===//
// Replay Protection
//===----------------------------------------------------------------------===//

TEST(AEADCodecTest, ReplayDetection) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  const char* test_message = "First message";
  iree_host_size_t payload_length = strlen(test_message);

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());
  iree_host_size_t frame_size =
      iree_net_codec_frame_size(overhead, payload_length);
  std::vector<uint8_t> frame1(frame_size);

  uint8_t* payload_ptr = frame1.data() + overhead.prefix;
  memcpy(payload_ptr, test_message, payload_length);

  IREE_ASSERT_OK(
      iree_net_codec_encode(client.get(), payload_ptr, payload_length));

  // Make a copy for replay attempt.
  std::vector<uint8_t> frame1_copy = frame1;

  // First decode should succeed.
  iree_byte_span_t decoded_payload = {nullptr, 0};
  IREE_ASSERT_OK(iree_net_codec_decode(server.get(), frame1.data(), frame_size,
                                       &decoded_payload));

  // Send a second message to advance the nonce counter.
  const char* test_message2 = "Second message";
  payload_length = strlen(test_message2);
  frame_size = iree_net_codec_frame_size(overhead, payload_length);
  std::vector<uint8_t> frame2(frame_size);

  payload_ptr = frame2.data() + overhead.prefix;
  memcpy(payload_ptr, test_message2, payload_length);
  IREE_ASSERT_OK(
      iree_net_codec_encode(client.get(), payload_ptr, payload_length));
  IREE_ASSERT_OK(iree_net_codec_decode(server.get(), frame2.data(), frame_size,
                                       &decoded_payload));

  // Replay of first frame should fail (nonce too old).
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_net_codec_decode(server.get(), frame1_copy.data(),
                            frame1_copy.size(), &decoded_payload));
}

TEST(AEADCodecTest, MultipleMessagesSucceed) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, CreateCodecPair());
  auto& [client, server] = pair;

  iree_net_codec_overhead_t overhead =
      iree_net_codec_query_overhead(client.get());

  // Send 100 messages.
  for (int i = 0; i < 100; ++i) {
    char message[32] = {0};
    snprintf(message, sizeof(message), "Message %d", i);
    iree_host_size_t payload_length = strlen(message);

    iree_host_size_t frame_size =
        iree_net_codec_frame_size(overhead, payload_length);
    std::vector<uint8_t> frame(frame_size);

    uint8_t* payload_ptr = frame.data() + overhead.prefix;
    memcpy(payload_ptr, message, payload_length);

    IREE_ASSERT_OK(
        iree_net_codec_encode(client.get(), payload_ptr, payload_length));

    iree_byte_span_t decoded_payload = {nullptr, 0};
    IREE_ASSERT_OK(iree_net_codec_decode(server.get(), frame.data(), frame_size,
                                         &decoded_payload));

    ASSERT_EQ(decoded_payload.data_length, payload_length);
    EXPECT_EQ(memcmp(decoded_payload.data, message, payload_length), 0);
  }
}

}  // namespace
}  // namespace net
}  // namespace iree
