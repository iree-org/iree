// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan raw SPIR-V BDA coverage. Raw BDA executables deliberately have no IREE
// flatbuffer metadata; the driver publishes the exact HAL binding table the
// dispatch provides and the shader consumes it through the hidden BDA root.

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/vulkan/command_buffer.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

// Shader contract:
//   layout(local_size_x = 1) in;
//   output[i] = input[i] + 7;
//
// The shader reads input/output addresses from the BDA binding table addressed
// by iree_hal_vulkan_bda_dispatch_root_v1_t::binding_table_address.
static const uint32_t kRawBdaSpirv[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x0000004c, 0x00000000, 0x00020011,
    0x00000001, 0x00020011, 0x0000000b, 0x00020011, 0x000014e3, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x000014e4, 0x00000001, 0x0007000f, 0x00000005, 0x00000004, 0x6e69616d,
    0x00000000, 0x0000000f, 0x00000036, 0x00060010, 0x00000004, 0x00000011,
    0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001cc,
    0x00070004, 0x455f4c47, 0x625f5458, 0x65666675, 0x65725f72, 0x65726566,
    0x0065636e, 0x00080004, 0x455f4c47, 0x735f5458, 0x616c6163, 0x6c625f72,
    0x5f6b636f, 0x6f79616c, 0x00007475, 0x000d0004, 0x455f4c47, 0x735f5458,
    0x65646168, 0x78655f72, 0x63696c70, 0x615f7469, 0x68746972, 0x6974656d,
    0x79745f63, 0x5f736570, 0x36746e69, 0x00000034, 0x00040005, 0x00000004,
    0x6e69616d, 0x00000000, 0x00060005, 0x00000009, 0x72646441, 0x54737365,
    0x656c6261, 0x00000000, 0x00050006, 0x00000009, 0x00000000, 0x72646461,
    0x00737365, 0x00060005, 0x0000000b, 0x646e6962, 0x5f676e69, 0x6c626174,
    0x00000065, 0x00040005, 0x0000000d, 0x746f6f52, 0x00000000, 0x00090006,
    0x0000000d, 0x00000000, 0x646e6962, 0x5f676e69, 0x6c626174, 0x64615f65,
    0x73657264, 0x00000073, 0x00080006, 0x0000000d, 0x00000001, 0x736e6f63,
    0x746e6174, 0x64615f73, 0x73657264, 0x00000073, 0x00070006, 0x0000000d,
    0x00000002, 0x646e6962, 0x5f676e69, 0x65736162, 0x00000000, 0x00070006,
    0x0000000d, 0x00000003, 0x736e6f63, 0x746e6174, 0x7361625f, 0x00000065,
    0x00050006, 0x0000000d, 0x00000004, 0x67616c66, 0x00000073, 0x00060006,
    0x0000000d, 0x00000005, 0x65736572, 0x64657672, 0x00000030, 0x00040005,
    0x0000000f, 0x746f6f72, 0x00000000, 0x00050005, 0x00000020, 0x75706e49,
    0x66754274, 0x00726566, 0x00050006, 0x00000020, 0x00000000, 0x756c6176,
    0x00000065, 0x00060005, 0x00000022, 0x75706e69, 0x75625f74, 0x72656666,
    0x00000000, 0x00060005, 0x0000002a, 0x7074754f, 0x75427475, 0x72656666,
    0x00000000, 0x00050006, 0x0000002a, 0x00000000, 0x756c6176, 0x00000065,
    0x00060005, 0x0000002c, 0x7074756f, 0x625f7475, 0x65666675, 0x00000072,
    0x00040005, 0x00000033, 0x65646e69, 0x00000078, 0x00080005, 0x00000036,
    0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044,
    0x00040047, 0x00000008, 0x00000006, 0x00000008, 0x00030047, 0x00000009,
    0x00000002, 0x00040048, 0x00000009, 0x00000000, 0x00000018, 0x00050048,
    0x00000009, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000000b,
    0x000014ec, 0x00030047, 0x0000000d, 0x00000002, 0x00050048, 0x0000000d,
    0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000000d, 0x00000001,
    0x00000023, 0x00000008, 0x00050048, 0x0000000d, 0x00000002, 0x00000023,
    0x00000010, 0x00050048, 0x0000000d, 0x00000003, 0x00000023, 0x00000014,
    0x00050048, 0x0000000d, 0x00000004, 0x00000023, 0x00000018, 0x00050048,
    0x0000000d, 0x00000005, 0x00000023, 0x0000001c, 0x00040047, 0x0000001f,
    0x00000006, 0x00000004, 0x00030047, 0x00000020, 0x00000002, 0x00040048,
    0x00000020, 0x00000000, 0x00000018, 0x00050048, 0x00000020, 0x00000000,
    0x00000023, 0x00000000, 0x00030047, 0x00000022, 0x000014ec, 0x00040047,
    0x00000029, 0x00000006, 0x00000004, 0x00030047, 0x0000002a, 0x00000002,
    0x00040048, 0x0000002a, 0x00000000, 0x00000019, 0x00050048, 0x0000002a,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000002c, 0x000014ec,
    0x00040047, 0x00000036, 0x0000000b, 0x0000001c, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00030027, 0x00000006, 0x000014e5,
    0x00040015, 0x00000007, 0x00000040, 0x00000000, 0x0003001d, 0x00000008,
    0x00000007, 0x0003001e, 0x00000009, 0x00000008, 0x00040020, 0x00000006,
    0x000014e5, 0x00000009, 0x00040020, 0x0000000a, 0x00000007, 0x00000006,
    0x00040015, 0x0000000c, 0x00000020, 0x00000000, 0x0008001e, 0x0000000d,
    0x00000007, 0x00000007, 0x0000000c, 0x0000000c, 0x0000000c, 0x0000000c,
    0x00040020, 0x0000000e, 0x00000009, 0x0000000d, 0x0004003b, 0x0000000e,
    0x0000000f, 0x00000009, 0x00040015, 0x00000010, 0x00000020, 0x00000001,
    0x0004002b, 0x00000010, 0x00000011, 0x00000000, 0x00040020, 0x00000012,
    0x00000009, 0x00000007, 0x0004002b, 0x00000010, 0x00000015, 0x00000002,
    0x00040020, 0x00000016, 0x00000009, 0x0000000c, 0x0005002b, 0x00000007,
    0x0000001a, 0x00000008, 0x00000000, 0x00030027, 0x0000001e, 0x000014e5,
    0x0003001d, 0x0000001f, 0x00000010, 0x0003001e, 0x00000020, 0x0000001f,
    0x00040020, 0x0000001e, 0x000014e5, 0x00000020, 0x00040020, 0x00000021,
    0x00000007, 0x0000001e, 0x00040020, 0x00000024, 0x000014e5, 0x00000007,
    0x00030027, 0x00000028, 0x000014e5, 0x0003001d, 0x00000029, 0x00000010,
    0x0003001e, 0x0000002a, 0x00000029, 0x00040020, 0x00000028, 0x000014e5,
    0x0000002a, 0x00040020, 0x0000002b, 0x00000007, 0x00000028, 0x0004002b,
    0x00000010, 0x0000002e, 0x00000001, 0x00040020, 0x00000032, 0x00000007,
    0x0000000c, 0x00040017, 0x00000034, 0x0000000c, 0x00000003, 0x00040020,
    0x00000035, 0x00000001, 0x00000034, 0x0004003b, 0x00000035, 0x00000036,
    0x00000001, 0x0004002b, 0x0000000c, 0x00000037, 0x00000000, 0x00040020,
    0x00000038, 0x00000001, 0x0000000c, 0x00040020, 0x0000003f, 0x000014e5,
    0x00000010, 0x0004002b, 0x00000010, 0x00000042, 0x00000004, 0x0004002b,
    0x00000010, 0x00000047, 0x00000007, 0x0004002b, 0x0000000c, 0x0000004a,
    0x00000001, 0x0006002c, 0x00000034, 0x0000004b, 0x0000004a, 0x0000004a,
    0x0000004a, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000007,
    0x0004003b, 0x00000021, 0x00000022, 0x00000007, 0x0004003b, 0x0000002b,
    0x0000002c, 0x00000007, 0x0004003b, 0x00000032, 0x00000033, 0x00000007,
    0x00050041, 0x00000012, 0x00000013, 0x0000000f, 0x00000011, 0x0004003d,
    0x00000007, 0x00000014, 0x00000013, 0x00050041, 0x00000016, 0x00000017,
    0x0000000f, 0x00000015, 0x0004003d, 0x0000000c, 0x00000018, 0x00000017,
    0x00040071, 0x00000007, 0x00000019, 0x00000018, 0x00050084, 0x00000007,
    0x0000001b, 0x00000019, 0x0000001a, 0x00050080, 0x00000007, 0x0000001c,
    0x00000014, 0x0000001b, 0x00040078, 0x00000006, 0x0000001d, 0x0000001c,
    0x0003003e, 0x0000000b, 0x0000001d, 0x0004003d, 0x00000006, 0x00000023,
    0x0000000b, 0x00060041, 0x00000024, 0x00000025, 0x00000023, 0x00000011,
    0x00000011, 0x0006003d, 0x00000007, 0x00000026, 0x00000025, 0x00000002,
    0x00000008, 0x00040078, 0x0000001e, 0x00000027, 0x00000026, 0x0003003e,
    0x00000022, 0x00000027, 0x0004003d, 0x00000006, 0x0000002d, 0x0000000b,
    0x00060041, 0x00000024, 0x0000002f, 0x0000002d, 0x00000011, 0x0000002e,
    0x0006003d, 0x00000007, 0x00000030, 0x0000002f, 0x00000002, 0x00000008,
    0x00040078, 0x00000028, 0x00000031, 0x00000030, 0x0003003e, 0x0000002c,
    0x00000031, 0x00050041, 0x00000038, 0x00000039, 0x00000036, 0x00000037,
    0x0004003d, 0x0000000c, 0x0000003a, 0x00000039, 0x0003003e, 0x00000033,
    0x0000003a, 0x0004003d, 0x00000028, 0x0000003b, 0x0000002c, 0x0004003d,
    0x0000000c, 0x0000003c, 0x00000033, 0x0004003d, 0x0000001e, 0x0000003d,
    0x00000022, 0x0004003d, 0x0000000c, 0x0000003e, 0x00000033, 0x00060041,
    0x0000003f, 0x00000040, 0x0000003d, 0x00000011, 0x0000003e, 0x0006003d,
    0x00000010, 0x00000041, 0x00000040, 0x00000002, 0x00000004, 0x00050041,
    0x00000016, 0x00000043, 0x0000000f, 0x00000042, 0x0004003d, 0x0000000c,
    0x00000044, 0x00000043, 0x0004007c, 0x00000010, 0x00000045, 0x00000044,
    0x00050080, 0x00000010, 0x00000046, 0x00000041, 0x00000045, 0x00050080,
    0x00000010, 0x00000048, 0x00000046, 0x00000047, 0x00060041, 0x0000003f,
    0x00000049, 0x0000003b, 0x00000011, 0x0000003c, 0x0005003e, 0x00000049,
    0x00000048, 0x00000002, 0x00000004, 0x000100fd, 0x00010038,
};

// Valid descriptor-free BDA-environment no-op shader with no push-constant
// root. Normal raw loading rejects it because reflection cannot prove the HAL
// root convention; explicitly unverified raw loading accepts it for
// hand-authored/custom-argument kernels.
static const uint32_t kRawBdaNoopSpirvWithoutPushConstantRoot[] = {
    0x07230203u,
    0x00010600u,
    0u,
    5u,
    0u,
    // Declares OpCapability Shader.
    0x00020011u,
    1u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
    0x0003000eu,
    5348u,
    1u,
    // Declares OpEntryPoint GLCompute %main "main".
    0x0005000fu,
    5u,
    3u,
    0x6e69616du,
    0u,
    // Declares OpExecutionMode %main LocalSize 1 1 1.
    0x00060010u,
    3u,
    17u,
    1u,
    1u,
    1u,
    // Declares OpTypeVoid %void.
    0x00020013u,
    1u,
    // Declares OpTypeFunction %fn %void.
    0x00030021u,
    2u,
    1u,
    // Defines %main as an empty compute function.
    0x00050036u,
    1u,
    3u,
    0u,
    2u,
    0x000200f8u,
    4u,
    0x000100fdu,
    0x00010038u,
};

static const uint32_t
    kRawBdaSpirvMissingPhysicalStorageBufferAddressesCapability[] = {
        0x07230203u,
        0x00010600u,
        0u,
        8u,
        0u,
        // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
        0x0003000eu,
        5348u,
        1u,
        // Declares OpEntryPoint GLCompute %1 "main".
        0x0005000fu,
        5u,
        1u,
        0x6e69616du,
        0u,
        // Declares OpExecutionMode %1 LocalSize 1 1 1.
        0x00060010u,
        1u,
        17u,
        1u,
        1u,
        1u,
};

static const uint32_t kRawBdaSpirvWithoutComputeEntryPoints[] = {
    0x07230203u,
    0x00010600u,
    0u,
    8u,
    0u,
    // Declares OpCapability Shader.
    0x00020011u,
    1u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
    0x0003000eu,
    5348u,
    1u,
};

static const uint32_t kRawBdaSpirvWithDuplicateComputeEntryNames[] = {
    0x07230203u,
    0x00010600u,
    0u,
    8u,
    0u,
    // Declares OpCapability Shader.
    0x00020011u,
    1u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
    0x0003000eu,
    5348u,
    1u,
    // Declares OpEntryPoint GLCompute %1 "main".
    0x0005000fu,
    5u,
    1u,
    0x6e69616du,
    0u,
    // Declares OpEntryPoint GLCompute %2 "main".
    0x0005000fu,
    5u,
    2u,
    0x6e69616du,
    0u,
};

static const uint32_t kRawBdaSpirvWithTruncatedLocalSize[] = {
    0x07230203u,
    0x00010600u,
    0u,
    8u,
    0u,
    // Declares OpCapability Shader.
    0x00020011u,
    1u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
    0x0003000eu,
    5348u,
    1u,
    // Declares OpEntryPoint GLCompute %1 "main".
    0x0005000fu,
    5u,
    1u,
    0x6e69616du,
    0u,
    // Declares truncated OpExecutionMode %1 LocalSize 1.
    0x00040010u,
    1u,
    17u,
    1u,
};

static const uint32_t kRawBdaSpirvMissingPushConstantRoot[] = {
    0x07230203u,
    0x00010600u,
    0u,
    8u,
    0u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
    0x0003000eu,
    5348u,
    1u,
    // Declares OpEntryPoint GLCompute %1 "main".
    0x0005000fu,
    5u,
    1u,
    0x6e69616du,
    0u,
    // Declares OpExecutionMode %1 LocalSize 1 1 1.
    0x00060010u,
    1u,
    17u,
    1u,
    1u,
    1u,
};

static const uint32_t kRawBdaSpirvWithDescriptorStorageVariable[] = {
    0x07230203u,
    0x00010600u,
    0u,
    8u,
    0u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
    0x0003000eu,
    5348u,
    1u,
    // Declares OpEntryPoint GLCompute %1 "main".
    0x0005000fu,
    5u,
    1u,
    0x6e69616du,
    0u,
    // Declares OpExecutionMode %1 LocalSize 1 1 1.
    0x00060010u,
    1u,
    17u,
    1u,
    1u,
    1u,
    // Declares OpVariable %3 in PushConstant storage class.
    0x0004003bu,
    2u,
    3u,
    9u,
    // Declares OpVariable %5 in StorageBuffer storage class.
    0x0004003bu,
    4u,
    5u,
    12u,
};

static std::vector<uint32_t> MakeDescriptorDecoratedRawBdaSpirv() {
  std::vector<uint32_t> spirv(kRawBdaSpirv,
                              kRawBdaSpirv + IREE_ARRAYSIZE(kRawBdaSpirv));
  // Appends OpDecorate %root DescriptorSet 0.
  spirv.push_back(0x00040047u);
  spirv.push_back(0x0000000fu);
  spirv.push_back(0x00000022u);
  spirv.push_back(0x00000000u);
  return spirv;
}

static void AppendSpirvStringInstruction(uint16_t opcode, const char* value,
                                         std::vector<uint32_t>* words) {
  const size_t byte_length = std::strlen(value) + 1;
  const size_t string_word_count =
      (byte_length + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  words->push_back(
      (uint32_t)(((string_word_count + 1) << 16) | (uint32_t)opcode));
  const size_t string_word_offset = words->size();
  words->resize(string_word_offset + string_word_count, 0);
  std::memcpy(&(*words)[string_word_offset], value, byte_length);
}

static std::vector<uint32_t> MakeRawBdaSpirvWithMetadata(
    std::initializer_list<const char*> metadata_strings) {
  std::vector<uint32_t> source(kRawBdaSpirv,
                               kRawBdaSpirv + IREE_ARRAYSIZE(kRawBdaSpirv));
  iree_host_size_t insertion_offset = IREE_ARRAYSIZE(kRawBdaSpirv);
  for (iree_host_size_t word_offset = 5; word_offset < source.size();) {
    const uint32_t instruction = source[word_offset];
    const uint16_t word_count = (uint16_t)(instruction >> 16);
    const uint16_t opcode = (uint16_t)(instruction & 0xFFFFu);
    if (opcode == 71u) {
      insertion_offset = word_offset;
      break;
    }
    word_offset += word_count;
  }

  std::vector<uint32_t> spirv;
  spirv.insert(spirv.end(), source.begin(), source.begin() + insertion_offset);
  for (const char* metadata_string : metadata_strings) {
    AppendSpirvStringInstruction(/*OpModuleProcessed=*/330u, metadata_string,
                                 &spirv);
  }
  spirv.insert(spirv.end(), source.begin() + insertion_offset, source.end());
  return spirv;
}

class BdaRawSpirvTest : public CtsTestBase<> {
 protected:
  static constexpr iree_host_size_t kElementCount = 4;
  static constexpr iree_device_size_t kDispatchByteLength =
      kElementCount * sizeof(int32_t);

  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    IREE_ASSERT_OK(PrepareRawBdaExecutable(
        iree_make_const_byte_span(kRawBdaSpirv, sizeof(kRawBdaSpirv)),
        &executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_status_t PrepareRawBdaExecutable(
      iree_const_byte_span_t executable_data,
      iree_hal_executable_caching_mode_t caching_mode,
      iree_hal_executable_t** out_executable) {
    *out_executable = nullptr;
    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode = caching_mode;
    executable_params.executable_format =
        iree_make_cstring_view("vulkan-spirv-bda-raw");
    executable_params.executable_data = executable_data;
    return iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, out_executable);
  }

  iree_status_t PrepareRawBdaExecutable(
      iree_const_byte_span_t executable_data,
      iree_hal_executable_t** out_executable) {
    return PrepareRawBdaExecutable(
        executable_data, IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA,
        out_executable);
  }

  void CreateInputOutputBuffers(Ref<iree_hal_buffer_t>* input_buffer,
                                Ref<iree_hal_buffer_t>* output_buffer) {
    const int32_t input_data[kElementCount] = {1, -2, 30, 400};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(input_data, sizeof(input_data),
                                              input_buffer->out()));
    IREE_ASSERT_OK(
        CreateZeroedDeviceBuffer(sizeof(input_data), output_buffer->out()));
  }

  static iree_hal_buffer_ref_list_t MakeBindings(
      iree_hal_buffer_t* input_buffer, iree_hal_buffer_t* output_buffer,
      iree_hal_buffer_ref_t binding_refs[2]) {
    binding_refs[0] = iree_hal_make_buffer_ref(
        input_buffer, /*offset=*/0, iree_hal_buffer_byte_length(input_buffer));
    binding_refs[1] =
        iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                                 iree_hal_buffer_byte_length(output_buffer));
    return {
        /*.count=*/2,
        /*.values=*/binding_refs,
    };
  }

  static std::vector<iree_hal_buffer_ref_t> MakeOversizedPublicationBindings(
      iree_hal_buffer_t* input_buffer, iree_hal_buffer_t* output_buffer) {
    static constexpr iree_host_size_t kDefaultPublicationBlockLength =
        64 * 1024;
    static constexpr iree_host_size_t kBindingCount =
        kDefaultPublicationBlockLength / sizeof(uint64_t) + 1;

    std::vector<iree_hal_buffer_ref_t> binding_refs(kBindingCount);
    binding_refs[0] = iree_hal_make_buffer_ref(
        input_buffer, /*offset=*/0, iree_hal_buffer_byte_length(input_buffer));
    binding_refs[1] =
        iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                                 iree_hal_buffer_byte_length(output_buffer));
    for (iree_host_size_t i = 2; i < binding_refs.size(); ++i) {
      binding_refs[i] =
          iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                                   iree_hal_buffer_byte_length(input_buffer));
    }
    return binding_refs;
  }

  void CreateIndirectDispatchCommandBuffer(
      uint32_t workgroup_count,
      Ref<iree_hal_command_buffer_t>* command_buffer) {
    iree_hal_buffer_ref_t binding_refs[2] = {
        iree_hal_make_indirect_buffer_ref(
            /*buffer_slot=*/0, /*offset=*/0,
            /*length=*/(iree_device_size_t)workgroup_count * sizeof(int32_t)),
        iree_hal_make_indirect_buffer_ref(
            /*buffer_slot=*/1, /*offset=*/0,
            /*length=*/(iree_device_size_t)workgroup_count * sizeof(int32_t)),
    };
    iree_hal_buffer_ref_list_t bindings = {
        /*.count=*/IREE_ARRAYSIZE(binding_refs),
        /*.values=*/binding_refs,
    };

    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/2, command_buffer->out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer->get()));
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        command_buffer->get(), executable_,
        iree_hal_executable_function_from_index(0),
        iree_hal_make_static_dispatch_config(workgroup_count, 1, 1),
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer->get()));
  }

  static iree_hal_buffer_binding_table_t MakeBindingTable(
      iree_hal_buffer_t* input_buffer, iree_hal_buffer_t* output_buffer,
      iree_hal_buffer_binding_t binding_table_entries[2]) {
    binding_table_entries[0] = {
        /*buffer=*/input_buffer,
        /*offset=*/0,
        /*length=*/iree_hal_buffer_byte_length(input_buffer),
    };
    binding_table_entries[1] = {
        /*buffer=*/output_buffer,
        /*offset=*/0,
        /*length=*/iree_hal_buffer_byte_length(output_buffer),
    };
    return {
        /*.count=*/2,
        /*.bindings=*/binding_table_entries,
    };
  }

  void ExpectOutput(iree_hal_buffer_t* output_buffer) {
    std::vector<int32_t> output_data = ReadBufferData<int32_t>(output_buffer);
    EXPECT_THAT(output_data, ContainerEq(std::vector<int32_t>{8, 5, 37, 407}));
  }

  void ExpectFilledOutputPrefix(iree_hal_buffer_t* output_buffer,
                                int32_t expected_value) {
    std::vector<uint8_t> bytes =
        ReadBufferBytes(output_buffer, /*offset=*/0, kDispatchByteLength);
    ASSERT_EQ(kDispatchByteLength, bytes.size());
    std::vector<int32_t> output_data(kElementCount);
    std::memcpy(output_data.data(), bytes.data(), kDispatchByteLength);
    EXPECT_THAT(output_data, ContainerEq(std::vector<int32_t>(kElementCount,
                                                              expected_value)));
  }

  int64_t QueryNativeReplayCache(iree_string_view_t key) {
    int64_t value = 0;
    IREE_EXPECT_OK(iree_hal_device_query_i64(
        device_, IREE_SV("vulkan.queue.native_replay_cache"), key, &value));
    return value;
  }

  int64_t QueryBdaPublicationCache(iree_string_view_t key) {
    int64_t value = 0;
    IREE_EXPECT_OK(iree_hal_device_query_i64(
        device_, IREE_SV("vulkan.queue.bda_publication_cache"), key, &value));
    return value;
  }

  void ExecuteCommandBufferAndWait(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table) {
    SemaphoreList execute_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        execute_signal, command_buffer, binding_table,
        IREE_HAL_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        execute_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

static iree_hal_buffer_params_t SparseDispatchBufferParams() {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  return params;
}

TEST_P(BdaRawSpirvTest, PrepareRejectsDescriptorDecoratedRawBdaSpirv) {
  std::vector<uint32_t> decorated_spirv = MakeDescriptorDecoratedRawBdaSpirv();
  iree_hal_executable_t* decorated_executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareRawBdaExecutable(
          iree_make_const_byte_span(decorated_spirv.data(),
                                    decorated_spirv.size() * sizeof(uint32_t)),
          &decorated_executable));
  EXPECT_EQ(nullptr, decorated_executable);
  iree_hal_executable_release(decorated_executable);
}

TEST_P(BdaRawSpirvTest,
       PrepareRejectsRawBdaSpirvWithoutPhysicalStorageBufferAddresses) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareRawBdaExecutable(
          iree_make_const_byte_span(
              kRawBdaSpirvMissingPhysicalStorageBufferAddressesCapability,
              sizeof(
                  kRawBdaSpirvMissingPhysicalStorageBufferAddressesCapability)),
          &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaRawSpirvTest, PrepareRejectsMalformedRawBdaSpirvHeader) {
  const uint32_t not_spirv[5] = {0};
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareRawBdaExecutable(
          iree_make_const_byte_span(not_spirv, sizeof(not_spirv)),
          &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaRawSpirvTest, PrepareRejectsRawBdaSpirvWithoutComputeEntryPoints) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        PrepareRawBdaExecutable(
                            iree_make_const_byte_span(
                                kRawBdaSpirvWithoutComputeEntryPoints,
                                sizeof(kRawBdaSpirvWithoutComputeEntryPoints)),
                            &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaRawSpirvTest, PrepareRejectsRawBdaSpirvWithDuplicateEntryNames) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareRawBdaExecutable(
          iree_make_const_byte_span(
              kRawBdaSpirvWithDuplicateComputeEntryNames,
              sizeof(kRawBdaSpirvWithDuplicateComputeEntryNames)),
          &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaRawSpirvTest, PrepareRejectsRawBdaSpirvWithTruncatedLocalSize) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareRawBdaExecutable(
          iree_make_const_byte_span(kRawBdaSpirvWithTruncatedLocalSize,
                                    sizeof(kRawBdaSpirvWithTruncatedLocalSize)),
          &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaRawSpirvTest, PrepareRejectsRawBdaSpirvWithoutPushConstantRoot) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareRawBdaExecutable(iree_make_const_byte_span(
                                  kRawBdaSpirvMissingPushConstantRoot,
                                  sizeof(kRawBdaSpirvMissingPushConstantRoot)),
                              &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaRawSpirvTest, PrepareRejectsRawBdaNoopWithoutVerificationDisabled) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareRawBdaExecutable(
          iree_make_const_byte_span(
              kRawBdaNoopSpirvWithoutPushConstantRoot,
              sizeof(kRawBdaNoopSpirvWithoutPushConstantRoot)),
          &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaRawSpirvTest, QueueDispatchExecutesUnverifiedRawBdaNoop) {
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(PrepareRawBdaExecutable(
      iree_make_const_byte_span(
          kRawBdaNoopSpirvWithoutPushConstantRoot,
          sizeof(kRawBdaNoopSpirvWithoutPushConstantRoot)),
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA |
          IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION,
      executable.out()));

  iree_hal_buffer_ref_list_t bindings = {};
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable.get(),
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(BdaRawSpirvTest, PrepareRejectsRawBdaSpirvWithDescriptorVariable) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareRawBdaExecutable(
          iree_make_const_byte_span(
              kRawBdaSpirvWithDescriptorStorageVariable,
              sizeof(kRawBdaSpirvWithDescriptorStorageVariable)),
          &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

class BdaSparseVirtualBufferRef {
 public:
  explicit BdaSparseVirtualBufferRef(iree_hal_allocator_t* allocator)
      : allocator_(allocator) {}
  ~BdaSparseVirtualBufferRef() { reset(); }

  BdaSparseVirtualBufferRef(const BdaSparseVirtualBufferRef&) = delete;
  BdaSparseVirtualBufferRef& operator=(const BdaSparseVirtualBufferRef&) =
      delete;

  iree_hal_buffer_t* get() const { return buffer_; }
  iree_hal_buffer_t** out() { return &buffer_; }

  void reset() {
    if (buffer_) {
      IREE_EXPECT_OK(
          iree_hal_allocator_virtual_memory_release(allocator_, buffer_));
      buffer_ = nullptr;
    }
  }

 private:
  // Allocator that owns the virtual reservation.
  iree_hal_allocator_t* allocator_ = nullptr;

  // Reserved virtual sparse buffer released through |allocator_|.
  iree_hal_buffer_t* buffer_ = nullptr;
};

class BdaSparsePhysicalMemoryRef {
 public:
  explicit BdaSparsePhysicalMemoryRef(iree_hal_allocator_t* allocator)
      : allocator_(allocator) {}
  ~BdaSparsePhysicalMemoryRef() { reset(); }

  BdaSparsePhysicalMemoryRef(const BdaSparsePhysicalMemoryRef&) = delete;
  BdaSparsePhysicalMemoryRef& operator=(const BdaSparsePhysicalMemoryRef&) =
      delete;

  iree_hal_physical_memory_t* get() const { return memory_; }
  iree_hal_physical_memory_t** out() { return &memory_; }

  void reset() {
    if (memory_) {
      IREE_EXPECT_OK(
          iree_hal_allocator_physical_memory_free(allocator_, memory_));
      memory_ = nullptr;
    }
  }

 private:
  // Allocator that owns the physical sparse memory.
  iree_hal_allocator_t* allocator_ = nullptr;

  // Physical sparse memory handle released through |allocator_|.
  iree_hal_physical_memory_t* memory_ = nullptr;
};

TEST_P(BdaRawSpirvTest, QueueDispatchExecutesRawBdaShader) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer, output_buffer, binding_refs);

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectOutput(output_buffer);
}

TEST_P(BdaRawSpirvTest, QueueDispatchExecutesRawBdaShaderWithMetadata) {
  std::vector<uint32_t> metadata_spirv = MakeRawBdaSpirvWithMetadata({
      "iree.vulkan.bda.v1",
      "iree.vulkan.bda.v1.bindings=2",
  });
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(PrepareRawBdaExecutable(
      iree_make_const_byte_span(metadata_spirv.data(),
                                metadata_spirv.size() * sizeof(uint32_t)),
      executable.out()));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer, output_buffer, binding_refs);

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable.get(),
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectOutput(output_buffer);
}

TEST_P(BdaRawSpirvTest, QueueDispatchRejectsRawBdaMetadataBindingMismatch) {
  std::vector<uint32_t> metadata_spirv = MakeRawBdaSpirvWithMetadata({
      "iree.vulkan.bda.v1",
      "iree.vulkan.bda.v1.bindings=2",
  });
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(PrepareRawBdaExecutable(
      iree_make_const_byte_span(metadata_spirv.data(),
                                metadata_spirv.size() * sizeof(uint32_t)),
      executable.out()));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[1] = {
      iree_hal_make_buffer_ref(input_buffer.get(), /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer.get())),
  };
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable.get(),
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(4, 1, 1),
          iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
}

TEST_P(BdaRawSpirvTest, QueueDispatchRejectsRawBdaMetadataBindingLength) {
  std::vector<uint32_t> metadata_spirv = MakeRawBdaSpirvWithMetadata({
      "iree.vulkan.bda.v1",
      "iree.vulkan.bda.v1.bindings=2",
      "iree.vulkan.bda.v1.binding.1=4,17",
  });
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(PrepareRawBdaExecutable(
      iree_make_const_byte_span(metadata_spirv.data(),
                                metadata_spirv.size() * sizeof(uint32_t)),
      executable.out()));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer, output_buffer, binding_refs);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable.get(),
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(4, 1, 1),
          iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
}

TEST_P(BdaRawSpirvTest, QueueDispatchUsesRawBdaMetadataConstantCount) {
  std::vector<uint32_t> metadata_spirv = MakeRawBdaSpirvWithMetadata({
      "iree.vulkan.bda.v1",
      "iree.vulkan.bda.v1.bindings=2",
      "iree.vulkan.bda.v1.constants=1",
  });
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(PrepareRawBdaExecutable(
      iree_make_const_byte_span(metadata_spirv.data(),
                                metadata_spirv.size() * sizeof(uint32_t)),
      executable.out()));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer, output_buffer, binding_refs);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable.get(),
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(4, 1, 1),
          iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));

  const uint32_t ignored_constant = 123u;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable.get(),
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_make_const_byte_span(&ignored_constant, sizeof(ignored_constant)),
      bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectOutput(output_buffer);
}

TEST_P(BdaRawSpirvTest, QueueDispatchHandlesOversizedBdaPublication) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  std::vector<iree_hal_buffer_ref_t> binding_refs =
      MakeOversizedPublicationBindings(input_buffer, output_buffer);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/binding_refs.size(),
      /*.values=*/binding_refs.data(),
  };

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectOutput(output_buffer);
}

TEST_P(BdaRawSpirvTest, CommandBufferExecutesRawBdaShader) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer, output_buffer, binding_refs);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const int64_t initial_one_shot_bypass_count =
      QueryNativeReplayCache(IREE_SV("one_shot_bypass_count"));
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));
  ExpectOutput(output_buffer);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("one_shot_bypass_count")) -
                initial_one_shot_bypass_count,
            1);
}

TEST_P(BdaRawSpirvTest, CommandBufferCachesBdaPublicationRequirements) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer.get(), output_buffer.get(), binding_refs);
  std::vector<iree_hal_buffer_ref_t> oversized_bindings =
      MakeOversizedPublicationBindings(input_buffer.get(), output_buffer.get());

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(),
      iree_hal_buffer_ref_list_t{
          /*.count=*/oversized_bindings.size(),
          /*.values=*/oversized_bindings.data(),
      },
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  EXPECT_TRUE(
      iree_hal_vulkan_command_buffer_has_native_commands(command_buffer));
  EXPECT_EQ(iree_hal_vulkan_command_buffer_dispatch_count(command_buffer), 2u);

  iree_hal_vulkan_command_buffer_descriptor_requirements_t requirements = {0};
  IREE_ASSERT_OK(
      iree_hal_vulkan_command_buffer_native_descriptor_pool_requirements(
          command_buffer, &requirements));
  EXPECT_EQ(requirements.set_count, 0u);
  EXPECT_EQ(requirements.sampler_count, 0u);
  EXPECT_EQ(requirements.uniform_buffer_count, 0u);
  EXPECT_EQ(requirements.storage_buffer_count, 0u);

  iree_device_size_t bda_publication_length = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_native_bda_publication_length(
      command_buffer, &bda_publication_length));
  EXPECT_EQ(bda_publication_length,
            (bindings.count + oversized_bindings.size()) * sizeof(uint64_t));
}

TEST_P(BdaRawSpirvTest, TrimDropsIdleOversizedBdaPublicationBlock) {
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  const int64_t initial_block_count =
      QueryBdaPublicationCache(IREE_SV("block_count"));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);
  std::vector<iree_hal_buffer_ref_t> oversized_bindings =
      MakeOversizedPublicationBindings(input_buffer.get(), output_buffer.get());

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(),
      iree_hal_buffer_ref_list_t{
          /*.count=*/oversized_bindings.size(),
          /*.values=*/oversized_bindings.data(),
      },
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_GT(QueryBdaPublicationCache(IREE_SV("block_count")),
            initial_block_count);
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_LE(QueryBdaPublicationCache(IREE_SV("block_count")),
            initial_block_count);
}

TEST_P(BdaRawSpirvTest, CommandBufferReusesCachedNativeReplay) {
  if (QueryNativeReplayCache(IREE_SV("max_instance_count")) == 0) {
    GTEST_SKIP() << "Vulkan BDA native replay cache is disabled";
  }
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  if (QueryNativeReplayCache(IREE_SV("instance_count")) >=
      QueryNativeReplayCache(IREE_SV("max_instance_count"))) {
    GTEST_SKIP() << "Vulkan BDA native replay cache is already full";
  }

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_entries[2];
  iree_hal_buffer_binding_table_t binding_table = MakeBindingTable(
      input_buffer.get(), output_buffer.get(), binding_table_entries);

  const int64_t initial_hit_count =
      QueryNativeReplayCache(IREE_SV("hit_count"));
  const int64_t initial_miss_count =
      QueryNativeReplayCache(IREE_SV("miss_count"));
  const int64_t initial_create_count =
      QueryNativeReplayCache(IREE_SV("create_count"));
  const int64_t initial_publication_bytes =
      QueryNativeReplayCache(IREE_SV("publication_bytes"));

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());
  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  EXPECT_GE(QueryNativeReplayCache(IREE_SV("miss_count")) - initial_miss_count,
            1);
  EXPECT_GE(
      QueryNativeReplayCache(IREE_SV("create_count")) - initial_create_count,
      1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("hit_count")) - initial_hit_count,
            1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("publication_bytes")) -
                initial_publication_bytes,
            (int64_t)(2 * sizeof(uint64_t)));

  command_buffer.reset();
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
}

TEST_P(BdaRawSpirvTest, CommandBufferProfilingBypassesCachedNativeReplay) {
  if (QueryNativeReplayCache(IREE_SV("max_instance_count")) == 0) {
    GTEST_SKIP() << "Vulkan BDA native replay cache is disabled";
  }

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_entries[2];
  iree_hal_buffer_binding_table_t binding_table = MakeBindingTable(
      input_buffer.get(), output_buffer.get(), binding_table_entries);

  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);
  sink.expected_dispatch_flags =
      IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  sink.expected_dispatch_command_indices = {0};
  sink.expected_workgroup_count[0] = kElementCount;
  sink.expected_workgroup_count[1] = 1;
  sink.expected_workgroup_count[2] = 1;
  DeviceProfilingScope profiling(device_);
  iree_status_t status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                      TestProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(status)) {
    iree_status_ignore(status);
    GTEST_SKIP() << "Vulkan dispatch profiling is unavailable";
  }
  IREE_ASSERT_OK(status);

  const int64_t initial_profile_bypass_count =
      QueryNativeReplayCache(IREE_SV("profile_bypass_count"));
  const int64_t initial_hit_count =
      QueryNativeReplayCache(IREE_SV("hit_count"));
  const int64_t initial_create_count =
      QueryNativeReplayCache(IREE_SV("create_count"));

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());

  EXPECT_GE(QueryNativeReplayCache(IREE_SV("profile_bypass_count")) -
                initial_profile_bypass_count,
            1);
  EXPECT_EQ(initial_hit_count, QueryNativeReplayCache(IREE_SV("hit_count")));
  EXPECT_EQ(initial_create_count,
            QueryNativeReplayCache(IREE_SV("create_count")));
  EXPECT_GE(sink.dispatch_event_count, 1);
}

TEST_P(BdaRawSpirvTest, CommandBufferHandlesOversizedBdaPublication) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  std::vector<iree_hal_buffer_ref_t> binding_refs =
      MakeOversizedPublicationBindings(input_buffer, output_buffer);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/binding_refs.size(),
      /*.values=*/binding_refs.data(),
  };

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));
  ExpectOutput(output_buffer);
}

TEST_P(BdaRawSpirvTest, CommandBufferExecutesRawBdaShaderWithSparseBindings) {
  if (!iree_hal_allocator_supports_virtual_memory(device_allocator_)) {
    GTEST_SKIP() << "Vulkan sparse virtual memory is not available";
  }

  iree_hal_buffer_params_t params = SparseDispatchBufferParams();
  iree_device_size_t minimum_page_size = 0;
  iree_device_size_t recommended_page_size = 0;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      device_allocator_, params, &minimum_page_size, &recommended_page_size));
  ASSERT_NE(0u, minimum_page_size);
  ASSERT_GE(recommended_page_size, minimum_page_size);
  ASSERT_GE(recommended_page_size, 4 * sizeof(int32_t));

  BdaSparseVirtualBufferRef input_buffer(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      device_allocator_, IREE_HAL_QUEUE_AFFINITY_ANY, recommended_page_size,
      input_buffer.out()));
  BdaSparseVirtualBufferRef output_buffer(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      device_allocator_, IREE_HAL_QUEUE_AFFINITY_ANY, recommended_page_size,
      output_buffer.out()));

  BdaSparsePhysicalMemoryRef input_memory(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      device_allocator_, params, recommended_page_size, iree_allocator_system(),
      input_memory.out()));
  BdaSparsePhysicalMemoryRef output_memory(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      device_allocator_, params, recommended_page_size, iree_allocator_system(),
      output_memory.out()));

  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      device_allocator_, input_buffer.get(), /*virtual_offset=*/0,
      input_memory.get(), /*physical_offset=*/0, recommended_page_size));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      device_allocator_, output_buffer.get(), /*virtual_offset=*/0,
      output_memory.get(), /*physical_offset=*/0, recommended_page_size));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      device_allocator_, input_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_PROTECTION_READ_WRITE));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      device_allocator_, output_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_PROTECTION_READ_WRITE));

  const int32_t input_pattern = 5;
  SemaphoreList fill_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      fill_signal, input_buffer.get(), /*target_offset=*/0,
      4 * sizeof(input_pattern), &input_pattern, sizeof(input_pattern),
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer.get(), output_buffer.get(), binding_refs);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  Ref<iree_hal_buffer_t> readback_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(int32_t), readback_buffer.out()));
  SemaphoreList copy_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      copy_signal, output_buffer.get(), /*source_offset=*/0,
      readback_buffer.get(), /*target_offset=*/0, 4 * sizeof(int32_t),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      copy_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<int32_t> output_data = ReadBufferData<int32_t>(readback_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<int32_t>{12, 12, 12, 12}));

  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      device_allocator_, output_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      device_allocator_, input_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size));
}

class BdaRawSpirvReplayCacheTest : public BdaRawSpirvTest {};

TEST_P(BdaRawSpirvReplayCacheTest, TrimDropsIdleCachedNativeReplay) {
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("retained_instance_count")));

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_entries[2];
  iree_hal_buffer_binding_table_t binding_table = MakeBindingTable(
      input_buffer.get(), output_buffer.get(), binding_table_entries);

  const int64_t initial_trim_count =
      QueryNativeReplayCache(IREE_SV("trim_count"));
  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  ASSERT_GE(QueryNativeReplayCache(IREE_SV("instance_count")), 1);
  ASSERT_GE(QueryNativeReplayCache(IREE_SV("publication_bytes")),
            (int64_t)(2 * sizeof(uint64_t)));

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));
  EXPECT_EQ(0, QueryNativeReplayCache(IREE_SV("publication_bytes")));
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("trim_count")) - initial_trim_count,
            1);
}

TEST_P(BdaRawSpirvReplayCacheTest, CommandBufferSkipsUnchangedBdaPublication) {
  ASSERT_GE(QueryNativeReplayCache(IREE_SV("max_instance_count")), 1);
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("retained_instance_count")));

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_entries[2];
  iree_hal_buffer_binding_table_t binding_table = MakeBindingTable(
      input_buffer.get(), output_buffer.get(), binding_table_entries);

  const int64_t initial_hit_count =
      QueryNativeReplayCache(IREE_SV("hit_count"));
  const int64_t initial_miss_count =
      QueryNativeReplayCache(IREE_SV("miss_count"));
  const int64_t initial_publication_skip_count =
      QueryNativeReplayCache(IREE_SV("publication_skip_count"));
  const int64_t initial_publication_update_count =
      QueryNativeReplayCache(IREE_SV("publication_update_count"));

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  const int32_t reset_pattern = 0;
  SemaphoreList fill_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      fill_signal, output_buffer.get(), /*target_offset=*/0,
      kDispatchByteLength, &reset_pattern, sizeof(reset_pattern),
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectFilledOutputPrefix(output_buffer.get(), /*expected_value=*/0);

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  EXPECT_GE(QueryNativeReplayCache(IREE_SV("miss_count")) - initial_miss_count,
            1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("hit_count")) - initial_hit_count,
            1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("publication_skip_count")) -
                initial_publication_skip_count,
            1);
  EXPECT_EQ(initial_publication_update_count,
            QueryNativeReplayCache(IREE_SV("publication_update_count")));
}

TEST_P(BdaRawSpirvReplayCacheTest,
       CommandBufferRepublishesChangedBdaPublication) {
  ASSERT_GE(QueryNativeReplayCache(IREE_SV("max_instance_count")), 1);
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("retained_instance_count")));

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_a_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_a_buffer);
  Ref<iree_hal_buffer_t> output_b_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kDispatchByteLength, output_b_buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_a_entries[2];
  iree_hal_buffer_binding_table_t binding_table_a = MakeBindingTable(
      input_buffer.get(), output_a_buffer.get(), binding_table_a_entries);
  iree_hal_buffer_binding_t binding_table_b_entries[2];
  iree_hal_buffer_binding_table_t binding_table_b = MakeBindingTable(
      input_buffer.get(), output_b_buffer.get(), binding_table_b_entries);

  const int64_t initial_hit_count =
      QueryNativeReplayCache(IREE_SV("hit_count"));
  const int64_t initial_miss_count =
      QueryNativeReplayCache(IREE_SV("miss_count"));
  const int64_t initial_publication_skip_count =
      QueryNativeReplayCache(IREE_SV("publication_skip_count"));
  const int64_t initial_publication_update_count =
      QueryNativeReplayCache(IREE_SV("publication_update_count"));

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table_a);
  ExpectOutput(output_a_buffer.get());
  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table_b);
  ExpectOutput(output_b_buffer.get());

  EXPECT_GE(QueryNativeReplayCache(IREE_SV("miss_count")) - initial_miss_count,
            1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("hit_count")) - initial_hit_count,
            1);
  EXPECT_EQ(initial_publication_skip_count,
            QueryNativeReplayCache(IREE_SV("publication_skip_count")));
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("publication_update_count")) -
                initial_publication_update_count,
            1);
}

TEST_P(BdaRawSpirvReplayCacheTest, ConcurrentExecutionsForkCachedNativeReplay) {
  ASSERT_GE(QueryNativeReplayCache(IREE_SV("max_instance_count")), 2);
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("retained_instance_count")));

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));

  static constexpr uint32_t kLargeDispatchElementCount = 8 * 1024 * 1024;
  const iree_device_size_t dispatch_byte_length =
      kLargeDispatchElementCount * sizeof(int32_t);

  Ref<iree_hal_buffer_t> input_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(dispatch_byte_length, input_buffer.out()));
  Ref<iree_hal_buffer_t> output_a_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(dispatch_byte_length, output_a_buffer.out()));
  Ref<iree_hal_buffer_t> output_b_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(dispatch_byte_length, output_b_buffer.out()));

  const int32_t input_pattern = 5;
  SemaphoreList fill_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      fill_signal, input_buffer.get(), /*target_offset=*/0,
      dispatch_byte_length, &input_pattern, sizeof(input_pattern),
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(kLargeDispatchElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_a_entries[2];
  iree_hal_buffer_binding_table_t binding_table_a = MakeBindingTable(
      input_buffer.get(), output_a_buffer.get(), binding_table_a_entries);
  iree_hal_buffer_binding_t binding_table_b_entries[2];
  iree_hal_buffer_binding_table_t binding_table_b = MakeBindingTable(
      input_buffer.get(), output_b_buffer.get(), binding_table_b_entries);

  const int64_t initial_create_count =
      QueryNativeReplayCache(IREE_SV("create_count"));
  const int64_t initial_fork_count =
      QueryNativeReplayCache(IREE_SV("fork_count"));
  SemaphoreList execute_a_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      execute_a_signal, command_buffer.get(), binding_table_a,
      IREE_HAL_EXECUTE_FLAG_NONE));
  SemaphoreList execute_b_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      execute_b_signal, command_buffer.get(), binding_table_b,
      IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_a_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_b_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectFilledOutputPrefix(output_a_buffer.get(), /*expected_value=*/12);
  ExpectFilledOutputPrefix(output_b_buffer.get(), /*expected_value=*/12);

  EXPECT_GE(
      QueryNativeReplayCache(IREE_SV("create_count")) - initial_create_count,
      2);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("fork_count")) - initial_fork_count,
            1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("peak_instance_count")), 2);

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(BdaRawSpirvTest, {"vulkan_bda"},
                                  {"vulkan_bda_replay_cache"});
CTS_REGISTER_TEST_SUITE_WITH_TAGS(BdaRawSpirvReplayCacheTest,
                                  {"vulkan_bda_replay_cache"}, {});

}  // namespace iree::hal::cts
