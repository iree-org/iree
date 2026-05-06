// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan raw SPIR-V BDA coverage. Raw BDA executables deliberately have no IREE
// flatbuffer metadata; the driver publishes the exact HAL binding table the
// dispatch provides and the shader consumes it through the hidden BDA root.

#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

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

class BdaRawSpirvTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view("vulkan-spirv-bda-raw");
    executable_params.executable_data =
        iree_make_const_byte_span(kRawBdaSpirv, sizeof(kRawBdaSpirv));
    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  void CreateInputOutputBuffers(Ref<iree_hal_buffer_t>* input_buffer,
                                Ref<iree_hal_buffer_t>* output_buffer) {
    const int32_t input_data[4] = {1, -2, 30, 400};
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

  void ExpectOutput(iree_hal_buffer_t* output_buffer) {
    std::vector<int32_t> output_data = ReadBufferData<int32_t>(output_buffer);
    EXPECT_THAT(output_data, ContainerEq(std::vector<int32_t>{8, 5, 37, 407}));
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
      dispatch_signal, executable_, /*export_ordinal=*/0,
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
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
      dispatch_signal, executable_, /*export_ordinal=*/0,
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
      command_buffer, executable_, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));
  ExpectOutput(output_buffer);
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
      command_buffer, executable_, /*entry_point=*/0,
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
      command_buffer, executable_, /*entry_point=*/0,
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

CTS_REGISTER_TEST_SUITE_WITH_TAGS(BdaRawSpirvTest, {"vulkan_bda"}, {});

}  // namespace iree::hal::cts
