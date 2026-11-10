// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/builtins.h"

#include <string.h>

#include "iree/hal/drivers/vulkan/physical_device.h"

typedef struct iree_hal_vulkan_fill_unaligned_push_constants_t {
  // Repeated fill pattern stored in the low bytes.
  uint32_t fill_pattern;

  // Byte width of the fill pattern.
  uint32_t fill_pattern_width;

  // Storage-buffer uint32 element containing the first edge byte.
  uint32_t target_word_index;

  // Byte offset within target_word_index where the edge begins.
  uint32_t target_word_byte_offset;

  // Byte offset into fill_pattern corresponding to the first edge byte.
  uint32_t pattern_byte_offset;

  // Number of bytes to patch in target_word_index.
  uint32_t fill_length_bytes;
} iree_hal_vulkan_fill_unaligned_push_constants_t;

static_assert(sizeof(iree_hal_vulkan_fill_unaligned_push_constants_t) == 24,
              "push constant layout must match the built-in shader");

typedef struct iree_hal_vulkan_update_unaligned_push_constants_t {
  // Source bytes to patch into the target word, stored in the low bytes.
  uint32_t update_word;

  // Storage-buffer uint32 element containing the first edge byte.
  uint32_t target_word_index;

  // Byte offset within target_word_index where the edge begins.
  uint32_t target_word_byte_offset;

  // Number of bytes to patch in target_word_index.
  uint32_t update_length_bytes;
} iree_hal_vulkan_update_unaligned_push_constants_t;

static_assert(sizeof(iree_hal_vulkan_update_unaligned_push_constants_t) == 16,
              "push constant layout must match the built-in shader");

static const uint32_t iree_hal_vulkan_fill_unaligned_spirv[] = {
    0x07230203, 0x00010000, 0x000d000b, 0x00000073, 0x00000000, 0x00020011,
    0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
    0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x00000005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011,
    0x00000001, 0x00000001, 0x00000001, 0x00030047, 0x0000000d, 0x00000002,
    0x00050048, 0x0000000d, 0x00000000, 0x00000023, 0x00000000, 0x00050048,
    0x0000000d, 0x00000001, 0x00000023, 0x00000004, 0x00050048, 0x0000000d,
    0x00000002, 0x00000023, 0x00000008, 0x00050048, 0x0000000d, 0x00000003,
    0x00000023, 0x0000000c, 0x00050048, 0x0000000d, 0x00000004, 0x00000023,
    0x00000010, 0x00050048, 0x0000000d, 0x00000005, 0x00000023, 0x00000014,
    0x00040047, 0x00000027, 0x00000006, 0x00000004, 0x00030047, 0x00000028,
    0x00000003, 0x00050048, 0x00000028, 0x00000000, 0x00000023, 0x00000000,
    0x00040047, 0x0000002a, 0x00000021, 0x00000000, 0x00040047, 0x0000002a,
    0x00000022, 0x00000000, 0x00040047, 0x00000060, 0x0000000b, 0x00000019,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015,
    0x00000006, 0x00000020, 0x00000000, 0x0008001e, 0x0000000d, 0x00000006,
    0x00000006, 0x00000006, 0x00000006, 0x00000006, 0x00000006, 0x00040020,
    0x0000000e, 0x00000009, 0x0000000d, 0x0004003b, 0x0000000e, 0x0000000f,
    0x00000009, 0x00040015, 0x00000010, 0x00000020, 0x00000001, 0x0004002b,
    0x00000010, 0x00000011, 0x00000004, 0x00040020, 0x00000012, 0x00000009,
    0x00000006, 0x0004002b, 0x00000010, 0x00000017, 0x00000001, 0x0004002b,
    0x00000010, 0x0000001b, 0x00000000, 0x0004002b, 0x00000006, 0x0000001e,
    0x00000008, 0x0004002b, 0x00000006, 0x00000022, 0x000000ff, 0x0003001d,
    0x00000027, 0x00000006, 0x0003001e, 0x00000028, 0x00000027, 0x00040020,
    0x00000029, 0x00000002, 0x00000028, 0x0004003b, 0x00000029, 0x0000002a,
    0x00000002, 0x0004002b, 0x00000010, 0x0000002b, 0x00000002, 0x00040020,
    0x0000002e, 0x00000002, 0x00000006, 0x0004002b, 0x00000006, 0x00000034,
    0x00000000, 0x0004002b, 0x00000010, 0x0000003b, 0x00000005, 0x00020014,
    0x0000003e, 0x0004002b, 0x00000010, 0x00000041, 0x00000003, 0x00040017,
    0x0000005e, 0x00000006, 0x00000003, 0x0004002b, 0x00000006, 0x0000005f,
    0x00000001, 0x0006002c, 0x0000005e, 0x00000060, 0x0000005f, 0x0000005f,
    0x0000005f, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x00050041, 0x00000012, 0x0000002c, 0x0000000f,
    0x0000002b, 0x0004003d, 0x00000006, 0x0000002d, 0x0000002c, 0x00060041,
    0x0000002e, 0x0000002f, 0x0000002a, 0x0000001b, 0x0000002d, 0x0004003d,
    0x00000006, 0x00000030, 0x0000002f, 0x000200f9, 0x00000035, 0x000200f8,
    0x00000035, 0x000700f5, 0x00000006, 0x00000072, 0x00000030, 0x00000005,
    0x00000057, 0x00000036, 0x000700f5, 0x00000006, 0x00000071, 0x00000034,
    0x00000005, 0x00000059, 0x00000036, 0x00050041, 0x00000012, 0x0000003c,
    0x0000000f, 0x0000003b, 0x0004003d, 0x00000006, 0x0000003d, 0x0000003c,
    0x000500b0, 0x0000003e, 0x0000003f, 0x00000071, 0x0000003d, 0x000400f6,
    0x00000037, 0x00000036, 0x00000000, 0x000400fa, 0x0000003f, 0x00000036,
    0x00000037, 0x000200f8, 0x00000036, 0x00050041, 0x00000012, 0x00000042,
    0x0000000f, 0x00000041, 0x0004003d, 0x00000006, 0x00000043, 0x00000042,
    0x00050080, 0x00000006, 0x00000045, 0x00000043, 0x00000071, 0x00050084,
    0x00000006, 0x00000048, 0x0000001e, 0x00000045, 0x000500c4, 0x00000006,
    0x0000004b, 0x00000022, 0x00000048, 0x00050041, 0x00000012, 0x00000064,
    0x0000000f, 0x00000011, 0x0004003d, 0x00000006, 0x00000065, 0x00000064,
    0x00050080, 0x00000006, 0x00000067, 0x00000065, 0x00000071, 0x00050041,
    0x00000012, 0x00000068, 0x0000000f, 0x00000017, 0x0004003d, 0x00000006,
    0x00000069, 0x00000068, 0x00050089, 0x00000006, 0x0000006a, 0x00000067,
    0x00000069, 0x00050041, 0x00000012, 0x0000006b, 0x0000000f, 0x0000001b,
    0x0004003d, 0x00000006, 0x0000006c, 0x0000006b, 0x00050084, 0x00000006,
    0x0000006e, 0x0000001e, 0x0000006a, 0x000500c2, 0x00000006, 0x0000006f,
    0x0000006c, 0x0000006e, 0x000500c7, 0x00000006, 0x00000070, 0x0000006f,
    0x00000022, 0x000500c4, 0x00000006, 0x00000051, 0x00000070, 0x00000048,
    0x000400c8, 0x00000006, 0x00000054, 0x0000004b, 0x000500c7, 0x00000006,
    0x00000055, 0x00000072, 0x00000054, 0x000500c5, 0x00000006, 0x00000057,
    0x00000055, 0x00000051, 0x00050080, 0x00000006, 0x00000059, 0x00000071,
    0x00000017, 0x000200f9, 0x00000035, 0x000200f8, 0x00000037, 0x0003003e,
    0x0000002f, 0x00000072, 0x000100fd, 0x00010038,
};

static const uint32_t iree_hal_vulkan_update_unaligned_spirv[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x0000004c, 0x00000000, 0x00020011,
    0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
    0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x00000005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011,
    0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2,
    0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00050005, 0x00000008,
    0x5f646c6f, 0x64726f77, 0x00000000, 0x00040005, 0x0000000a, 0x67726154,
    0x00007465, 0x00050006, 0x0000000a, 0x00000000, 0x64726f77, 0x00000073,
    0x00040005, 0x0000000c, 0x67726174, 0x00007465, 0x00060005, 0x0000000f,
    0x68737550, 0x736e6f43, 0x746e6174, 0x00000073, 0x00060006, 0x0000000f,
    0x00000000, 0x61647075, 0x775f6574, 0x0064726f, 0x00080006, 0x0000000f,
    0x00000001, 0x67726174, 0x775f7465, 0x5f64726f, 0x65646e69, 0x00000078,
    0x00090006, 0x0000000f, 0x00000002, 0x67726174, 0x775f7465, 0x5f64726f,
    0x65747962, 0x66666f5f, 0x00746573, 0x00080006, 0x0000000f, 0x00000003,
    0x61647075, 0x6c5f6574, 0x74676e65, 0x79625f68, 0x00736574, 0x00030005,
    0x00000011, 0x00006370, 0x00030005, 0x00000019, 0x00000069, 0x00060005,
    0x00000026, 0x67726174, 0x735f7465, 0x74666968, 0x00000000, 0x00060005,
    0x0000002e, 0x61647075, 0x735f6574, 0x74666968, 0x00000000, 0x00040005,
    0x00000031, 0x6b73616d, 0x00000000, 0x00050005, 0x00000035, 0x65747962,
    0x6c61765f, 0x00006575, 0x00040047, 0x00000009, 0x00000006, 0x00000004,
    0x00030047, 0x0000000a, 0x00000003, 0x00050048, 0x0000000a, 0x00000000,
    0x00000023, 0x00000000, 0x00040047, 0x0000000c, 0x00000021, 0x00000000,
    0x00040047, 0x0000000c, 0x00000022, 0x00000000, 0x00030047, 0x0000000f,
    0x00000002, 0x00050048, 0x0000000f, 0x00000000, 0x00000023, 0x00000000,
    0x00050048, 0x0000000f, 0x00000001, 0x00000023, 0x00000004, 0x00050048,
    0x0000000f, 0x00000002, 0x00000023, 0x00000008, 0x00050048, 0x0000000f,
    0x00000003, 0x00000023, 0x0000000c, 0x00040047, 0x0000004b, 0x0000000b,
    0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007,
    0x00000007, 0x00000006, 0x0003001d, 0x00000009, 0x00000006, 0x0003001e,
    0x0000000a, 0x00000009, 0x00040020, 0x0000000b, 0x00000002, 0x0000000a,
    0x0004003b, 0x0000000b, 0x0000000c, 0x00000002, 0x00040015, 0x0000000d,
    0x00000020, 0x00000001, 0x0004002b, 0x0000000d, 0x0000000e, 0x00000000,
    0x0006001e, 0x0000000f, 0x00000006, 0x00000006, 0x00000006, 0x00000006,
    0x00040020, 0x00000010, 0x00000009, 0x0000000f, 0x0004003b, 0x00000010,
    0x00000011, 0x00000009, 0x0004002b, 0x0000000d, 0x00000012, 0x00000001,
    0x00040020, 0x00000013, 0x00000009, 0x00000006, 0x00040020, 0x00000016,
    0x00000002, 0x00000006, 0x0004002b, 0x00000006, 0x0000001a, 0x00000000,
    0x0004002b, 0x0000000d, 0x00000021, 0x00000003, 0x00020014, 0x00000024,
    0x0004002b, 0x00000006, 0x00000027, 0x00000008, 0x0004002b, 0x0000000d,
    0x00000028, 0x00000002, 0x0004002b, 0x00000006, 0x00000032, 0x000000ff,
    0x00040017, 0x00000049, 0x00000006, 0x00000003, 0x0004002b, 0x00000006,
    0x0000004a, 0x00000001, 0x0006002c, 0x00000049, 0x0000004b, 0x0000004a,
    0x0000004a, 0x0000004a, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008,
    0x00000007, 0x0004003b, 0x00000007, 0x00000019, 0x00000007, 0x0004003b,
    0x00000007, 0x00000026, 0x00000007, 0x0004003b, 0x00000007, 0x0000002e,
    0x00000007, 0x0004003b, 0x00000007, 0x00000031, 0x00000007, 0x0004003b,
    0x00000007, 0x00000035, 0x00000007, 0x00050041, 0x00000013, 0x00000014,
    0x00000011, 0x00000012, 0x0004003d, 0x00000006, 0x00000015, 0x00000014,
    0x00060041, 0x00000016, 0x00000017, 0x0000000c, 0x0000000e, 0x00000015,
    0x0004003d, 0x00000006, 0x00000018, 0x00000017, 0x0003003e, 0x00000008,
    0x00000018, 0x0003003e, 0x00000019, 0x0000001a, 0x000200f9, 0x0000001b,
    0x000200f8, 0x0000001b, 0x000400f6, 0x0000001d, 0x0000001e, 0x00000000,
    0x000200f9, 0x0000001f, 0x000200f8, 0x0000001f, 0x0004003d, 0x00000006,
    0x00000020, 0x00000019, 0x00050041, 0x00000013, 0x00000022, 0x00000011,
    0x00000021, 0x0004003d, 0x00000006, 0x00000023, 0x00000022, 0x000500b0,
    0x00000024, 0x00000025, 0x00000020, 0x00000023, 0x000400fa, 0x00000025,
    0x0000001c, 0x0000001d, 0x000200f8, 0x0000001c, 0x00050041, 0x00000013,
    0x00000029, 0x00000011, 0x00000028, 0x0004003d, 0x00000006, 0x0000002a,
    0x00000029, 0x0004003d, 0x00000006, 0x0000002b, 0x00000019, 0x00050080,
    0x00000006, 0x0000002c, 0x0000002a, 0x0000002b, 0x00050084, 0x00000006,
    0x0000002d, 0x00000027, 0x0000002c, 0x0003003e, 0x00000026, 0x0000002d,
    0x0004003d, 0x00000006, 0x0000002f, 0x00000019, 0x00050084, 0x00000006,
    0x00000030, 0x00000027, 0x0000002f, 0x0003003e, 0x0000002e, 0x00000030,
    0x0004003d, 0x00000006, 0x00000033, 0x00000026, 0x000500c4, 0x00000006,
    0x00000034, 0x00000032, 0x00000033, 0x0003003e, 0x00000031, 0x00000034,
    0x00050041, 0x00000013, 0x00000036, 0x00000011, 0x0000000e, 0x0004003d,
    0x00000006, 0x00000037, 0x00000036, 0x0004003d, 0x00000006, 0x00000038,
    0x0000002e, 0x000500c2, 0x00000006, 0x00000039, 0x00000037, 0x00000038,
    0x000500c7, 0x00000006, 0x0000003a, 0x00000039, 0x00000032, 0x0004003d,
    0x00000006, 0x0000003b, 0x00000026, 0x000500c4, 0x00000006, 0x0000003c,
    0x0000003a, 0x0000003b, 0x0003003e, 0x00000035, 0x0000003c, 0x0004003d,
    0x00000006, 0x0000003d, 0x00000008, 0x0004003d, 0x00000006, 0x0000003e,
    0x00000031, 0x000400c8, 0x00000006, 0x0000003f, 0x0000003e, 0x000500c7,
    0x00000006, 0x00000040, 0x0000003d, 0x0000003f, 0x0004003d, 0x00000006,
    0x00000041, 0x00000035, 0x000500c5, 0x00000006, 0x00000042, 0x00000040,
    0x00000041, 0x0003003e, 0x00000008, 0x00000042, 0x000200f9, 0x0000001e,
    0x000200f8, 0x0000001e, 0x0004003d, 0x00000006, 0x00000043, 0x00000019,
    0x00050080, 0x00000006, 0x00000044, 0x00000043, 0x00000012, 0x0003003e,
    0x00000019, 0x00000044, 0x000200f9, 0x0000001b, 0x000200f8, 0x0000001d,
    0x00050041, 0x00000013, 0x00000045, 0x00000011, 0x00000012, 0x0004003d,
    0x00000006, 0x00000046, 0x00000045, 0x0004003d, 0x00000006, 0x00000047,
    0x00000008, 0x00060041, 0x00000016, 0x00000048, 0x0000000c, 0x0000000e,
    0x00000046, 0x0003003e, 0x00000048, 0x00000047, 0x000100fd, 0x00010038,
};

static iree_status_t iree_hal_vulkan_fill_unaligned_expand_pattern(
    const uint8_t* pattern, iree_host_size_t pattern_length,
    uint32_t* out_pattern) {
  *out_pattern = 0;
  switch (pattern_length) {
    case 1:
    case 2:
    case 4:
      memcpy(out_pattern, pattern, pattern_length);
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan built-in fill pattern length must be 1, 2, or 4 bytes "
          "(got %" PRIhsz ")",
          pattern_length);
  }
}

static iree_status_t iree_hal_vulkan_builtins_create_compute_pipeline(
    iree_hal_vulkan_builtins_t* builtins, const uint32_t* spirv_code,
    iree_host_size_t spirv_code_size, VkPipeline* out_pipeline) {
  *out_pipeline = VK_NULL_HANDLE;
  VkShaderModule shader_module = VK_NULL_HANDLE;
  VkShaderModuleCreateInfo shader_module_create_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv_code_size,
      .pCode = spirv_code,
  };
  iree_status_t status = iree_vkCreateShaderModule(
      IREE_VULKAN_DEVICE(&builtins->syms), builtins->logical_device,
      &shader_module_create_info, /*pAllocator=*/NULL, &shader_module);
  if (iree_status_is_ok(status)) {
    VkPipelineShaderStageCreateInfo stage_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
    };
    VkComputePipelineCreateInfo pipeline_create_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage_create_info,
        .layout = builtins->storage_buffer_pipeline_layout,
    };
    status = iree_vkCreateComputePipelines(
        IREE_VULKAN_DEVICE(&builtins->syms), builtins->logical_device,
        /*pipelineCache=*/VK_NULL_HANDLE, /*createInfoCount=*/1,
        &pipeline_create_info, /*pAllocator=*/NULL, out_pipeline);
  }
  if (shader_module) {
    iree_vkDestroyShaderModule(IREE_VULKAN_DEVICE(&builtins->syms),
                               builtins->logical_device, shader_module,
                               /*pAllocator=*/NULL);
  }
  return status;
}

iree_status_t iree_hal_vulkan_builtins_initialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_builtins_t* out_builtins) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(out_builtins);
  memset(out_builtins, 0, sizeof(*out_builtins));
  out_builtins->syms = *syms;
  out_builtins->logical_device = logical_device;

  VkDeviceSize min_storage_buffer_offset_alignment =
      physical_device->properties2.properties.limits
          .minStorageBufferOffsetAlignment;
  if (min_storage_buffer_offset_alignment == 0) {
    min_storage_buffer_offset_alignment = 1;
  }
  if ((min_storage_buffer_offset_alignment &
       (min_storage_buffer_offset_alignment - 1)) != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan reported non-power-of-two "
                            "minStorageBufferOffsetAlignment %" PRIu64,
                            (uint64_t)min_storage_buffer_offset_alignment);
  }
  out_builtins->min_storage_buffer_offset_alignment =
      min_storage_buffer_offset_alignment;

  VkDescriptorSetLayoutBinding fill_layout_binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
  };
  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &fill_layout_binding,
  };
  iree_status_t status = iree_vkCreateDescriptorSetLayout(
      IREE_VULKAN_DEVICE(&out_builtins->syms), out_builtins->logical_device,
      &descriptor_set_layout_create_info, /*pAllocator=*/NULL,
      &out_builtins->storage_buffer_descriptor_set_layout);

  if (iree_status_is_ok(status)) {
    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size =
            (uint32_t)sizeof(iree_hal_vulkan_fill_unaligned_push_constants_t),
    };
    VkPipelineLayoutCreateInfo pipeline_layout_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &out_builtins->storage_buffer_descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
    };
    status = iree_vkCreatePipelineLayout(
        IREE_VULKAN_DEVICE(&out_builtins->syms), out_builtins->logical_device,
        &pipeline_layout_create_info, /*pAllocator=*/NULL,
        &out_builtins->storage_buffer_pipeline_layout);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_builtins_create_compute_pipeline(
        out_builtins, iree_hal_vulkan_fill_unaligned_spirv,
        sizeof(iree_hal_vulkan_fill_unaligned_spirv),
        &out_builtins->fill_pipeline);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_builtins_create_compute_pipeline(
        out_builtins, iree_hal_vulkan_update_unaligned_spirv,
        sizeof(iree_hal_vulkan_update_unaligned_spirv),
        &out_builtins->update_pipeline);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_builtins_deinitialize(out_builtins);
  }
  return status;
}

void iree_hal_vulkan_builtins_deinitialize(
    iree_hal_vulkan_builtins_t* builtins) {
  if (!builtins || !builtins->logical_device) return;
  if (builtins->update_pipeline) {
    iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(&builtins->syms),
                           builtins->logical_device, builtins->update_pipeline,
                           /*pAllocator=*/NULL);
  }
  if (builtins->fill_pipeline) {
    iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(&builtins->syms),
                           builtins->logical_device, builtins->fill_pipeline,
                           /*pAllocator=*/NULL);
  }
  if (builtins->storage_buffer_pipeline_layout) {
    iree_vkDestroyPipelineLayout(IREE_VULKAN_DEVICE(&builtins->syms),
                                 builtins->logical_device,
                                 builtins->storage_buffer_pipeline_layout,
                                 /*pAllocator=*/NULL);
  }
  if (builtins->storage_buffer_descriptor_set_layout) {
    iree_vkDestroyDescriptorSetLayout(
        IREE_VULKAN_DEVICE(&builtins->syms), builtins->logical_device,
        builtins->storage_buffer_descriptor_set_layout,
        /*pAllocator=*/NULL);
  }
  memset(builtins, 0, sizeof(*builtins));
}

static iree_status_t iree_hal_vulkan_builtins_record_fill_edge(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDescriptorSet descriptor_set, VkBuffer target_buffer,
    VkDeviceSize fill_offset, VkDeviceSize edge_offset, uint32_t edge_length,
    const uint8_t* pattern, iree_host_size_t pattern_length) {
  if (edge_length == 0) return iree_ok_status();
  if (!descriptor_set) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan built-in fill requires a descriptor set");
  }

  const VkDeviceSize target_word_offset = edge_offset & ~(VkDeviceSize)3;
  const VkDeviceSize descriptor_offset =
      target_word_offset & ~(builtins->min_storage_buffer_offset_alignment - 1);
  const VkDeviceSize target_word_index =
      (target_word_offset - descriptor_offset) / sizeof(uint32_t);
  if (target_word_index > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in fill descriptor alignment "
                            "produced word index %" PRIu64,
                            (uint64_t)target_word_index);
  }

  uint32_t fill_pattern = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_fill_unaligned_expand_pattern(
      pattern, pattern_length, &fill_pattern));

  VkDescriptorBufferInfo buffer_info = {
      .buffer = target_buffer,
      .offset = descriptor_offset,
      .range = VK_WHOLE_SIZE,
  };
  VkWriteDescriptorSet write_info = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buffer_info,
  };
  iree_vkUpdateDescriptorSets(IREE_VULKAN_DEVICE(&builtins->syms),
                              builtins->logical_device,
                              /*descriptorWriteCount=*/1, &write_info,
                              /*descriptorCopyCount=*/0,
                              /*pDescriptorCopies=*/NULL);

  const iree_hal_vulkan_fill_unaligned_push_constants_t constants = {
      .fill_pattern = fill_pattern,
      .fill_pattern_width = (uint32_t)pattern_length,
      .target_word_index = (uint32_t)target_word_index,
      .target_word_byte_offset = (uint32_t)(edge_offset % sizeof(uint32_t)),
      .pattern_byte_offset =
          (uint32_t)((edge_offset - fill_offset) % pattern_length),
      .fill_length_bytes = edge_length,
  };
  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE,
                         builtins->fill_pipeline);
  iree_vkCmdBindDescriptorSets(
      IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
      VK_PIPELINE_BIND_POINT_COMPUTE, builtins->storage_buffer_pipeline_layout,
      /*firstSet=*/0, /*descriptorSetCount=*/1, &descriptor_set,
      /*dynamicOffsetCount=*/0, /*pDynamicOffsets=*/NULL);
  iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                          builtins->storage_buffer_pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, /*offset=*/0,
                          sizeof(constants), &constants);
  iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                     /*groupCountX=*/1, /*groupCountY=*/1,
                     /*groupCountZ=*/1);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_builtins_record_fill_unaligned_impl(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    const VkDescriptorSet* descriptor_sets, uint32_t descriptor_set_count,
    VkBuffer target_buffer, VkDeviceSize target_offset, VkDeviceSize length,
    const uint8_t* pattern, iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (length == 0) return iree_ok_status();
  if (length > UINT64_MAX - target_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in fill range overflows");
  }

  const VkDeviceSize target_end = target_offset + length;
  const VkDeviceSize target_start_word_byte_offset =
      target_offset % sizeof(uint32_t);
  uint32_t descriptor_set_ordinal = 0;
  VkDeviceSize start_edge_length = 0;
  if (target_start_word_byte_offset != 0) {
    start_edge_length = sizeof(uint32_t) - target_start_word_byte_offset;
    if (start_edge_length > length) start_edge_length = length;
  } else if (length < sizeof(uint32_t)) {
    start_edge_length = length;
  }

  iree_status_t status = iree_ok_status();
  if (start_edge_length != 0) {
    if (descriptor_set_ordinal >= descriptor_set_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan built-in fill received too few descriptor sets");
    }
    status = iree_hal_vulkan_builtins_record_fill_edge(
        builtins, command_buffer, descriptor_sets[descriptor_set_ordinal++],
        target_buffer, target_offset, target_offset,
        (uint32_t)start_edge_length, pattern, pattern_length);
  }

  const VkDeviceSize end_edge_length = target_end % sizeof(uint32_t);
  const VkDeviceSize end_edge_offset = target_end - end_edge_length;
  if (iree_status_is_ok(status) && end_edge_length != 0 &&
      end_edge_offset >= target_offset + start_edge_length) {
    if (descriptor_set_ordinal >= descriptor_set_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan built-in fill received too few descriptor sets");
    }
    status = iree_hal_vulkan_builtins_record_fill_edge(
        builtins, command_buffer, descriptor_sets[descriptor_set_ordinal++],
        target_buffer, target_offset, end_edge_offset,
        (uint32_t)end_edge_length, pattern, pattern_length);
  }
  return status;
}

iree_status_t iree_hal_vulkan_builtins_record_fill_unaligned(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDescriptorPool descriptor_pool, VkBuffer target_buffer,
    VkDeviceSize target_offset, VkDeviceSize length, const uint8_t* pattern,
    iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (length == 0) return iree_ok_status();
  if (length > UINT64_MAX - target_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in fill range overflows");
  }
  if (!descriptor_pool) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan built-in fill requires a descriptor pool");
  }
  const uint32_t descriptor_set_count =
      iree_hal_vulkan_builtins_fill_unaligned_descriptor_set_count(
          target_offset, length);
  if (descriptor_set_count == 0) return iree_ok_status();
  VkDescriptorSet descriptor_sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptor_set_layouts[2] = {
      builtins->storage_buffer_descriptor_set_layout,
      builtins->storage_buffer_descriptor_set_layout,
  };
  VkDescriptorSetAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = descriptor_set_count,
      .pSetLayouts = descriptor_set_layouts,
  };
  IREE_RETURN_IF_ERROR(iree_vkAllocateDescriptorSets(
      IREE_VULKAN_DEVICE(&builtins->syms), builtins->logical_device,
      &allocate_info, descriptor_sets));
  return iree_hal_vulkan_builtins_record_fill_unaligned_impl(
      builtins, command_buffer, descriptor_sets, descriptor_set_count,
      target_buffer, target_offset, length, pattern, pattern_length);
}

uint32_t iree_hal_vulkan_builtins_fill_unaligned_descriptor_set_count(
    VkDeviceSize target_offset, VkDeviceSize length) {
  if (length == 0) return 0;
  const VkDeviceSize target_end = target_offset + length;
  const VkDeviceSize target_start_word_byte_offset =
      target_offset % sizeof(uint32_t);
  VkDeviceSize start_edge_length = 0;
  if (target_start_word_byte_offset != 0) {
    start_edge_length = sizeof(uint32_t) - target_start_word_byte_offset;
    if (start_edge_length > length) start_edge_length = length;
  } else if (length < sizeof(uint32_t)) {
    start_edge_length = length;
  }
  uint32_t descriptor_set_count = start_edge_length != 0 ? 1u : 0u;
  const VkDeviceSize end_edge_length = target_end % sizeof(uint32_t);
  const VkDeviceSize end_edge_offset = target_end - end_edge_length;
  if (end_edge_length != 0 &&
      end_edge_offset >= target_offset + start_edge_length) {
    descriptor_set_count = descriptor_set_count + 1;
  }
  return descriptor_set_count;
}

iree_status_t iree_hal_vulkan_builtins_record_fill_unaligned_descriptor_sets(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    const VkDescriptorSet* descriptor_sets, uint32_t descriptor_set_count,
    VkBuffer target_buffer, VkDeviceSize target_offset, VkDeviceSize length,
    const uint8_t* pattern, iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(descriptor_sets);
  return iree_hal_vulkan_builtins_record_fill_unaligned_impl(
      builtins, command_buffer, descriptor_sets, descriptor_set_count,
      target_buffer, target_offset, length, pattern, pattern_length);
}

static iree_status_t iree_hal_vulkan_builtins_record_update_edge(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDescriptorSet descriptor_set, VkBuffer target_buffer,
    VkDeviceSize edge_offset, uint32_t edge_length,
    const uint8_t* source_data) {
  if (edge_length == 0) return iree_ok_status();
  if (!descriptor_set) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan built-in update requires a descriptor set");
  }

  const VkDeviceSize target_word_offset = edge_offset & ~(VkDeviceSize)3;
  const VkDeviceSize descriptor_offset =
      target_word_offset & ~(builtins->min_storage_buffer_offset_alignment - 1);
  const VkDeviceSize target_word_index =
      (target_word_offset - descriptor_offset) / sizeof(uint32_t);
  if (target_word_index > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in update descriptor alignment "
                            "produced word index %" PRIu64,
                            (uint64_t)target_word_index);
  }

  VkDescriptorBufferInfo buffer_info = {
      .buffer = target_buffer,
      .offset = descriptor_offset,
      .range = VK_WHOLE_SIZE,
  };
  VkWriteDescriptorSet write_info = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buffer_info,
  };
  iree_vkUpdateDescriptorSets(IREE_VULKAN_DEVICE(&builtins->syms),
                              builtins->logical_device,
                              /*descriptorWriteCount=*/1, &write_info,
                              /*descriptorCopyCount=*/0,
                              /*pDescriptorCopies=*/NULL);

  uint32_t update_word = 0;
  memcpy(&update_word, source_data, edge_length);
  const iree_hal_vulkan_update_unaligned_push_constants_t constants = {
      .update_word = update_word,
      .target_word_index = (uint32_t)target_word_index,
      .target_word_byte_offset = (uint32_t)(edge_offset % sizeof(uint32_t)),
      .update_length_bytes = edge_length,
  };
  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE,
                         builtins->update_pipeline);
  iree_vkCmdBindDescriptorSets(
      IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
      VK_PIPELINE_BIND_POINT_COMPUTE, builtins->storage_buffer_pipeline_layout,
      /*firstSet=*/0, /*descriptorSetCount=*/1, &descriptor_set,
      /*dynamicOffsetCount=*/0, /*pDynamicOffsets=*/NULL);
  iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                          builtins->storage_buffer_pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, /*offset=*/0,
                          sizeof(constants), &constants);
  iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                     /*groupCountX=*/1, /*groupCountY=*/1,
                     /*groupCountZ=*/1);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_builtins_record_update_unaligned_impl(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    const VkDescriptorSet* descriptor_sets, uint32_t descriptor_set_count,
    VkBuffer target_buffer, VkDeviceSize target_offset, VkDeviceSize length,
    const uint8_t* source_data, iree_host_size_t source_data_length) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (length == 0) return iree_ok_status();
  if (!source_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan built-in update source is NULL");
  }
  if (length > UINT64_MAX - target_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in update range overflows");
  }
  if (length > (VkDeviceSize)source_data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in update source length %" PRIhsz
                            " is smaller than update length %" PRIu64,
                            source_data_length, (uint64_t)length);
  }

  const VkDeviceSize target_end = target_offset + length;
  const VkDeviceSize target_start_word_byte_offset =
      target_offset % sizeof(uint32_t);
  uint32_t descriptor_set_ordinal = 0;
  VkDeviceSize start_edge_length = 0;
  if (target_start_word_byte_offset != 0) {
    start_edge_length = sizeof(uint32_t) - target_start_word_byte_offset;
    if (start_edge_length > length) start_edge_length = length;
  } else if (length < sizeof(uint32_t)) {
    start_edge_length = length;
  }

  iree_status_t status = iree_ok_status();
  if (start_edge_length != 0) {
    if (descriptor_set_ordinal >= descriptor_set_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan built-in update received too few descriptor sets");
    }
    status = iree_hal_vulkan_builtins_record_update_edge(
        builtins, command_buffer, descriptor_sets[descriptor_set_ordinal++],
        target_buffer, target_offset, (uint32_t)start_edge_length, source_data);
  }

  const VkDeviceSize end_edge_length = target_end % sizeof(uint32_t);
  const VkDeviceSize end_edge_offset = target_end - end_edge_length;
  if (iree_status_is_ok(status) && end_edge_length != 0 &&
      end_edge_offset >= target_offset + start_edge_length) {
    if (descriptor_set_ordinal >= descriptor_set_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan built-in update received too few descriptor sets");
    }
    const VkDeviceSize source_offset = end_edge_offset - target_offset;
    status = iree_hal_vulkan_builtins_record_update_edge(
        builtins, command_buffer, descriptor_sets[descriptor_set_ordinal++],
        target_buffer, end_edge_offset, (uint32_t)end_edge_length,
        source_data + (iree_host_size_t)source_offset);
  }
  return status;
}

uint32_t iree_hal_vulkan_builtins_update_unaligned_descriptor_set_count(
    VkDeviceSize target_offset, VkDeviceSize length) {
  return iree_hal_vulkan_builtins_fill_unaligned_descriptor_set_count(
      target_offset, length);
}

iree_status_t iree_hal_vulkan_builtins_record_update_unaligned(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDescriptorPool descriptor_pool, VkBuffer target_buffer,
    VkDeviceSize target_offset, VkDeviceSize length, const uint8_t* source_data,
    iree_host_size_t source_data_length) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (length == 0) return iree_ok_status();
  if (length > UINT64_MAX - target_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in update range overflows");
  }
  if (!descriptor_pool) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan built-in update requires a descriptor pool");
  }
  const uint32_t descriptor_set_count =
      iree_hal_vulkan_builtins_update_unaligned_descriptor_set_count(
          target_offset, length);
  if (descriptor_set_count == 0) return iree_ok_status();
  VkDescriptorSet descriptor_sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptor_set_layouts[2] = {
      builtins->storage_buffer_descriptor_set_layout,
      builtins->storage_buffer_descriptor_set_layout,
  };
  VkDescriptorSetAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = descriptor_set_count,
      .pSetLayouts = descriptor_set_layouts,
  };
  IREE_RETURN_IF_ERROR(iree_vkAllocateDescriptorSets(
      IREE_VULKAN_DEVICE(&builtins->syms), builtins->logical_device,
      &allocate_info, descriptor_sets));
  return iree_hal_vulkan_builtins_record_update_unaligned_impl(
      builtins, command_buffer, descriptor_sets, descriptor_set_count,
      target_buffer, target_offset, length, source_data, source_data_length);
}

iree_status_t iree_hal_vulkan_builtins_record_update_unaligned_descriptor_sets(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    const VkDescriptorSet* descriptor_sets, uint32_t descriptor_set_count,
    VkBuffer target_buffer, VkDeviceSize target_offset, VkDeviceSize length,
    const uint8_t* source_data, iree_host_size_t source_data_length) {
  IREE_ASSERT_ARGUMENT(descriptor_sets);
  return iree_hal_vulkan_builtins_record_update_unaligned_impl(
      builtins, command_buffer, descriptor_sets, descriptor_set_count,
      target_buffer, target_offset, length, source_data, source_data_length);
}
