// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/executable.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/internal/flatcc/building.h"
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/metal_executable_def_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::metal {
namespace {

class MetalExecutableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = MTLCreateSystemDefaultDevice();  // +1
    ASSERT_NE(device_, nil);
    ASSERT_EQ(flatcc_builder_init(&builder_), 0);
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    flatcc_builder_clear(&builder_);
    [device_ release];  // -1
  }

  void buildExecutableData(const std::string& code, uint32_t bindingCount) {
    ASSERT_EQ(iree_hal_metal_ExecutableDef_start_as_root(&builder_), 0);
    auto codeRef = flatbuffers_string_create(&builder_, code.data(), code.size());
    auto source = iree_hal_metal_MSLSourceDef_create(&builder_, MTLLanguageVersion3_0, codeRef);
    ASSERT_NE(source, 0);
    ASSERT_EQ(iree_hal_metal_LibraryDef_start(&builder_), 0);
    ASSERT_EQ(iree_hal_metal_LibraryDef_source_add(&builder_, source), 0);
    auto library = iree_hal_metal_LibraryDef_end(&builder_);
    ASSERT_NE(library, 0);
    auto libraries = iree_hal_metal_LibraryDef_vec_create(&builder_, &library, 1);
    ASSERT_NE(libraries, 0);
    auto entry = flatbuffers_string_create_str(&builder_, "main0");
    std::vector<iree_hal_metal_BindingBits_enum_t> flags(bindingCount, 0);
    auto bindings = iree_hal_metal_BindingBits_vec_create(&builder_, flags.data(), flags.size());
    iree_hal_metal_ThreadgroupSize_t groupSize = {1, 1, 1};
    ASSERT_EQ(iree_hal_metal_PipelineDef_start(&builder_), 0);
    ASSERT_EQ(iree_hal_metal_PipelineDef_entry_point_add(&builder_, entry), 0);
    ASSERT_EQ(iree_hal_metal_PipelineDef_threadgroup_size_add(&builder_, &groupSize), 0);
    ASSERT_EQ(iree_hal_metal_PipelineDef_binding_flags_add(&builder_, bindings), 0);
    auto pipeline = iree_hal_metal_PipelineDef_end(&builder_);
    ASSERT_NE(pipeline, 0);
    auto pipelines = iree_hal_metal_PipelineDef_vec_create(&builder_, &pipeline, 1);
    ASSERT_EQ(iree_hal_metal_ExecutableDef_libraries_add(&builder_, libraries), 0);
    ASSERT_EQ(iree_hal_metal_ExecutableDef_pipelines_add(&builder_, pipelines), 0);
    ASSERT_NE(iree_hal_metal_ExecutableDef_end_as_root(&builder_), 0);

    size_t size = 0;
    void* data = flatcc_builder_finalize_aligned_buffer(&builder_, &size);
    ASSERT_NE(data, nullptr);
    iree_flatbuffer_file_header_t header = {};
    std::memcpy(&header.magic, iree_hal_metal_ExecutableDef_file_identifier, sizeof(header.magic));
    header.content_size = size;
    data_.resize(sizeof(header) + size);
    std::memcpy(data_.data(), &header, sizeof(header));
    std::memcpy(data_.data() + sizeof(header), data, size);
    flatcc_builder_aligned_free(data);
  }

  iree_status_t loadExecutable() {
    iree_hal_executable_params_t params;
    iree_hal_executable_params_initialize(&params);
    params.executable_data = iree_make_const_byte_span(data_.data(), data_.size());
    return iree_hal_metal_executable_create(device_, &params, iree_allocator_system(),
                                            &executable_);
  }

  const iree_hal_metal_pipeline_t* lookupPipeline() {
    const iree_hal_metal_pipeline_t* pipeline = nullptr;
    IREE_CHECK_OK(iree_hal_metal_executable_lookup_pipeline(
        executable_, iree_hal_executable_function_from_index(0), &pipeline));
    return pipeline;
  }

  // Binding1 is absent from the shader's argument-buffer type, as happens
  // when scan lowering removes its initializer input.
  static std::string makeSparseShader(uint32_t lastBinding) {
    return "#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct Args { device int* input [[id(0)]]; "
           "device int* output [[id(" +
           std::to_string(lastBinding) +
           ")]]; };\n"
           "kernel void main0(constant Args& args [[buffer(0)]]) { "
           "args.output[0] = args.input[0]; }\n";
  }

  id<MTLDevice> device_ = nil;
  flatbuffers_builder_t builder_ = {};
  std::vector<uint8_t> data_;
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_F(MetalExecutableTest, SparseArgumentBindings) {
  ASSERT_NO_FATAL_FAILURE(buildExecutableData(makeSparseShader(2), 3));
  IREE_ASSERT_OK(loadExecutable());
  EXPECT_EQ(lookupPipeline()->binding_count, 3);
  EXPECT_EQ(lookupPipeline()->binding_used_bits, 0b101u);
}

TEST_F(MetalExecutableTest, HighestArgumentBinding) {
  constexpr uint32_t kCount = IREE_HAL_METAL_MAX_DESCRIPTOR_SET_BINDING_COUNT;
  ASSERT_NO_FATAL_FAILURE(buildExecutableData(makeSparseShader(kCount - 1), kCount));
  IREE_ASSERT_OK(loadExecutable());
  EXPECT_EQ(lookupPipeline()->binding_used_bits, 1ull | (1ull << (kCount - 1)));
}

TEST_F(MetalExecutableTest, NoArgumentBindings) {
  ASSERT_NO_FATAL_FAILURE(buildExecutableData("kernel void main0() {}", 0));
  IREE_ASSERT_OK(loadExecutable());
  EXPECT_EQ(lookupPipeline()->binding_used_bits, 0u);
}

TEST_F(MetalExecutableTest, RejectUndeclaredArgumentBinding) {
  ASSERT_NO_FATAL_FAILURE(buildExecutableData(makeSparseShader(2), 2));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, loadExecutable());
  EXPECT_EQ(executable_, nullptr);
}

}  // namespace
}  // namespace iree::hal::metal
