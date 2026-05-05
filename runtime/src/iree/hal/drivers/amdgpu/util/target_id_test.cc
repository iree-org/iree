// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/target_id.h"

#include <cstring>
#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static constexpr iree_hal_amdgpu_target_id_parse_flags_t
    kArchFeatureParseFlags =
        IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY |
        IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_FEATURE_SUFFIXES;

static iree_hal_amdgpu_target_id_t ParseTargetId(
    const char* value, iree_hal_amdgpu_target_id_parse_flags_t flags =
                           IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_NONE) {
  iree_hal_amdgpu_target_id_t target_id;
  IREE_CHECK_OK(iree_hal_amdgpu_target_id_parse(iree_make_cstring_view(value),
                                                flags, &target_id));
  return target_id;
}

static std::string FormatTargetId(
    const iree_hal_amdgpu_target_id_t* target_id) {
  char buffer[64] = {0};
  IREE_CHECK_OK(iree_hal_amdgpu_target_id_format(
      target_id, sizeof(buffer), buffer, /*out_buffer_length=*/nullptr));
  return std::string(buffer);
}

TEST(TargetIdTest, ParsesExactProcessor) {
  auto target_id = ParseTargetId("gfx1100");
  EXPECT_EQ(target_id.kind, IREE_HAL_AMDGPU_TARGET_KIND_EXACT);
  EXPECT_EQ(target_id.version.major, 11u);
  EXPECT_EQ(target_id.version.minor, 0u);
  EXPECT_EQ(target_id.version.stepping, 0u);
  EXPECT_TRUE(iree_string_view_equal(target_id.processor, IREE_SV("gfx1100")));
  EXPECT_EQ(target_id.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED);
  EXPECT_EQ(target_id.xnack, IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED);
}

TEST(TargetIdTest, ParsesExactProcessorWithHexStepping) {
  auto target_id = ParseTargetId("gfx90a");
  EXPECT_EQ(target_id.kind, IREE_HAL_AMDGPU_TARGET_KIND_EXACT);
  EXPECT_EQ(target_id.version.major, 9u);
  EXPECT_EQ(target_id.version.minor, 0u);
  EXPECT_EQ(target_id.version.stepping, 10u);
}

TEST(TargetIdTest, ParsesGenericProcessor) {
  auto target_id = ParseTargetId("gfx9-4-generic");
  EXPECT_EQ(target_id.kind, IREE_HAL_AMDGPU_TARGET_KIND_GENERIC);
  EXPECT_EQ(target_id.version.major, 9u);
  EXPECT_EQ(target_id.version.minor, 4u);
  EXPECT_EQ(target_id.version.stepping, 0u);
  EXPECT_TRUE(
      iree_string_view_equal(target_id.processor, IREE_SV("gfx9-4-generic")));

  target_id = ParseTargetId("gfx11-generic");
  EXPECT_EQ(target_id.kind, IREE_HAL_AMDGPU_TARGET_KIND_GENERIC);
  EXPECT_EQ(target_id.version.major, 11u);
  EXPECT_EQ(target_id.version.minor, 0u);
  EXPECT_EQ(target_id.version.stepping, 0u);
}

TEST(TargetIdTest, ParsesKnownFeatureSupport) {
  auto target_id = ParseTargetId("gfx942");
  EXPECT_EQ(target_id.sramecc, IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY);
  EXPECT_EQ(target_id.xnack, IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY);

  target_id = ParseTargetId("gfx1030");
  EXPECT_EQ(target_id.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED);
  EXPECT_EQ(target_id.xnack, IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED);

  target_id = ParseTargetId("gfx1013");
  EXPECT_EQ(target_id.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED);
  EXPECT_EQ(target_id.xnack, IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY);
}

TEST(TargetIdTest, ParsesHsaIsaNameWithFeatureSuffixes) {
  iree_hal_amdgpu_target_id_t target_id;
  IREE_ASSERT_OK(iree_hal_amdgpu_target_id_parse_hsa_isa_name(
      IREE_SV("amdgcn-amd-amdhsa--gfx942:xnack-:sramecc+"), &target_id));
  EXPECT_EQ(target_id.kind, IREE_HAL_AMDGPU_TARGET_KIND_EXACT);
  EXPECT_EQ(target_id.version.major, 9u);
  EXPECT_EQ(target_id.version.minor, 4u);
  EXPECT_EQ(target_id.version.stepping, 2u);
  EXPECT_EQ(target_id.sramecc, IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON);
  EXPECT_EQ(target_id.xnack, IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF);
  EXPECT_EQ(FormatTargetId(&target_id), "gfx942:sramecc+:xnack-");
}

TEST(TargetIdTest, RejectsUnsupportedSyntax) {
  iree_hal_amdgpu_target_id_t target_id;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_target_id_parse(IREE_SV("amdgcn-amd-amdhsa--gfx942"),
                                      IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_NONE,
                                      &target_id));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_target_id_parse(IREE_SV("gfx942:xnack+"),
                                      IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_NONE,
                                      &target_id));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_target_id_parse(IREE_SV("gfx942foo"),
                                      IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_NONE,
                                      &target_id));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_target_id_parse(IREE_SV("gfx942:xnack+:xnack-"),
                                      kArchFeatureParseFlags, &target_id));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_target_id_parse(IREE_SV("gfx942:wavefrontsize64+"),
                                      kArchFeatureParseFlags, &target_id));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_target_id_parse(IREE_SV("gfx942:"),
                                      kArchFeatureParseFlags, &target_id));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_target_id_parse(IREE_SV("gfx942:xnack+:"),
                                      kArchFeatureParseFlags, &target_id));
}

TEST(TargetIdTest, FormatsIntoQueriedBufferLength) {
  auto target_id =
      ParseTargetId("gfx942:sramecc+:xnack-", kArchFeatureParseFlags);
  iree_host_size_t required_length = 0;
  IREE_EXPECT_OK(iree_hal_amdgpu_target_id_format(
      &target_id, /*buffer_capacity=*/0, /*buffer=*/nullptr, &required_length));
  EXPECT_EQ(required_length, strlen("gfx942:sramecc+:xnack-"));

  char buffer[8] = {0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_target_id_format(&target_id, sizeof(buffer), buffer,
                                       &required_length));
  EXPECT_EQ(required_length, strlen("gfx942:sramecc+:xnack-"));
}

TEST(TargetIdTest, LooksUpCodeObjectTarget) {
  auto target_id =
      ParseTargetId("gfx942:sramecc+:xnack-", kArchFeatureParseFlags);
  iree_hal_amdgpu_target_id_t code_object_target_id;
  IREE_ASSERT_OK(iree_hal_amdgpu_target_id_lookup_code_object_target(
      &target_id, &code_object_target_id));
  EXPECT_EQ(code_object_target_id.kind, IREE_HAL_AMDGPU_TARGET_KIND_GENERIC);
  EXPECT_TRUE(iree_string_view_equal(code_object_target_id.processor,
                                     IREE_SV("gfx9-4-generic")));
  EXPECT_EQ(code_object_target_id.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON);
  EXPECT_EQ(code_object_target_id.xnack,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF);

  target_id = ParseTargetId("gfx908");
  IREE_ASSERT_OK(iree_hal_amdgpu_target_id_lookup_code_object_target(
      &target_id, &code_object_target_id));
  EXPECT_EQ(code_object_target_id.kind, IREE_HAL_AMDGPU_TARGET_KIND_EXACT);
  EXPECT_TRUE(iree_string_view_equal(code_object_target_id.processor,
                                     IREE_SV("gfx908")));

  target_id = ParseTargetId("gfx1300");
  IREE_ASSERT_OK(iree_hal_amdgpu_target_id_lookup_code_object_target(
      &target_id, &code_object_target_id));
  EXPECT_EQ(code_object_target_id.kind, IREE_HAL_AMDGPU_TARGET_KIND_EXACT);
  EXPECT_TRUE(iree_string_view_equal(code_object_target_id.processor,
                                     IREE_SV("gfx1300")));
}

TEST(TargetIdTest, ChecksExactCompatibility) {
  auto code_object_target_id = ParseTargetId("gfx1100");
  auto agent_target_id = ParseTargetId("gfx1100");
  EXPECT_EQ(iree_hal_amdgpu_target_id_check_compatible(&code_object_target_id,
                                                       &agent_target_id),
            IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE);

  agent_target_id = ParseTargetId("gfx1101");
  EXPECT_TRUE(iree_any_bit_set(
      iree_hal_amdgpu_target_id_check_compatible(&code_object_target_id,
                                                 &agent_target_id),
      IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_PROCESSOR));
}

TEST(TargetIdTest, ChecksGenericCompatibilityWithMappedFamily) {
  auto code_object_target_id = ParseTargetId("gfx11-generic");
  auto agent_target_id = ParseTargetId("gfx1100");
  EXPECT_EQ(iree_hal_amdgpu_target_id_check_compatible(&code_object_target_id,
                                                       &agent_target_id),
            IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE);

  code_object_target_id = ParseTargetId("gfx9-4-generic");
  agent_target_id = ParseTargetId("gfx942");
  EXPECT_EQ(iree_hal_amdgpu_target_id_check_compatible(&code_object_target_id,
                                                       &agent_target_id),
            IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE);

  code_object_target_id = ParseTargetId("gfx9-generic");
  EXPECT_TRUE(iree_any_bit_set(
      iree_hal_amdgpu_target_id_check_compatible(&code_object_target_id,
                                                 &agent_target_id),
      IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY));

  code_object_target_id = ParseTargetId("gfx9-4-generic");
  agent_target_id = ParseTargetId("gfx940");
  EXPECT_EQ(iree_hal_amdgpu_target_id_check_compatible(&code_object_target_id,
                                                       &agent_target_id),
            IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE);

  code_object_target_id = ParseTargetId("gfx9-generic");
  EXPECT_TRUE(iree_any_bit_set(
      iree_hal_amdgpu_target_id_check_compatible(&code_object_target_id,
                                                 &agent_target_id),
      IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY));

  code_object_target_id = ParseTargetId("gfx12-5-generic");
  agent_target_id = ParseTargetId("gfx1250");
  EXPECT_EQ(iree_hal_amdgpu_target_id_check_compatible(&code_object_target_id,
                                                       &agent_target_id),
            IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE);

  code_object_target_id = ParseTargetId("gfx12-generic");
  EXPECT_TRUE(iree_any_bit_set(
      iree_hal_amdgpu_target_id_check_compatible(&code_object_target_id,
                                                 &agent_target_id),
      IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY));
}

TEST(TargetIdTest, ChecksFeatureCompatibility) {
  auto code_object_target_id =
      ParseTargetId("gfx942:xnack+", kArchFeatureParseFlags);
  auto agent_target_id = ParseTargetId("gfx942:xnack-", kArchFeatureParseFlags);
  EXPECT_TRUE(
      iree_any_bit_set(iree_hal_amdgpu_target_id_check_compatible(
                           &code_object_target_id, &agent_target_id),
                       IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_XNACK));

  code_object_target_id = ParseTargetId("gfx942");
  EXPECT_EQ(iree_hal_amdgpu_target_id_check_compatible(&code_object_target_id,
                                                       &agent_target_id),
            IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE);
}

TEST(TargetIdTest, FormatsCompatibilityReasons) {
  char buffer[64] = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_target_compatibility_format(
      IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY |
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_SRAMECC |
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_XNACK,
      sizeof(buffer), buffer, /*out_buffer_length=*/nullptr));
  EXPECT_STREQ(buffer, "generic family, sramecc, xnack");
}

}  // namespace
}  // namespace iree::hal::amdgpu
