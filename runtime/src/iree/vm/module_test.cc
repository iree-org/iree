// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/module.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

bool operator==(const iree_string_view_t lhs, const iree_string_view_t rhs) {
  return iree_string_view_equal(lhs, rhs);
}

std::ostream& operator<<(std::ostream& stream, iree_string_view_t const& str) {
  stream << '`';
  stream.write(str.data, str.size);
  stream << '`';
  return stream;
}

namespace {

using ::iree::Status;
using ::iree::StatusCode;
using ::iree::StatusOr;
using ::iree::testing::status::IsOkAndHolds;
using ::iree::testing::status::StatusIs;

iree_vm_function_signature_t MakeSignature(const char* cconv) {
  iree_vm_function_signature_t signature = {
      /*.calling_convention=*/iree_make_cstring_view(cconv),
  };
  return signature;
}

static StatusOr<std::pair<iree_string_view_t, iree_string_view_t>>
GetCconvFragments(const char* cconv) {
  auto signature = MakeSignature(cconv);
  iree_string_view_t arguments = iree_string_view_empty();
  iree_string_view_t results = iree_string_view_empty();
  iree_status_t status = iree_vm_function_call_get_cconv_fragments(
      &signature, &arguments, &results);
  if (iree_status_is_ok(status)) {
    return std::make_pair(arguments, results);
  }
  return status;
}

TEST(FunctionCallTest, GetCconvFragments) {
  // Empty cconv strings are treated as `()->()`.
  EXPECT_THAT(GetCconvFragments(""),
              IsOkAndHolds(std::make_pair(IREE_SV(""), IREE_SV(""))));

  // Only version 0 is supported and all others should fail.
  EXPECT_THAT(GetCconvFragments("1"), StatusIs(StatusCode::kUnimplemented));

  EXPECT_THAT(GetCconvFragments("0v"),
              IsOkAndHolds(std::make_pair(IREE_SV(""), IREE_SV(""))));
  EXPECT_THAT(GetCconvFragments("0v_v"),
              IsOkAndHolds(std::make_pair(IREE_SV(""), IREE_SV(""))));
  EXPECT_THAT(GetCconvFragments("0i"),
              IsOkAndHolds(std::make_pair(IREE_SV("i"), IREE_SV(""))));
  EXPECT_THAT(GetCconvFragments("0_i"),
              IsOkAndHolds(std::make_pair(IREE_SV(""), IREE_SV("i"))));
  EXPECT_THAT(GetCconvFragments("0v_i"),
              IsOkAndHolds(std::make_pair(IREE_SV(""), IREE_SV("i"))));
  EXPECT_THAT(GetCconvFragments("0i_f"),
              IsOkAndHolds(std::make_pair(IREE_SV("i"), IREE_SV("f"))));
  EXPECT_THAT(GetCconvFragments("0iIi_fFf"),
              IsOkAndHolds(std::make_pair(IREE_SV("iIi"), IREE_SV("fFf"))));
}

TEST(FunctionCallTest, IsVariadicCconv) {
  EXPECT_FALSE(iree_vm_function_call_is_variadic_cconv(IREE_SV("")));
  EXPECT_FALSE(iree_vm_function_call_is_variadic_cconv(IREE_SV("0i")));
  EXPECT_TRUE(iree_vm_function_call_is_variadic_cconv(IREE_SV("0CiD")));
}

static StatusOr<std::pair<iree_host_size_t, iree_host_size_t>>
CountArgumentsAndResults(const char* cconv) {
  auto signature = MakeSignature(cconv);
  iree_host_size_t argument_count = 0;
  iree_host_size_t result_count = 0;
  iree_status_t status = iree_vm_function_call_count_arguments_and_results(
      &signature, &argument_count, &result_count);
  if (iree_status_is_ok(status)) {
    return std::make_pair(argument_count, result_count);
  }
  return status;
}

TEST(FunctionCallTest, CountArgumentsAndResults) {
  // Variadic functions cannot be counted.
  EXPECT_THAT(CountArgumentsAndResults("0CiD"),
              StatusIs(StatusCode::kInvalidArgument));

  EXPECT_THAT(CountArgumentsAndResults(""), IsOkAndHolds(std::make_pair(0, 0)));
  EXPECT_THAT(CountArgumentsAndResults("0v"),
              IsOkAndHolds(std::make_pair(0, 0)));
  EXPECT_THAT(CountArgumentsAndResults("0v_v"),
              IsOkAndHolds(std::make_pair(0, 0)));
  EXPECT_THAT(CountArgumentsAndResults("0i"),
              IsOkAndHolds(std::make_pair(1, 0)));
  EXPECT_THAT(CountArgumentsAndResults("0i_v"),
              IsOkAndHolds(std::make_pair(1, 0)));
  EXPECT_THAT(CountArgumentsAndResults("0i_i"),
              IsOkAndHolds(std::make_pair(1, 1)));
  EXPECT_THAT(CountArgumentsAndResults("0iIfFr_iIfFr"),
              IsOkAndHolds(std::make_pair(5, 5)));
}

static StatusOr<iree_host_size_t> ComputeCconvFragmentSize(
    const char* cconv, std::vector<uint16_t> segment_sizes = {}) {
  // Convert the dynamically-sized segment size list to the VM size-prefixed
  // format.
  iree_vm_register_list_t* segment_size_list = nullptr;
  if (!segment_sizes.empty()) {
    segment_size_list = (iree_vm_register_list_t*)iree_alloca(
        sizeof(uint16_t) + sizeof(uint16_t) * segment_sizes.size());
    segment_size_list->size = (uint16_t)segment_sizes.size();
    memcpy(&segment_size_list->registers[0], segment_sizes.data(),
           sizeof(uint16_t) * segment_sizes.size());
  }

  iree_host_size_t required_size = 0;
  iree_status_t status = iree_vm_function_call_compute_cconv_fragment_size(
      iree_make_cstring_view(cconv), segment_size_list, &required_size);

  if (iree_status_is_ok(status)) {
    return required_size;
  }
  return status;
}

TEST(FunctionCallTest, ComputeCconvFragmentSize) {
  EXPECT_THAT(ComputeCconvFragmentSize(""), IsOkAndHolds(0));
  EXPECT_THAT(ComputeCconvFragmentSize("i"), IsOkAndHolds(sizeof(int32_t)));
  EXPECT_THAT(ComputeCconvFragmentSize("I"), IsOkAndHolds(sizeof(int64_t)));
  EXPECT_THAT(ComputeCconvFragmentSize("f"), IsOkAndHolds(sizeof(float)));
  EXPECT_THAT(ComputeCconvFragmentSize("F"), IsOkAndHolds(sizeof(double)));
  EXPECT_THAT(ComputeCconvFragmentSize("r"),
              IsOkAndHolds(sizeof(iree_vm_ref_t)));

  // No external trailing padding (to min alignment of int64_t) expected.
  EXPECT_THAT(ComputeCconvFragmentSize("Ii"),
              IsOkAndHolds(sizeof(int64_t) + sizeof(int32_t)));

  // No internal padding (to align the int64_t) expected.
  EXPECT_THAT(ComputeCconvFragmentSize("iI"),
              IsOkAndHolds(sizeof(int32_t) + sizeof(int64_t)));

  // No internal padding for the ref and external padding to min alignment of
  // ref. We bake out the logic here for readability: this is what the function
  // does internally (generically).
  iree_host_size_t iri_size = 0;
  iri_size += sizeof(int32_t);        // `i`
  iri_size += sizeof(iree_vm_ref_t);  // `r`
  iri_size += sizeof(int32_t);        // `i`
  EXPECT_THAT(ComputeCconvFragmentSize("iri"), IsOkAndHolds(iri_size));
}

TEST(FunctionCallTest, ComputeVariadicCconvFragmentSize) {
  // Require a segment size list of the appropriate size is provided if there
  // are any variadic segments.
  EXPECT_THAT(ComputeCconvFragmentSize("CiD", {}),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ComputeCconvFragmentSize("CiDCID", {0}),
              StatusIs(StatusCode::kInvalidArgument));

  // There's always an int32_t used for the span count that is embedded in the
  // ABI storage. Though we have the information in the segment size list used
  // in this test not all callers/callees have the list and hermeticity of the
  // storage is required.
  EXPECT_THAT(ComputeCconvFragmentSize("CfD", {0}),
              IsOkAndHolds(sizeof(int32_t)));
  EXPECT_THAT(ComputeCconvFragmentSize("CfD", {1}),
              IsOkAndHolds(sizeof(int32_t) + 1 * sizeof(float)));
  EXPECT_THAT(ComputeCconvFragmentSize("CfD", {2}),
              IsOkAndHolds(sizeof(int32_t) + 2 * sizeof(float)));

  // Spans have no padding.
  iree_host_size_t iri_size = 0;
  iri_size += sizeof(int32_t);  // span count
  for (iree_host_size_t i = 0; i < 2; ++i) {
    iri_size += sizeof(int32_t);        // `i`
    iri_size += sizeof(iree_vm_ref_t);  // `r`
    iri_size += sizeof(int32_t);        // `i`
  }
  EXPECT_THAT(ComputeCconvFragmentSize("CiriD", {2}), IsOkAndHolds(iri_size));
}

}  // namespace
