// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
#define CHECK_STATUS_MESSAGE(status, message_substr)         \
  EXPECT_THAT(status.ToString(),                             \
              HasSubstr(StatusCodeToString(status.code()))); \
  EXPECT_THAT(status.ToString(), HasSubstr(message_substr))
#define CHECK_STREAM_MESSAGE(status, os, message_substr)               \
  EXPECT_THAT(os.str(), HasSubstr(StatusCodeToString(status.code()))); \
  EXPECT_THAT(os.str(), HasSubstr(message_substr))
#else
#define CHECK_STATUS_MESSAGE(status, message_substr) \
  EXPECT_THAT(status.ToString(), HasSubstr(StatusCodeToString(status.code())));
#define CHECK_STREAM_MESSAGE(status, os, message_substr) \
  EXPECT_THAT(os.str(), HasSubstr(StatusCodeToString(status.code())));
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS

TEST(Status, ConstructedWithMessage) {
  Status status = Status(StatusCode::kInvalidArgument, "message");
  CHECK_STATUS_MESSAGE(status, "message");
}

TEST(Status, StreamInsertion) {
  Status status = Status(StatusCode::kInvalidArgument, "message");
  std::ostringstream os;
  os << status;
  CHECK_STREAM_MESSAGE(status, os, "message");
}

TEST(Status, StreamInsertionContinued) {
  Status status = Status(StatusCode::kInvalidArgument, "message");
  std::ostringstream os;
  os << status << " annotation";
  CHECK_STREAM_MESSAGE(status, os, "message");
  CHECK_STREAM_MESSAGE(status, os, "annotation");
}

TEST(StatusMacro, ReturnIfError) {
  auto returnIfError = [](iree_status_t status) -> iree_status_t {
    IREE_RETURN_IF_ERROR(status, "annotation");
    return iree_ok_status();
  };
  Status status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "message");
  status = returnIfError(std::move(status));
  EXPECT_THAT(status, StatusIs(StatusCode::kInvalidArgument));
  CHECK_STATUS_MESSAGE(status, "message");
  CHECK_STATUS_MESSAGE(status, "annotation");

  IREE_EXPECT_OK(returnIfError(OkStatus()));
}

TEST(StatusMacro, ReturnIfErrorFormat) {
  auto returnIfError = [](iree_status_t status) -> iree_status_t {
    IREE_RETURN_IF_ERROR(status, "annotation %d %d %d", 1, 2, 3);
    return iree_ok_status();
  };
  Status status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "message");
  status = returnIfError(std::move(status));
  EXPECT_THAT(status, StatusIs(StatusCode::kInvalidArgument));
  CHECK_STATUS_MESSAGE(status, "message");
  CHECK_STATUS_MESSAGE(status, "annotation 1 2 3");

  IREE_EXPECT_OK(returnIfError(OkStatus()));
}

TEST(StatusMacro, AssignOrReturn) {
  auto assignOrReturn = [](StatusOr<std::string> statusOr) -> iree_status_t {
    IREE_ASSIGN_OR_RETURN(auto ret, std::move(statusOr));
    (void)ret;
    return iree_ok_status();
  };
  StatusOr<std::string> statusOr =
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "message");
  Status status = assignOrReturn(std::move(statusOr));
  EXPECT_THAT(status, StatusIs(StatusCode::kInvalidArgument));
  CHECK_STATUS_MESSAGE(status, "message");

  IREE_EXPECT_OK(assignOrReturn("foo"));
}

// Helper: collects iree_status_format_to output into a std::string.
static std::string FormatStatusTo(iree_status_t status) {
  std::string result;
  iree_status_format_to(
      status,
      [](iree_string_view_t chunk, void* user_data) -> bool {
        auto* str = static_cast<std::string*>(user_data);
        str->append(chunk.data, chunk.size);
        return true;
      },
      &result);
  return result;
}

// Helper: collects iree_status_format output into a std::string.
static std::string FormatStatusBuffer(iree_status_t status) {
  iree_host_size_t buffer_length = 0;
  if (!iree_status_format(status, 0, NULL, &buffer_length)) return "<!>";
  std::string result(buffer_length, '\0');
  iree_host_size_t actual_length = 0;
  if (!iree_status_format(status, result.size() + 1,
                          const_cast<char*>(result.data()), &actual_length)) {
    return "<!>";
  }
  result.resize(actual_length);
  return result;
}

TEST(StatusFormatTo, OkStatus) {
  iree_status_t status = iree_ok_status();
  std::string cb_result = FormatStatusTo(status);
  std::string buffer_result = FormatStatusBuffer(status);
  EXPECT_EQ(cb_result, buffer_result);
  EXPECT_THAT(cb_result, HasSubstr("OK"));
}

TEST(StatusFormatTo, CodeOnly) {
  iree_status_t status = iree_status_from_code(IREE_STATUS_INTERNAL);
  std::string cb_result = FormatStatusTo(status);
  std::string buffer_result = FormatStatusBuffer(status);
  EXPECT_EQ(cb_result, buffer_result);
  EXPECT_THAT(cb_result, HasSubstr("INTERNAL"));
}

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0

TEST(StatusFormatTo, WithMessage) {
  iree_status_t status =
      iree_status_allocate(IREE_STATUS_NOT_FOUND, NULL, 0,
                           iree_make_cstring_view("something missing"));
  std::string cb_result = FormatStatusTo(status);
  std::string buffer_result = FormatStatusBuffer(status);
  EXPECT_EQ(cb_result, buffer_result);
  EXPECT_THAT(cb_result, HasSubstr("NOT_FOUND"));
  EXPECT_THAT(cb_result, HasSubstr("something missing"));
  iree_status_free(status);
}

TEST(StatusFormatTo, WithFormatMessage) {
  iree_status_t status =
      iree_status_allocate_f(IREE_STATUS_INVALID_ARGUMENT, NULL, 0,
                             "value %d out of range [%d, %d]", 42, 0, 10);
  std::string cb_result = FormatStatusTo(status);
  std::string buffer_result = FormatStatusBuffer(status);
  EXPECT_EQ(cb_result, buffer_result);
  EXPECT_THAT(cb_result, HasSubstr("INVALID_ARGUMENT"));
  EXPECT_THAT(cb_result, HasSubstr("value 42 out of range [0, 10]"));
  iree_status_free(status);
}

TEST(StatusFormatTo, WithAnnotation) {
  iree_status_t status =
      iree_status_allocate_f(IREE_STATUS_INTERNAL, NULL, 0, "base error %d", 1);
  status = iree_status_annotate_f(status, "annotation %d", 2);
  std::string cb_result = FormatStatusTo(status);
  std::string buffer_result = FormatStatusBuffer(status);
  EXPECT_EQ(cb_result, buffer_result);
  EXPECT_THAT(cb_result, HasSubstr("INTERNAL"));
  EXPECT_THAT(cb_result, HasSubstr("base error 1"));
  EXPECT_THAT(cb_result, HasSubstr("annotation 2"));
  iree_status_free(status);
}

TEST(StatusFormatTo, WithMultipleAnnotations) {
  iree_status_t status =
      iree_status_allocate_f(IREE_STATUS_UNAVAILABLE, NULL, 0, "root cause");
  status = iree_status_annotate_f(status, "layer 1: %s", "retry failed");
  status = iree_status_annotate_f(status, "layer 2: attempt %d of %d", 3, 3);
  std::string cb_result = FormatStatusTo(status);
  std::string buffer_result = FormatStatusBuffer(status);
  EXPECT_EQ(cb_result, buffer_result);
  EXPECT_THAT(cb_result, HasSubstr("UNAVAILABLE"));
  EXPECT_THAT(cb_result, HasSubstr("root cause"));
  EXPECT_THAT(cb_result, HasSubstr("layer 1: retry failed"));
  EXPECT_THAT(cb_result, HasSubstr("layer 2: attempt 3 of 3"));
  iree_status_free(status);
}

TEST(StatusFormatTo, CallbackShortCircuit) {
  iree_status_t status =
      iree_status_allocate_f(IREE_STATUS_INTERNAL, NULL, 0, "error %d", 42);
  status = iree_status_annotate_f(status, "annotation %d", 1);
  status = iree_status_annotate_f(status, "annotation %d", 2);

  // Callback that stops after receiving the first chunk.
  int call_count = 0;
  struct State {
    int* call_count;
    std::string first_chunk;
  } state = {&call_count, {}};
  iree_status_format_to(
      status,
      [](iree_string_view_t chunk, void* user_data) -> bool {
        auto* s = static_cast<State*>(user_data);
        ++(*s->call_count);
        s->first_chunk = std::string(chunk.data, chunk.size);
        return false;  // Stop after first chunk.
      },
      &state);
  // Should have been called exactly once.
  EXPECT_EQ(call_count, 1);
  // The first chunk should be the status code string.
  EXPECT_THAT(state.first_chunk, HasSubstr("INTERNAL"));
  iree_status_free(status);
}

TEST(StatusToString, SinglePass) {
  iree_status_t status =
      iree_status_allocate_f(IREE_STATUS_INTERNAL, NULL, 0, "error %d", 42);
  status = iree_status_annotate_f(status, "context: %s", "test");
  iree_allocator_t allocator = iree_allocator_system();
  char* buffer = NULL;
  iree_host_size_t buffer_length = 0;
  ASSERT_TRUE(
      iree_status_to_string(status, &allocator, &buffer, &buffer_length));
  std::string result(buffer, buffer_length);
  EXPECT_THAT(result, HasSubstr("INTERNAL"));
  EXPECT_THAT(result, HasSubstr("error 42"));
  EXPECT_THAT(result, HasSubstr("context: test"));
  // Verify NUL termination.
  EXPECT_EQ(buffer[buffer_length], '\0');
  iree_allocator_free(allocator, buffer);
  iree_status_free(status);
}

#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS

}  // namespace
}  // namespace iree
