// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/base/status.h"

#include "iree/base/status_matchers.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace {

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

TEST(Status, ConstructedWithMessage) {
  Status status = Status(StatusCode::kInvalidArgument, "message");
  EXPECT_THAT(status.ToString(), HasSubstr("message"));
}

TEST(Statue, Annotate) {
  Status status = Status(StatusCode::kInvalidArgument, "message");
  status = Annotate(status, "annotation");
  EXPECT_THAT(status.ToString(), HasSubstr("message"));
  EXPECT_THAT(status.ToString(), HasSubstr("annotation"));
}

TEST(Status, StreamInsertion) {
  Status status = Status(StatusCode::kInvalidArgument, "message");
  std::ostringstream os;
  os << status;
  EXPECT_THAT(os.str(), HasSubstr("message"));
}

TEST(Status, StreamInsertionContinued) {
  Status status = Status(StatusCode::kInvalidArgument, "message");
  std::ostringstream os;
  os << status << " annotation";
  EXPECT_THAT(os.str(), HasSubstr("message"));
  EXPECT_THAT(os.str(), HasSubstr("annotation"));
}

TEST(StatusBuilder, StreamInsertion) {
  Status status = InvalidArgumentErrorBuilder(IREE_LOC) << "message";
  EXPECT_THAT(status.ToString(), HasSubstr("message"));
}

TEST(StatusBuilder, StreamInsertionMultiple) {
  Status status = InvalidArgumentErrorBuilder(IREE_LOC) << "message"
                                                        << " goes"
                                                        << " like"
                                                        << " this.";
  EXPECT_THAT(status.ToString(), HasSubstr("message goes like this."));
}

TEST(StatusBuilder, StreamInsertionFlag) {
  Status status = InvalidArgumentErrorBuilder(IREE_LOC)
                  << "message " << std::hex << 32;
  EXPECT_THAT(status.ToString(), HasSubstr("message 20"));
}

TEST(StatusMacro, ReturnIfError) {
  auto returnIfError = [](Status status) -> Status {
    RETURN_IF_ERROR(status) << "annotation";
    return OkStatus();
  };
  Status status = InvalidArgumentErrorBuilder(IREE_LOC) << "message";
  status = returnIfError(status);
  EXPECT_THAT(status, StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(), HasSubstr("message"));
  EXPECT_THAT(status.ToString(), HasSubstr("annotation"));

  EXPECT_OK(returnIfError(OkStatus()));
}

TEST(StatusMacro, AssignOrReturn) {
  auto assignOrReturn = [](StatusOr<std::string> statusOr) -> Status {
    ASSIGN_OR_RETURN(auto ret, statusOr, _ << "annotation");
    (void)ret;
    return OkStatus();
  };
  StatusOr<std::string> statusOr = InvalidArgumentErrorBuilder(IREE_LOC)
                                   << "message";
  Status status = assignOrReturn(statusOr);
  EXPECT_THAT(status, StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(), HasSubstr("message"));
  EXPECT_THAT(status.ToString(), HasSubstr("annotation"));

  EXPECT_OK(assignOrReturn("foo"));
}

}  // namespace
}  // namespace iree
