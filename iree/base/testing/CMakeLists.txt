# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# bazel_to_cmake: DO NOT EDIT (some scotttodd todos remain)

# TODO(scotttodd): clean up bazel_to_cmake handling here
#   * this is a cc_binary in Bazel, but `linkshared` fits iree_cc_library better
#   * the output file name is platform-specific, get it with $<TARGET_FILE:>
iree_cc_library(
  NAME
    dynamic_library_test_library.so
  OUT
    dynamic_library_test_library.so
  SRCS
    "dynamic_library_test_library.cc"
  TESTONLY
  SHARED
)

iree_cc_embed_data(
  NAME
    dynamic_library_test_library
  GENERATED_SRCS
    "$<TARGET_FILE:iree::base::testing::dynamic_library_test_library.so>"
  CC_FILE_OUTPUT
    "dynamic_library_test_library_embed.cc"
  H_FILE_OUTPUT
    "dynamic_library_test_library_embed.h"
  TESTONLY
  CPP_NAMESPACE
    "iree"
  FLATTEN
  PUBLIC
)

iree_cc_test(
  NAME
    dynamic_library_test
  SRCS
    "dynamic_library_test.cc"
  DEPS
    ::dynamic_library_test_library
    iree::base::core_headers
    iree::base::dynamic_library
    iree::base::internal::file_io
    iree::base::status
    iree::testing::gtest
    iree::testing::gtest_main
)
