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

# bazel_to_cmake: DO NOT EDIT (tracing rule move)

iree_add_all_subdirs()

iree_cc_library(
  NAME
    api
  HDRS
    "api.h"
  SRCS
    "api.c"
  DEPS
    ::core_headers
    ::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    core_headers
  HDRS
    "alignment.h"
    "attributes.h"
    "math.h"
    "target_platform.h"
  DEPS
    absl::core_headers
  PUBLIC
)

iree_cc_library(
  NAME
    target_platform
  HDRS
    "target_platform.h"
  PUBLIC
)

iree_cc_library(
  NAME
    dynamic_library
  HDRS
    "dynamic_library.h"
  SRCS
    "dynamic_library_posix.cc"
    "dynamic_library_win32.cc"
  DEPS
    ::core_headers
    ::logging
    ::status
    ::tracing
    absl::memory
    absl::span
    absl::strings
    iree::base::internal::file_path
  PUBLIC
)

iree_cc_library(
  NAME
    flatcc
  HDRS
    "flatcc.h"
  DEPS
    ::flatcc_dummy
    flatcc::runtime
  PUBLIC
)

flatbuffer_c_library(
  NAME
    flatcc_dummy
  SRCS
    "flatcc.fbs"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
    "--json"
  PUBLIC
)

iree_cc_library(
  NAME
    logging
  HDRS
    "logging.h"
  SRCS
    "logging.cc"
  DEPS
    ::core_headers
    ::tracing
    absl::flags
    absl::str_format
  PUBLIC
)

iree_cc_library(
  NAME
    signature_mangle
  HDRS
    "signature_mangle.h"
  SRCS
    "signature_mangle.cc"
  DEPS
    ::status
    absl::inlined_vector
    absl::memory
    absl::optional
    absl::span
    absl::strings
  PUBLIC
)

iree_cc_test(
  NAME
    signature_mangle_test
  SRCS
    "signature_mangle_test.cc"
  DEPS
    ::signature_mangle
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    status
  HDRS
    "status.h"
  DEPS
    iree::base::internal::status_internal
  PUBLIC
)

iree_cc_test(
  NAME
    status_test
  SRCS
    "status_test.cc"
  DEPS
    ::status
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    synchronization
  HDRS
    "synchronization.h"
  SRCS
    "synchronization.c"
  DEPS
    ::api
    ::core_headers
    ::tracing
    iree::base::internal
  PUBLIC
)

iree_cc_binary(
  NAME
    synchronization_benchmark
  SRCS
    "synchronization_benchmark.cc"
  DEPS
    ::synchronization
    benchmark
    iree::testing::benchmark_main
  TESTONLY
)

iree_run_binary_test(
  NAME
    synchronization_benchmark_test
  TEST_BINARY
    ::synchronization_benchmark
  ARGS
    "--benchmark_min_time=0"
)

iree_cc_test(
  NAME
    synchronization_test
  SRCS
    "synchronization_test.cc"
  DEPS
    ::synchronization
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    threading
  HDRS
    "threading.h"
  SRCS
    "threading.c"
    "threading_darwin.c"
    "threading_impl.h"
    "threading_pthreads.c"
    "threading_win32.c"
  DEPS
    ::api
    ::core_headers
    ::synchronization
    ::tracing
    iree::base::internal
  PUBLIC
)

iree_cc_test(
  NAME
    threading_benchmark
  SRCS
    "threading_benchmark.cc"
  DEPS
    ::threading
    benchmark
    iree::testing::benchmark_main
)

iree_cc_test(
  NAME
    threading_test
  SRCS
    "threading_impl.h"
    "threading_test.cc"
  DEPS
    ::synchronization
    ::threading
    iree::testing::gtest
    iree::testing::gtest_main
)

# TODO(benvanik): evaluate if we want this as part of the API. Could restrict it
# to excusively static linkage scenarios and note that it's unstable. It's just
# really really useful and the only way for applications to interleave with our
# tracing (today).
if(${IREE_ENABLE_RUNTIME_TRACING})
  iree_cc_library(
    NAME
      tracing
    HDRS
      "tracing.h"
      "${IREE_ROOT_DIR}/third_party/tracy/Tracy.hpp"
      "${IREE_ROOT_DIR}/third_party/tracy/TracyC.h"
    SRCS
      "tracing.cc"
    DEPS
      ::core_headers
    DEFINES
      # TODO(#2114): Change the mode to 2.
      "IREE_TRACING_MODE=1"
    PUBLIC
  )
else()
  iree_cc_library(
    NAME
      tracing
    HDRS
      "tracing.h"
    DEPS
      ::core_headers
    PUBLIC
  )
endif()
