# Copyright 2020 Google LLC
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

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "check_test",
    srcs = ["check_test.cc"],
    deps = [
        ":native_module",
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/base:status",
        "//iree/base/internal",
        "//iree/hal",
        "//iree/hal/vmla/registration",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "native_module",
    testonly = True,
    srcs = ["native_module.cc"],
    hdrs = ["native_module.h"],
    deps = [
        "//iree/base:api",
        "//iree/base:status",
        "//iree/base/internal",
        "//iree/hal",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/vm",
        "//iree/vm:cc",
    ],
)
