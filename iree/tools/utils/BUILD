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

# TODO(b/146898896): Refactor these into more coherent packages.
cc_library(
    name = "vm_util",
    srcs = ["vm_util.cc"],
    hdrs = ["vm_util.h"],
    deps = [
        "//iree/base:signature_mangle",
        "//iree/base:status",
        "//iree/base/internal:file_io",
        "//iree/hal:api",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "vm_util_test",
    srcs = ["vm_util_test.cc"],
    deps = [
        ":vm_util",
        "//iree/base:api",
        "//iree/hal:api",
        "//iree/hal/vmla/registration",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "//iree/vm",
        "@com_google_absl//absl/strings",
    ],
)
