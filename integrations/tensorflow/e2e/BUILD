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

load(
    "//bindings/python:build_defs.oss.bzl",
    "INTREE_TENSORFLOW_PY_DEPS",
    "NUMPY_DEPS",
)
load(
    "//integrations/tensorflow/e2e:iree_e2e_test_suite.bzl",
    "iree_e2e_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_e2e_test_suite(
    # TODO(GH-2082): `linspace_test.py` fails in the `bazel-tensorflow` image.
    name = "linspace_test",
    srcs = ["linspace_test.py"],
    backends = [
        "iree_vmla",
        "iree_llvmjit",
        "iree_vulkan",
    ],
    reference_backend = "tf",
    tags = [
        "nokokoro",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_e2e_test_suite(
    name = "e2e",
    srcs = glob(
        ["*_test.py"],
        exclude = ["linspace_test.py"],
    ),
    backends = [
        "iree_vmla",
        "iree_llvmjit",
        "iree_vulkan",
    ],
    reference_backend = "tf",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)
