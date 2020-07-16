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
    "//iree:build_defs.oss.bzl",
    "INTREE_FILECHECK_TARGET",
)
load(
    "//bindings/python:build_defs.oss.bzl",
    "INTREE_TENSORFLOW_PY_DEPS",
    "iree_py_test",
)
load("//iree:lit_test.bzl", "iree_lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = glob(["*.mlir"]),
    data = [
        "//integrations/tensorflow/compiler:iree-tf-opt",
        "//iree/tools:IreeFileCheck",
    ],
)

iree_py_test(
    name = "saved_model_adopt_exports",
    srcs = [
        "saved_model_adopt_exports.py",
    ],
    args = [
        "--filecheck_binary=$(rootpath %s)" % INTREE_FILECHECK_TARGET,
    ],
    data = [
        INTREE_FILECHECK_TARGET,
    ],
    python_version = "PY3",
    deps = INTREE_TENSORFLOW_PY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)
