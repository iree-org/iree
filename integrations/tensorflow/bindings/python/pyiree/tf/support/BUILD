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
    "iree_py_library",
    "iree_py_test",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_py_library(
    name = "support",
    srcs = [
        "__init__.py",
        "tf_test_driver.py",
        "tf_test_utils.py",
        "tf_utils.py",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + [
        "//integrations/tensorflow/bindings/python:pathsetup",  # build_cleaner: keep
        "//integrations/tensorflow/bindings/python/pyiree/tf/compiler",
        "//bindings/python/pyiree/rt",
    ],
)

iree_py_test(
    name = "tf_test_utils_test",
    srcs = [
        "tf_test_utils.py",
        "tf_test_utils_test.py",
    ],
    python_version = "PY3",
    deps = INTREE_TENSORFLOW_PY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_py_test(
    name = "tf_utils_test",
    srcs = [
        "tf_utils.py",
        "tf_utils_test.py",
    ],
    python_version = "PY3",
    tags = [
        "driver=llvm",
        "driver=vmla",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)
