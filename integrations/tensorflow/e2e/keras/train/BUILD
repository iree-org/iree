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
    "//integrations/tensorflow/e2e/keras/train:iree_train_test_suite.bzl",
    "iree_train_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(meadowlark): Refactor this rule to match iree_vision_test_suite.bzl
iree_train_test_suite(
    name = "train_tests",
    configurations = [
        # tuples of (optimizer, backends)
        ("sgd", "tf"),
    ],
    tags = [
        "guitar",
        "manual",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_train_test_suite(
    name = "train_tests_failing",
    configurations = [
        # tuples of (optimizer, backends)
        # TODO: Combine this suite with keras_model_train once these tests pass.
        ("sgd", "tf,iree_vmla"),
        ("adam", "tf,iree_vmla"),  # TODO(b/157581521)
    ],
    tags = [
        "manual",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)
