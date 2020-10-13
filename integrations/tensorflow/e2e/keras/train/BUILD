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
    "//integrations/tensorflow/e2e:iree_e2e_product_test_suite.bzl",
    "iree_e2e_product_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_e2e_product_test_suite(
    name = "train_tests",
    srcs = ["model_train_test.py"],
    failing_configurations = [
        {
            "target_backends": [
                "tflite",
                "iree_vmla",  # TODO(b/157581521)
                "iree_llvmjit",
                "iree_vulkan",
            ],
        },
    ],
    flags_to_values = {
        "reference_backend": "tf",
        "optimizer_name": [
            "sgd",
            "adam",
        ],
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    main = "model_train_test.py",
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
