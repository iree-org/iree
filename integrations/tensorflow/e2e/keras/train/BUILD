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
    "//e2e:iree_e2e_cartesian_product_test_suite.bzl",
    "iree_e2e_cartesian_product_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

[
    py_binary(
        name = src.replace(".py", "_manual"),
        srcs = [src],
        main = src,
        python_version = "PY3",
    )
    for src in glob(["*_test.py"])
]

iree_e2e_cartesian_product_test_suite(
    name = "training_tests",
    failing_configurations = [
        {
            "target_backends": [
                "tflite",  # TFLite doesn't support training
                # TODO(b/157581521)
                "iree_vmla",
                "iree_llvmaot",
                "iree_vulkan",
            ],
        },
    ],
    matrix = {
        "src": [
            "classification_training_test.py",
            "regression_training_test.py",
        ],
        "reference_backend": "tf",
        "optimizer": [
            "adadelta",
            "adagrad",
            "adam",
            "adamax",
            "ftrl",
            "nadam",
            "rmsprop",
            "sgd",
        ],
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
)
