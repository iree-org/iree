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

# Test coverage across backends for e2e tests is defined directly in the BUILD
# files. Coverage tables generated from this file can be viewed here:
#   https://google.github.io/iree/tensorflow-coverage
# Updates made to test suite names should also be reflected here:
#   https://github.com/google/iree/blob/main/scripts/update_e2e_coverage.py

load(
    "//bindings/python:build_defs.oss.bzl",
    "INTREE_TENSORFLOW_PY_DEPS",
    "NUMPY_DEPS",
    "iree_py_binary",
)
load(
    "//integrations/tensorflow/e2e:iree_e2e_cartesian_product_test_suite.bzl",
    "iree_e2e_cartesian_product_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

[
    iree_py_binary(
        name = src.replace(".py", "_manual"),
        srcs = [src],
        main = src,
        python_version = "PY3",
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    for src in glob(
        ["*_test.py"],
        exclude = ["keyword_spotting_streaming_test.py"],
    )
]

# Keyword Spotting Tests:
KEYWORD_SPOTTING_MODELS = [
    "svdf",
    "svdf_resnet",
    "ds_cnn",
    "gru",
    "lstm",
    "cnn_stride",
    "cnn",
    "tc_resnet",
    "crnn",
    "dnn",
    "att_rnn",
    "att_mh_rnn",
    "mobilenet",
    "mobilenet_v2",
    "xception",
    "inception",
    "inception_resnet",
    "ds_tc_resnet",
]

NON_STREAMING_KEYWORD_SPOTTING_MODELS = [
    "att_mh_rnn",
    "att_rnn",
    "ds_cnn",
    "inception",
    "inception_resnet",
    "mobilenet",
    "mobilenet_v2",
    "svdf_resnet",
    "tc_resnet",
    "xception",
]

iree_e2e_cartesian_product_test_suite(
    name = "keyword_spotting_tests",
    srcs = ["keyword_spotting_streaming_test.py"],
    failing_configurations = [
        {
            # TODO(#4065): VMLA raises INVALID_ARGUMENT errors after DeviceQueue failure.
            "model": [
                "att_mh_rnn",
                "att_rnn",
                "crnn",
                "gru",
                "lstm",
            ],
            "target_backends": "iree_vmla",
        },
    ],
    flags_to_values = {
        "reference_backend": "tf",
        "mode": "non_streaming",
        "model": KEYWORD_SPOTTING_MODELS,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    main = "keyword_spotting_streaming_test.py",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        "@kws_streaming//:models_lib",
        "@kws_streaming//:train_lib",
    ],
)

iree_e2e_cartesian_product_test_suite(
    name = "keyword_spotting_internal_streaming_tests",
    srcs = ["keyword_spotting_streaming_test.py"],
    failing_configurations = [
        {
            # TFLite cannot compile variables.
            "target_backends": "tflite",
        },
        {
            # These models do not currently support streaming.
            "model": NON_STREAMING_KEYWORD_SPOTTING_MODELS,
        },
    ],
    flags_to_values = {
        "reference_backend": "tf",
        "mode": "internal_streaming",
        "model": KEYWORD_SPOTTING_MODELS,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    main = "keyword_spotting_streaming_test.py",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        "@kws_streaming//:models_lib",
        "@kws_streaming//:train_lib",
    ],
)

iree_e2e_cartesian_product_test_suite(
    name = "keyword_spotting_external_streaming_tests",
    srcs = ["keyword_spotting_streaming_test.py"],
    failing_configurations = [
        {
            # A bug in keras causes the external steraming conversion to fail
            # when TensorFlow 2.x is used.
            "target_backends": [
                "tf",
                "tflite",
                "iree_vmla",
                "iree_llvmjit",
                "iree_vulkan",
            ],
        },
        {
            # These models do not currently support streaming.
            "model": NON_STREAMING_KEYWORD_SPOTTING_MODELS,
        },
    ],
    flags_to_values = {
        "reference_backend": "tf",
        "mode": "external_streaming",
        "model": KEYWORD_SPOTTING_MODELS,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    main = "keyword_spotting_streaming_test.py",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        "@kws_streaming//:models_lib",
        "@kws_streaming//:train_lib",
    ],
)

iree_py_binary(
    name = "keyword_spotting_streaming_test_manual",
    srcs = ["keyword_spotting_streaming_test.py"],
    main = "keyword_spotting_streaming_test.py",
    python_version = "PY3",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        "@kws_streaming//:models_lib",
        "@kws_streaming//:train_lib",
    ],
)
