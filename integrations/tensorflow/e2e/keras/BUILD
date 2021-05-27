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

# ***THIS FILE DOES NOT BUILD WITH BAZEL***
#
# It is open sourced to enable Bazel->CMake conversion to maintain test coverage
# of our integration tests in open source while we figure out a long term plan
# for our integration testing.

load(
    "@iree//integrations/tensorflow/e2e:iree_e2e_cartesian_product_test_suite.bzl",
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
        deps = [
            "//third_party/py/absl:app",
            "//third_party/py/absl/flags",
            "//third_party/py/iree:pylib_tf_support",
            "//third_party/py/numpy",
            "//third_party/py/tensorflow",
            "//util/debuginfo:signalsafe_addr2line_installer",
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
    failing_configurations = [
        {
            "model": [
                # Timing out on SwiftShader
                "att_mh_rnn",
                "att_rnn",
                "gru",
                # Compilation fails
                "crnn",
                "lstm",
            ],
            "target_backends": "iree_vulkan",
        },
        {
            "model": [
                "att_mh_rnn",
                "att_rnn",
                "crnn",
                "gru",
                "lstm",
            ],
            "target_backends": "iree_llvmaot",
        },
    ],
    matrix = {
        "src": "keyword_spotting_streaming_test.py",
        "reference_backend": "tf",
        "mode": "non_streaming",
        "model": KEYWORD_SPOTTING_MODELS,
        "target_backends": [
            "tf",
            "tflite",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    deps = [
        "//third_party/google_research/google_research/kws_streaming/models:models_lib",
        "//third_party/google_research/google_research/kws_streaming/train:train_lib",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/iree:pylib_tf_support",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/debuginfo:signalsafe_addr2line_installer",
    ],
)

iree_e2e_cartesian_product_test_suite(
    name = "keyword_spotting_internal_streaming_tests",
    failing_configurations = [
        {
            # TFLite cannot compile variables.
            "target_backends": "tflite",
        },
        {
            # These models do not currently support streaming.
            "model": NON_STREAMING_KEYWORD_SPOTTING_MODELS,
        },
        {
            "model": [
                "crnn",  # TODO(b/188221333): Get this test working.
            ],
            "target_backends": "iree_vulkan",
        },
    ],
    matrix = {
        "src": "keyword_spotting_streaming_test.py",
        "reference_backend": "tf",
        "mode": "internal_streaming",
        "model": KEYWORD_SPOTTING_MODELS,
        "target_backends": [
            "tf",
            "tflite",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    deps = [
        "//third_party/google_research/google_research/kws_streaming/models:models_lib",
        "//third_party/google_research/google_research/kws_streaming/train:train_lib",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/iree:pylib_tf_support",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/debuginfo:signalsafe_addr2line_installer",
    ],
)

iree_e2e_cartesian_product_test_suite(
    name = "keyword_spotting_external_streaming_tests",
    failing_configurations = [
        {
            # A bug in keras causes the external steraming conversion to fail
            # when TensorFlow 2.x is used.
            "target_backends": [
                "tf",
                "tflite",
                "iree_llvmaot",
                "iree_vulkan",
            ],
        },
        {
            # These models do not currently support streaming.
            "model": NON_STREAMING_KEYWORD_SPOTTING_MODELS,
        },
    ],
    matrix = {
        "src": "keyword_spotting_streaming_test.py",
        "reference_backend": "tf",
        "mode": "external_streaming",
        "model": KEYWORD_SPOTTING_MODELS,
        "target_backends": [
            "tf",
            "tflite",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    deps = [
        "//third_party/google_research/google_research/kws_streaming/models:models_lib",
        "//third_party/google_research/google_research/kws_streaming/train:train_lib",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/iree:pylib_tf_support",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/debuginfo:signalsafe_addr2line_installer",
    ],
)

py_binary(
    name = "keyword_spotting_streaming_test_manual",
    srcs = ["keyword_spotting_streaming_test.py"],
    main = "keyword_spotting_streaming_test.py",
    python_version = "PY3",
    deps = [
        "//third_party/google_research/google_research/kws_streaming/models:models_lib",
        "//third_party/google_research/google_research/kws_streaming/train:train_lib",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/iree:pylib_tf_support",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/debuginfo:signalsafe_addr2line_installer",
    ],
)
