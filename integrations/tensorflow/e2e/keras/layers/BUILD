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
# files. A coverage table generated from this file can be viewed here:
#   https://google.github.io/iree/tf-e2e-coverage
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

LAYERS = [
    "Activation",
    "ActivityRegularization",
    "Add",
    "AdditiveAttention",
    "AlphaDropout",
    "Attention",
    "Average",
    "AveragePooling1D",
    "AveragePooling2D",
    "AveragePooling3D",
    "BatchNormalization",
    "Concatenate",
    "Conv1D",
    "Conv1DTranspose",
    "Conv2D",
    "Conv2DTranspose",
    "Conv3D",
    "Conv3DTranspose",
    "ConvLSTM2D",
    "Cropping1D",
    "Cropping2D",
    "Cropping3D",
    "Dense",
    "DepthwiseConv2D",
    "Dot",
    "Dropout",
    "ELU",
    "Embedding",
    "Flatten",
    "GRU",
    "GRUCell",
    "GaussianDropout",
    "GaussianNoise",
    "GlobalAveragePooling1D",
    "GlobalAveragePooling2D",
    "GlobalAveragePooling3D",
    "GlobalMaxPool1D",
    "GlobalMaxPool2D",
    "GlobalMaxPool3D",
    "InputLayer",
    "LSTM",
    "LSTMCell",
    "Lambda",
    "LayerNormalization",
    "LeakyReLU",
    "LocallyConnected1D",
    "LocallyConnected2D",
    "Masking",
    "MaxPool1D",
    "MaxPool2D",
    "MaxPool3D",
    "Maximum",
    "Minimum",
    "MultiHeadAttention",
    "Multiply",
    "PReLU",
    "Permute",
    "ReLU",
    "RepeatVector",
    "Reshape",
    "SeparableConv1D",
    "SeparableConv2D",
    "SimpleRNN",
    "SimpleRNNCell",
    "Softmax",
    "SpatialDropout1D",
    "SpatialDropout2D",
    "SpatialDropout3D",
    "Subtract",
    "ThresholdedReLU",
    "UpSampling1D",
    "UpSampling2D",
    "UpSampling3D",
    "ZeroPadding1D",
    "ZeroPadding2D",
    "ZeroPadding3D",
]

FAILING_STATIC = [
    {
        # Wrapping these in a tf.function appears to cause a keras bug.
        "layer": [
            "GRUCell",
            "LSTMCell",
            "SimpleRNNCell",
        ],
    },
    {
        # Failing on TFLite
        "layer": [
            "AveragePooling3D",
            "Conv3DTranspose",
            "Conv3D",
            "ConvLSTM2D",
            "LayerNormalization",
            "MaxPool3D",
            "ZeroPadding3D",
        ],
        "target_backends": "tflite",
    },
    {
        # Failing on TFLite and VMLA
        "layer": [
            "Conv3DTranspose",
            "Conv3D",
        ],
        "target_backends": [
            "tflite",
            "iree_vmla",
        ],
    },
    {
        # Failing on IREE
        "layer": [
            "ConvLSTM2D",
            "LayerNormalization",
            "LeakyReLU",
            "LocallyConnected2D",
            "Masking",
            "MultiHeadAttention",
            "SimpleRNN",
            "UpSampling2D",
        ],
        "target_backends": [
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    {
        # Failing on LLVM and Vulkan
        "layer": "Lambda",
        "target_backends": [
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    {
        # Failing on Vulkan
        "layer": [
            "AveragePooling1D",
            "AveragePooling2D",
            "AveragePooling3D",
            "GRU",
            "LSTM",  # TODO(silvasean): Get this test working on Vulkan.
            "MaxPool1D",
            "MaxPool2D",
            "MaxPool3D",
            "ThresholdedReLU",
        ],
        "target_backends": "iree_vulkan",
    },
]

iree_e2e_cartesian_product_test_suite(
    name = "layers_static_batch_tests",
    srcs = ["layers_test.py"],
    failing_configurations = FAILING_STATIC,
    flags_to_values = {
        "reference_backend": "tf",
        "layer": LAYERS,
        "dynamic_batch": False,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    main = "layers_test.py",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

FAILING_DYNAMIC = FAILING_STATIC + [
    {
        # TFLite does not support dynamic shapes.
        "target_backends": "tflite",
    },
    {
        # Failing on IREE
        "layer": [
            "AdditiveAttention",
            "AveragePooling1D",
            "AveragePooling2D",
            "AveragePooling3D",
            "Concatenate",
            "Conv1DTranspose",
            "Conv1D",
            "Conv2DTranspose",
            "Conv2D",
            "Conv3DTranspose",
            "Conv3D",
            "Cropping1D",
            "Cropping2D",
            "Cropping3D",
            "DepthwiseConv2D",
            "ELU",
            "Flatten",
            "GRU",
            "LocallyConnected1D",
            "LSTM",  # TODO(silvasean): Get this test working on IREE.
            "MaxPool1D",
            "MaxPool2D",
            "MaxPool3D",
            "RepeatVector",
            "Reshape",
            "SeparableConv1D",
            "SeparableConv2D",
            "ThresholdedReLU",
            "UpSampling1D",
            "UpSampling3D",
            "ZeroPadding1D",
            "ZeroPadding2D",
            "ZeroPadding3D",
        ],
        "target_backends": [
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    {
        # Failing on LLVM and Vulkan
        "layer": [
            "Activation",
            "Add",
            "Attention",
            "Average",
            "BatchNormalization",
            "Dense",
            "Dot",
            "GlobalAveragePooling1D",
            "GlobalAveragePooling2D",
            "GlobalAveragePooling3D",
            "Maximum",
            "Minimum",
            "Multiply",
            "PReLU",
            "ReLU",
            "Softmax",
            "Subtract",
        ],
        "target_backends": [
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    {
        # Failing on Vulkan
        "layer": [
            "Embedding",
        ],
        "target_backends": "iree_vulkan",
    },
]

iree_e2e_cartesian_product_test_suite(
    name = "layers_dynamic_batch_tests",
    srcs = ["layers_test.py"],
    failing_configurations = FAILING_DYNAMIC,
    flags_to_values = {
        "reference_backend": "tf",
        "layer": LAYERS,
        "dynamic_batch": True,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    main = "layers_test.py",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)
