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

# These layers were selected by:
#   1. Getting all subclasses of `tf.keras.layers.Layer`
#   2. Removing deperacated layers based on the tf.keras docs
#   3. Removing irrelevant layers
#   4. Removing layers that don't fit in the testing framework (Wrappers, DenseFeatures, ...)
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
        # This layer is numerically flaky – TODO(meadowlark) create minimal reproducer.
        "layer": "ConvLSTM2D",
    },
    {
        # Wrapping these in a tf.function appears to cause a keras bug.
        "layer": [
            "ConvLSTM2D",
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
        # Failing on VMLA
        "layer": [
            "Conv3DTranspose",
            "Conv3D",
        ],
        "target_backends": "iree_vmla",
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
    name = "layers_tests",
    srcs = ["layers_test.py"],
    failing_configurations = FAILING_STATIC,
    flags_to_values = {
        "reference_backend": "tf",
        "layer": LAYERS,
        "dynamic_batch": False,
        "training": False,
        "test_full_api": False,
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

# A list of all layers with non-default api tests can be generated by running:
#   bazel run integrations/tensorflow/e2e/keras/layers:layers_test_manual -- \
#     --list_layers_with_full_api_tests
LAYERS_WITH_FULL_API_TESTS = [
    "AdditiveAttention",
    "Attention",
    "AveragePooling1D",
    "AveragePooling2D",
    "AveragePooling3D",
    "BatchNormalization",
    "Conv1D",
    "Conv1DTranspose",
    "Conv2D",
    "Conv2DTranspose",
    "Conv3D",
    "Conv3DTranspose",
    "ConvLSTM2D",
    "DepthwiseConv2D",
    "GlobalAveragePooling1D",
    "GlobalAveragePooling2D",
    "GlobalAveragePooling3D",
    "GlobalMaxPool1D",
    "GlobalMaxPool2D",
    "GlobalMaxPool3D",
    "GRU",
    "LocallyConnected1D",
    "LocallyConnected2D",
    "LSTM",
    "MaxPool1D",
    "MaxPool2D",
    "MaxPool3D",
    "SeparableConv1D",
    "SeparableConv2D",
    "SimpleRNN",
]

FAILING_FULL_API = [
    {
        # This layer is numerically flaky – TODO(meadowlark) create minimal reproducer.
        "layer": "ConvLSTM2D",
    },
    {
        # Failing on TFLite
        "layer": [
            "AveragePooling3D",
            "Conv2DTranspose",
            "Conv3D",
            "Conv3DTranspose",
            "ConvLSTM2D",
            "GRU",
            "LocallyConnected1D",
            "LocallyConnected2D",
            "LSTM",
            "MaxPool1D",
            "MaxPool3D",
            "SimpleRNN",
        ],
        "target_backends": "tflite",
    },
    {
        # Failing on IREE
        "layer": [
            "Conv1D",
            "Conv2D",
            "Conv2DTranspose",
            "Conv3DTranspose",
            "Conv3D",
            "ConvLSTM2D",
            "DepthwiseConv2D",
            "GRU",
            "LocallyConnected1D",
            "LocallyConnected2D",
            "LSTM",
            "SeparableConv1D",
            "SeparableConv2D",
            "SimpleRNN",
        ],
        "target_backends": [
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    {
        # Failing on LLVM and Vulakn
        "layer": [
            "AdditiveAttention",
            "Attention",
            "AveragePooling1D",
            "AveragePooling2D",
            "AveragePooling3D",
            "Conv1DTranspose",
        ],
        "target_backends": [
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    {
        # Failing on Vulkan
        "layer": [
            "MaxPool1D",
            "MaxPool2D",
            "MaxPool3D",
        ],
        "target_backends": "iree_vulkan",
    },
]

iree_e2e_cartesian_product_test_suite(
    name = "layers_full_api_tests",
    srcs = ["layers_test.py"],
    failing_configurations = FAILING_FULL_API,
    flags_to_values = {
        "reference_backend": "tf",
        "layer": LAYERS_WITH_FULL_API_TESTS,
        "dynamic_batch": False,
        "training": False,
        "test_full_api": True,
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

FAILING_DYNAMIC = [
    {
        # This layer is numerically flaky – TODO(meadowlark) create minimal reproducer.
        "layer": "ConvLSTM2D",
    },
    {
        # Wrapping these in a tf.function appears to cause a keras bug.
        "layer": [
            "GRUCell",
            "LSTMCell",
            "SimpleRNNCell",
        ],
    },
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
            "DepthwiseConv2D",
            "ELU",
            "Flatten",
            "GRU",
            "LayerNormalization",
            "LeakyReLU",
            "LocallyConnected1D",
            "LocallyConnected2D",
            "LSTM",  # TODO(silvasean): Get this test working on IREE.
            "Masking",
            "MaxPool1D",
            "MaxPool2D",
            "MaxPool3D",
            "MultiHeadAttention",
            "RepeatVector",
            "Reshape",
            "SeparableConv1D",
            "SeparableConv2D",
            "SimpleRNN",
            "ThresholdedReLU",
            "UpSampling1D",
            "UpSampling2D",
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
            "Lambda",
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
        "layer": "Embedding",
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
        "training": False,
        "test_full_api": False,
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

# Layers that mention a training kwarg in their doc.
LAYERS_WITH_TRAINING_BEHAVIOR = [
    "AdditiveAttention",
    "AlphaDropout",
    "Attention",
    "BatchNormalization",
    "ConvLSTM2D",
    "Dropout",
    "GRU",
    "GRUCell",
    "GaussianDropout",
    "GaussianNoise",
    "LSTM",
    "LSTMCell",
    "MultiHeadAttention",
    "SimpleRNN",
    "SimpleRNNCell",
    "SpatialDropout1D",
    "SpatialDropout2D",
    "SpatialDropout3D",
]

FAILING_TRAINING = [
    {
        # This layer is numerically flaky – TODO(meadowlark) create minimal reproducer.
        "layer": "ConvLSTM2D",
    },
    {
        # Wrapping these in a tf.function appears to cause a keras bug.
        "layer": [
            "GRUCell",
            "LSTMCell",
            "SimpleRNNCell",
        ],
    },
    {
        # Failing on TFLite:
        "layer": [
            "AlphaDropout",
            "BatchNormalization",
            "ConvLSTM2D",
            "GaussianDropout",
            "GaussianNoise",
            "GRU",
            "LSTM",
            "SimpleRNN",
        ],
        "target_backends": "tflite",
    },
    {
        # Failing on IREE
        "layer": [
            "AdditiveAttention",
            "AlphaDropout",
            "Attention",
            "ConvLSTM2D",
            "Dropout",
            "GaussianDropout",
            "GaussianNoise",
            "GRU",
            "LSTM",
            "MultiHeadAttention",
            "SimpleRNN",
            "SpatialDropout1D",
            "SpatialDropout2D",
            "SpatialDropout3D",
        ],
        "target_backends": [
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
]

iree_e2e_cartesian_product_test_suite(
    name = "layers_training_tests",
    srcs = ["layers_test.py"],
    failing_configurations = FAILING_TRAINING,
    flags_to_values = {
        "reference_backend": "tf",
        "layer": LAYERS_WITH_TRAINING_BEHAVIOR,
        "dynamic_batch": False,
        "training": True,
        "test_full_api": False,
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
