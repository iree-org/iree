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
    for src in glob(["*_test.py"])
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
    # "ConvLSTM2D",  # TODO(meadowlark): Debug flakiness.
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
    # "SimpleRNN",  # TODO(meadowlark): Debug flakiness.
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
        # Failing on TFLite
        "layer": [
            "AveragePooling3D",
            "Conv3DTranspose",
            "Conv3D",
            "ConvLSTM2D",
            "LayerNormalization",
            "Softmax",
            "MaxPool3D",
            "ZeroPadding3D",
        ],
        "target_backends": "tflite",
    },
    {
        # Failing on IREE
        "layer": [
            "ConvLSTM2D",
            "GRU",
            "LSTM",  # Failing unless 'return_sequences = True'
            "LayerNormalization",
            "LeakyReLU",
            "LocallyConnected2D",  # TODO(#4065): VMLA raises INVALID_ARGUMENT errors after DeviceQueue failure.
            "MultiHeadAttention",
            "UpSampling2D",
        ],
        "target_backends": [
            "iree_vmla",
            "iree_llvmaot",
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
        "layer": [
            "Lambda",
            "Masking",
            "MaxPool1D",
            "MaxPool2D",
            "MaxPool3D",
            "AveragePooling3D",
        ],
        "target_backends": [
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    {
        # Failing on Vulkan
        "layer": [
            "AveragePooling1D",
            "AveragePooling2D",
            "ThresholdedReLU",
        ],
        "target_backends": "iree_vulkan",
    },
]

iree_e2e_cartesian_product_test_suite(
    name = "layers_tests",
    failing_configurations = FAILING_STATIC,
    matrix = {
        "src": "layers_test.py",
        "reference_backend": "tf",
        "layer": LAYERS,
        "dynamic_dims": False,
        "training": False,
        "test_default_kwargs_only": True,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/iree:pylib_tf_support",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/debuginfo:signalsafe_addr2line_installer",
    ],
)

# A list of all layers with non-default api tests can be generated by running:
#   bazel run integrations/tensorflow/e2e/keras/layers:layers_test_manual -- \
#     --list_layers_with_full_api_tests
LAYERS_WITH_FULL_API_TESTS = [
    "ActivityRegularization",
    "AdditiveAttention",
    "Attention",
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
    # "ConvLSTM2D",  # TODO(meadowlark): Debug flakiness.
    "Cropping1D",
    "Cropping2D",
    "Cropping3D",
    "DepthwiseConv2D",
    "GRU",
    "LSTM",
    "LocallyConnected1D",
    "LocallyConnected2D",
    "MaxPool1D",
    "MaxPool2D",
    "MaxPool3D",
    "SeparableConv1D",
    "SeparableConv2D",
    "SimpleRNN",
    # "SimpleRNN",  # TODO(meadowlark): Debug flakiness.
]

FAILING_FULL_API = [
    {
        # Failing on TFLite
        # keep sorted
        "layer": [
            "AveragePooling3D",
            "Conv2DTranspose",
            "Conv3D",
            "Conv3DTranspose",
            "ConvLSTM2D",
            "DepthwiseConv2D",
            "GRU",
            "LSTM",
            "LocallyConnected1D",
            "LocallyConnected2D",
            "MaxPool1D",
            "MaxPool3D",
            "SeparableConv1D",  # Failing on Kokoro.
            "SeparableConv2D",
            "SimpleRNN",
        ],
        "target_backends": "tflite",
    },
    {
        # Failing on IREE
        "layer": [
            "Conv3DTranspose",
            "ConvLSTM2D",
            "GRU",
            "LocallyConnected1D",
            "LocallyConnected2D",
            "LSTM",
            "SimpleRNN",
        ],
        "target_backends": [
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    {
        "layer": "Conv3D",
        "target_backends": "iree_vmla",
    },
    {
        # Failing on LLVM and Vulakn
        "layer": [
            "AveragePooling1D",
            "AveragePooling2D",
            "AveragePooling3D",
            "Conv1DTranspose",
            "Conv2DTranspose",
            "MaxPool1D",
            "MaxPool2D",
            "MaxPool3D",
        ],
        "target_backends": [
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
]

iree_e2e_cartesian_product_test_suite(
    name = "layers_full_api_tests",
    size = "large",
    failing_configurations = FAILING_FULL_API,
    matrix = {
        "src": "layers_test.py",
        "reference_backend": "tf",
        "layer": LAYERS_WITH_FULL_API_TESTS,
        "dynamic_dims": False,
        "training": False,
        "test_default_kwargs_only": False,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/iree:pylib_tf_support",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/debuginfo:signalsafe_addr2line_installer",
    ],
)

FAILING_DYNAMIC = [
    {
        "layer": [
            "Add",
            "Average",
            "AveragePooling3D",
            "BatchNormalization",
            "Conv3D",
            "Conv3DTranspose",
            "ConvLSTM2D",
            "Dense",
            "GRU",
            "LSTM",
            "Lambda",
            "LayerNormalization",
            "Masking",
            "MaxPool3D",
            "Multiply",
            "PReLU",
            "SimpleRNN",
            "Softmax",
            "ThresholdedReLU",
            "ZeroPadding3D",
        ],
        "target_backends": "tflite",
    },
    {
        # Failing on IREE
        # keep sorted
        "layer": [
            "AdditiveAttention",
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
            "ELU",
            "Flatten",
            "GRU",
            "LSTM",  # TODO(silvasean): Get this test working on IREE.
            "LayerNormalization",
            "LeakyReLU",
            "LocallyConnected1D",
            "LocallyConnected2D",
            "Masking",
            "MaxPool1D",
            "MaxPool2D",
            "MaxPool3D",
            "MultiHeadAttention",
            "PReLU",
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
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    {
        # Failing on LLVM and Vulkan
        # keep sorted
        "layer": [
            "Activation",
            "Add",
            "Attention",
            "Average",
            "GlobalAveragePooling1D",
            "GlobalAveragePooling2D",
            "GlobalAveragePooling3D",
            "GlobalMaxPool1D",
            "GlobalMaxPool2D",
            "GlobalMaxPool3D",
            "Lambda",
            "Maximum",
            "Minimum",
            "Multiply",
            "Permute",
            "ReLU",
            "Softmax",
            "Subtract",
        ],
        "target_backends": [
            "iree_llvmaot",
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
    name = "layers_dynamic_dims_tests",
    failing_configurations = FAILING_DYNAMIC,
    matrix = {
        "src": "layers_test.py",
        "reference_backend": "tf",
        "layer": LAYERS,
        "dynamic_dims": True,
        "training": False,
        "test_default_kwargs_only": True,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/iree:pylib_tf_support",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/debuginfo:signalsafe_addr2line_installer",
    ],
)

# Layers that mention a training kwarg in their doc.
LAYERS_WITH_TRAINING_BEHAVIOR = [
    "AdditiveAttention",
    "AlphaDropout",
    "Attention",
    "BatchNormalization",
    # "ConvLSTM2D",  # TODO(meadowlark): Debug flakiness.
    "Dropout",
    "GRU",
    "GaussianDropout",
    "GaussianNoise",
    "LSTM",
    "MultiHeadAttention",
    # "SimpleRNN",  # TODO(meadowlark): Debug flakiness.
    "SpatialDropout1D",
    "SpatialDropout2D",
    "SpatialDropout3D",
]

FAILING_TRAINING = [
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
        # keep sorted
        "layer": [
            "AdditiveAttention",
            "AlphaDropout",
            "Attention",
            "BatchNormalization",
            "ConvLSTM2D",
            "Dropout",
            "GRU",
            "GaussianDropout",
            "GaussianNoise",
            "LSTM",
            "MultiHeadAttention",
            "SimpleRNN",
            "SpatialDropout1D",
            "SpatialDropout2D",
            "SpatialDropout3D",
        ],
        "target_backends": [
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
]

iree_e2e_cartesian_product_test_suite(
    name = "layers_training_tests",
    failing_configurations = FAILING_TRAINING,
    matrix = {
        "src": "layers_test.py",
        "reference_backend": "tf",
        "layer": LAYERS_WITH_TRAINING_BEHAVIOR,
        "dynamic_dims": False,
        "training": True,
        "test_default_kwargs_only": True,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/iree:pylib_tf_support",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/debuginfo:signalsafe_addr2line_installer",
    ],
)
