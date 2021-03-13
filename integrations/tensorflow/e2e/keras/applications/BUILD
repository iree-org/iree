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
#   https://google.github.io/iree/tensorflow-coverage/vision-coverage
# Updates made to test suite names should also be reflected here:
#   https://github.com/google/iree/blob/main/scripts/update_e2e_coverage.py

load(
    "//e2e:iree_e2e_cartesian_product_test_suite.bzl",
    "iree_e2e_cartesian_product_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

# @unused
DOC = """
applications_test_manual is for manual testing of all keras vision models.
Test will run only manually with all parameters specified manually, for example:
bazel run -c opt integrations/tensorflow/e2e/keras/applications:applications_test_manual -- \
--target_backends=tf,iree_vmla \
--data=imagenet \
--url=https://storage.googleapis.com/iree_models/ \
--model=ResNet50

Command arguments description:
--target_backends: can be combination of these: tf,iree_vmla
--data: can be 'imagenet' or 'cifar10'.
    imagenet - input image size (1, 224, 224, 3)
    cifar10 - input image size (1, 32, 32, 3) - it is used for quick tests
            and needs pretrained weights, we pretrained models: ResNet50, MobileNet, MobileNetV2
--include_top: Whether or not to include the final (top) layers of the model.
--url: we need it only for cifar10 models to load weights from https://storage.googleapis.com/iree_models/
       imagenet pretrained weights url is specified by keras
--model: supports ResNet50, MobileNet, MobileNetV2, ResNet101, ResNet152,
    ResNet50V2, ResNet101V2, ResNet152V2, VGG16, VGG19, Xception,
    InceptionV3, InceptionResNetV2, DenseNet121, DenseNet169,
    DenseNet201, NASNetMobile, NASNetLarge
    All above models works with 'imagenet' data sets.
    ResNet50, MobileNet, MobileNetV2 work with both 'imagenet' and 'cifar10' data sets.
"""

[
    py_binary(
        name = src.replace(".py", "_manual"),
        srcs = [src],
        main = src,
        python_version = "PY3",
    )
    for src in glob(["*_test.py"])
]

KERAS_APPLICATIONS_MODELS = [
    "DenseNet121",
    "DenseNet169",
    "DenseNet201",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
    "EfficientNetB7",
    "InceptionResNetV2",
    "InceptionV3",
    "MobileNet",
    "MobileNetV2",
    "MobileNetV3Large",
    "MobileNetV3Small",
    "NASNetLarge",
    "NASNetMobile",
    "ResNet101",
    "ResNet101V2",
    "ResNet152",
    "ResNet152V2",
    "ResNet50",
    "ResNet50V2",
    "VGG16",
    "VGG19",
]

iree_e2e_cartesian_product_test_suite(
    name = "large_cifar10_tests",
    size = "large",
    matrix = {
        "src": "applications_test.py",
        "reference_backend": "tf",
        "data": "cifar10",
        "model": [
            # All models with runtime shorter than ResNet50.
            "MobileNet",  # Max: Vulkan 61.0s
            "MobileNetV2",  # Max: LLVM 96.3s
            "ResNet50",  # Max: LLVM 145.6s
            "VGG16",  # Max: LLVM 89.5s
            "VGG19",  # Max: LLVM 94.7s
        ],
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    tags = ["manual"],
)

iree_e2e_cartesian_product_test_suite(
    name = "enormous_cifar10_tests",
    size = "enormous",
    failing_configurations = [
        {
            # Failing on vmla with negative inputs.
            "model": [
                "NASNetLarge",
                "NASNetMobile",
            ],
            "target_backends": "iree_vmla",
        },
        {
            # Failing on llvm and vulkan:
            "model": [
                "NASNetLarge",
                "NASNetMobile",
                "ResNet50V2",
                "ResNet101V2",
                "ResNet152V2",
            ],
            "target_backends": [
                "iree_llvmaot",
                "iree_vulkan",
            ],
        },
    ],
    matrix = {
        "src": "applications_test.py",
        "reference_backend": "tf",
        "data": "cifar10",
        "model": [
            "DenseNet121",
            "DenseNet169",
            "DenseNet201",
            "NASNetLarge",
            "NASNetMobile",
            "ResNet50V2",
            "ResNet101",
            "ResNet101V2",
            "ResNet152",
            "ResNet152V2",
        ],
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    tags = [
        "guitar",
        "manual",
        "nokokoro",
        "notap",
    ],
)

# 'non_hermetic' tests use real model weights to test numerical correctness.
iree_e2e_cartesian_product_test_suite(
    name = "cifar10_non_hermetic_tests",
    size = "large",
    matrix = {
        "src": "applications_test.py",
        "reference_backend": "tf",
        "data": "cifar10",
        "url": "https://storage.googleapis.com/iree_models/",
        "use_external_weights": True,
        "model": [
            "MobileNet",
            "MobileNetV2",
            "ResNet50",
        ],
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    tags = [
        "external",
        "guitar",
        "manual",
        "no-remote",
        "nokokoro",
        "notap",
    ],
)

# 'non_hermetic' tests use real model weights to test numerical correctness.
iree_e2e_cartesian_product_test_suite(
    name = "imagenet_non_hermetic_tests",
    size = "enormous",
    failing_configurations = [
        {
            # Failing on vmla with negative inputs.
            "model": [
                "NASNetLarge",
                "NASNetMobile",
            ],
            "target_backends": "iree_vmla",
        },
        {
            # Failing vulkan:
            "model": [
                "InceptionResNetV2",
                "InceptionV3",
            ],
            "target_backends": [
                "iree_vulkan",
            ],
        },
        {
            # Failing llvm and vulkan:
            "model": [
                "NASNetLarge",
                "NASNetMobile",
                "ResNet50V2",
                "ResNet101V2",
                "ResNet152V2",
                "Xception",
            ],
            "target_backends": [
                "iree_llvmaot",
                "iree_vulkan",
            ],
        },
    ],
    matrix = {
        "src": "applications_test.py",
        "reference_backend": "tf",
        "data": "imagenet",
        "use_external_weights": True,
        "model": KERAS_APPLICATIONS_MODELS,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    tags = [
        "external",
        "guitar",
        "manual",
        "nokokoro",
        "notap",
    ],
)

# It is used to produce weights for keras vision models with input image size
# 32x32. These models are not optimized for accuracy or latency (they are for
# debugging only). They have the same neural net topology with keras vision
# models trained on imagenet data sets
py_binary(
    name = "train_vision_models_on_cifar",
    srcs = ["train_vision_models_on_cifar.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
)
