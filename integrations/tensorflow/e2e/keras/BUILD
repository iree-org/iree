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
#   https://google.github.io/iree/TensorFlowE2ECoverage
# Updates made to test suite names should also be reflected here:
#   https://github.com/google/iree/blob/main/scripts/update_e2e_coverage.py

load(
    "//bindings/python:build_defs.oss.bzl",
    "INTREE_TENSORFLOW_PY_DEPS",
    "NUMPY_DEPS",
    "iree_py_binary",
)
load(
    "//integrations/tensorflow/e2e/keras:iree_vision_test_suite.bzl",
    "iree_vision_test_suite",
)
load(
    "//integrations/tensorflow/e2e:iree_e2e_test_suite.bzl",
    "iree_e2e_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

# @unused
DOC = """
vision_model_test_manual is for manual testing of all keras vision models.
Test will run only manually with all parameters specified manually, for example:
bazel run -c opt integrations/tensorflow/e2e/keras:vision_model_test_manual -- \
--target_backends=tf,iree_vmla,iree_llvmjit \
--data=imagenet \
--include_top=1 \
--url=https://storage.googleapis.com/iree_models/ \
--model=ResNet50

Command arguments description:
--target_backends: can be combination of these: tf,iree_vmla,iree_llvmjit
--data: can be 'imagenet' or 'cifar10'.
    imagenet - input image size (1, 224, 224, 3)
    cifar10 - input image size (1, 32, 32, 3) - it is used for quick tests
            and needs pretrained weights, we pretrained models: ResNet50, MobileNet, MobileNetV2
--include_top: can be 1 or 0. Include top layer 1, not include top layer 0
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
    iree_py_binary(
        name = src.replace(".py", "_manual"),
        srcs = [src],
        main = src,
        python_version = "PY3",
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    for src in glob(["*_test.py"])
]

SPECIAL_CASES = [
    "vision_model_test.py",
]

VMLA_FAILING = [
    "lstm_test.py",
]

LLVM_FAILING = [
    # TODO(silvasean): Get this test working on IREE.
    "lstm_test.py",  # Needs TensorList with current Keras implementation.
    "lstm_static_test.py",  # TODO(silvasean): Get this test working on other backends.
]

VULKAN_FAILING = [
    "lstm_test.py",
    "lstm_static_test.py",
]

TF_PASSING = glob(
    ["*_test.py"],
    exclude = SPECIAL_CASES,
)

VMLA_PASSING = glob(
    ["*_test.py"],
    exclude = VMLA_FAILING + SPECIAL_CASES,
)

LLVM_PASSING = glob(
    ["*_test.py"],
    exclude = LLVM_FAILING + SPECIAL_CASES,
)

VULKAN_PASSING = glob(
    ["*_test.py"],
    exclude = VULKAN_FAILING + SPECIAL_CASES,
)

iree_e2e_test_suite(
    name = "keras_tests",
    backends_to_srcs = {
        "tf": TF_PASSING,
        "iree_vmla": VMLA_PASSING,
        "iree_llvmjit": LLVM_PASSING,
        "iree_vulkan": VULKAN_PASSING,
    },
    reference_backend = "tf",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_e2e_test_suite(
    name = "keras_tests_failing",
    backends_to_srcs = {
        "iree_vmla": VMLA_FAILING,
        "iree_llvmjit": LLVM_FAILING,
        "iree_vulkan": VULKAN_FAILING,
    },
    reference_backend = "tf",
    tags = [
        "manual",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_vision_test_suite(
    name = "vision_internal_tests",
    backends = [
        "tf",
        "iree_vmla",
        "iree_llvmjit",
        "iree_vulkan",
    ],
    datasets = ["cifar10"],
    models = ["ResNet50"],
    reference_backend = "tf",
    tags = ["manual"],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_vision_test_suite(
    name = "vision_external_tests",
    backends = [
        "tf",
        "iree_vmla",
        "iree_llvmjit",
        "iree_vulkan",
    ],
    datasets = [
        "cifar10",
        "imagenet",
    ],
    models = [
        "MobileNet",
        "MobileNetV2",
        "ResNet50",
    ],
    reference_backend = "tf",
    tags = [
        "external",
        "guitar",
        "manual",
        "no-remote",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

# It is used to produce weights for keras vision models with input image size
# 32x32. These models are not optimized for accuracy or latency (they are for
# debugging only). They have the same neural net topology with keras vision
# models trained on imagenet data sets
iree_py_binary(
    name = "train_vision_models_on_cifar",
    srcs = ["train_vision_models_on_cifar.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)
