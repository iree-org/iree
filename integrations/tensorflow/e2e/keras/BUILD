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

load(
    "//bindings/python:build_defs.oss.bzl",
    "INTREE_TENSORFLOW_PY_DEPS",
    "NUMPY_DEPS",
    "iree_py_test",
)
load(
    "//build_tools/bazel:iree_py_test_suite.bzl",
    "iree_py_test_suite",
)
load(
    "//integrations/tensorflow/e2e/keras:iree_vision_test_suite.bzl",
    "iree_vision_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_py_test(
    # TODO(GH-1620): Include after fixing the failure on the OSS CI.
    name = "keras_lstm_static_test",
    srcs = ["keras_lstm_static_test.py"],
    main = "keras_lstm_static_test.py",
    python_version = "PY3",
    tags = [
        "nokokoro",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

# @unused
DOC = """
keras_vision_models_test is for manual testing of all keras vision models.
Test will run only manually with all parameters specified manually, for example:
bazel run -c opt integrations/tensorflow/e2e/keras/keras_vision_models_test -- \
--override_backends=tf,iree_vmla,iree_llvmjit \
--data=imagenet \
--include_top=1 \
--url=https://storage.googleapis.com/iree_models/ \
--model=ResNet50

Command arguments description:
--override_backends: can be combination of these: tf,iree_vmla,iree_llvmjit
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

iree_py_test(
    name = "keras_vision_models_test",
    srcs = ["keras_vision_model_test.py"],
    main = "keras_vision_model_test.py",
    python_version = "PY3",
    tags = [
        "external",
        "manual",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_py_test_suite(
    name = "non_vision",
    srcs = glob(
        ["*_test.py"],
        exclude = [
            "keras_vision_model_test.py",
            "keras_lstm_static_test.py",
        ],
    ),
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_vision_test_suite(
    name = "keras_vision_models",
    datasets = ["cifar10"],
    models_to_backends = {
        "ResNet50": [
            "iree_llvmjit",
            "iree_vmla",
            "iree_vulkan",
        ],
    },
    reference_backend = "tf",
    tags = ["manual"],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_vision_test_suite(
    name = "keras_vision_models_external",
    datasets = [
        "cifar10",
        "imagenet",
    ],
    external_weights = "https://storage.googleapis.com/iree_models/",
    models_to_backends = {
        ("MobileNet", "MobileNetV2"): ["iree_vmla"],
        "ResNet50": [
            "iree_llvmjit",
            "iree_vmla",
            "iree_vulkan",
        ],
    },
    reference_backend = "tf",
    tags = [
        "external",
        "guitar",
        "manual",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_vision_test_suite(
    # TODO: Combine this suite with keras_vision_models_external once these
    # tests pass.
    name = "keras_vision_models_external_failing",
    datasets = [
        "cifar10",
        "imagenet",
    ],
    external_weights = "https://storage.googleapis.com/iree_models/",
    # TODO(b/150244105): Compiling fails with commands targeting IREE
    # interpreter and vulkan backends for these tests.
    models_to_backends = {
        ("MobileNet", "MobileNetV2"): [
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    reference_backend = "tf",
    tags = [
        "external",
        "manual",
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
py_binary(
    name = "train_vision_models_on_cifar",
    srcs = ["train_vision_models_on_cifar.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)
