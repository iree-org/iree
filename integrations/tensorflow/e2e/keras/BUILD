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
    "iree_py_test_suite",
    "iree_vision_test_suite",
)

E2E_INTEGRATIONS_DEPS = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
    "//integrations/tensorflow/bindings/python/pyiree/tf/support",
]

iree_py_test_suite(
    name = "non_vision",
    srcs = glob(["*_test.py"]),
    excludes = [
        "keras_vision_model_test.py",
        # TODO(GH-1620): Re-enable after fixing the failure on GitHub Actions.
        "keras_lstm_static_test.py"
    ],
    python_version = "PY3",
    deps = E2E_INTEGRATIONS_DEPS,
)

iree_vision_test_suite(
    name = "keras_vision_models",
    configurations = [
        # tuples of (data, include_top, model_name, backends)
        # "cifar10" has toy models with input 32x32, is good for debugging
        ("cifar10", 1, "ResNet50", "tf,iree_vmla"),
        ("cifar10", 1, "ResNet50", "tf,iree_llvmjit"),
        ("cifar10", 1, "ResNet50", "tf,iree_vulkan"),
    ],
    deps = E2E_INTEGRATIONS_DEPS,
    tags = ["manual"],
)

iree_vision_test_suite(
    name = "keras_vision_models_external",
    configurations = [
        # tuples of (data, include_top, model_name, backends)
        # "cifar10" has toy models with input 32x32, is good for debugging
        # "imagenet" has real model weights for input 224x224
        ("cifar10", 1, "MobileNet", "tf,iree_vmla"),
        ("cifar10", 1, "MobileNet", "tf,iree_vulkan"),  # TODO(b/150244105)
        ("cifar10", 1, "MobileNet", "tf,iree_llvmjit"),  # TODO(b/150244105)
        ("cifar10", 1, "MobileNetV2", "tf,iree_vmla"),
        ("cifar10", 1, "MobileNetV2", "tf,iree_vulkan"),  # TODO(b/150244105)
        ("cifar10", 1, "MobileNetV2", "tf,iree_llvmjit"),  # TODO(b/150244105)
        ("imagenet", 1, "ResNet50", "tf,iree_vmla"),
        ("imagenet", 1, "ResNet50", "tf,iree_vulkan"),
        ("imagenet", 1, "ResNet50", "tf,iree_llvmjit"),
        ("imagenet", 1, "MobileNet", "tf,iree_vmla"),
        ("imagenet", 1, "MobileNet", "tf,iree_vulkan"),  # TODO(b/150244105)
        ("imagenet", 1, "MobileNet", "tf,iree_llvmjit"),  # TODO(b/150244105)
        ("imagenet", 1, "MobileNetV2", "tf,iree_vmla"),
        ("imagenet", 1, "MobileNetV2", "tf,iree_vulkan"),  # TODO(b/150244105)
        ("imagenet", 1, "MobileNetV2", "tf,iree_llvmjit"),  # TODO(b/150244105)
    ],
    external_data = True,
    deps = E2E_INTEGRATIONS_DEPS,
    tags = [
        "external",
        "large",
        "manual",
        "noga",
        "notap",
    ],
)

# It is used to produce weights for keras vision models with input image size
# 32x32. These models are not optimized for accuracy or latency (they are for
# debugging only). They have the same neural net topology with keras vision
# models trained on imagenet data sets
py_binary(
    name = "train_vision_models_on_cifar",
    srcs = [
        "train_vision_models_on_cifar.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = E2E_INTEGRATIONS_DEPS,
)
