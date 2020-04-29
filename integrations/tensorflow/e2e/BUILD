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
    "//bindings/python:build_defs.oss.bzl",
    "INTREE_TENSORFLOW_PY_DEPS",
    "NUMPY_DEPS",
    "iree_py_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

[
    iree_py_test(
        name = name,
        srcs = [name + ".py"],
        python_version = "PY3",
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    for name in [
        "broadcasting_test",
        "batch_norm_test",
        "fill_test",
        "control_flow_test",
        "dynamic_mlp_test",
        "dynamic_mlp_relu_test",
        "depth_conv_test",
        "exported_names_test",
        "gather_test",
        "tensorlist_test",
        "keras_lstm_test",
        "mandelbrot_test",
        "matrix_ops_test",
        "ring_buffer_test",
        "sliding_window_test",
        "simple_arithmetic_test",
        "simple_stateful_test",
        "strings_test",
    ]
]

[
    iree_py_test(
        name = "_".join([
            "keras_vision_model",
            data,
            model_name,
            "test",
        ]),
        srcs = [
            "keras_vision_model_test.py",
        ],
        args = [
            "--model=%s" % model_name,
            "--override_backends=%s" % backends,
            "--data=%s" % data,
        ],
        main = "keras_vision_model_test.py",
        python_version = "PY3",
        # TODO(rybakov): Get this down to a manageable size for CI.
        tags = [
            "large",
            "manual",
            "noga",
            "notap",
        ],
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    # TODO(b/150244105) update backends with "tf,iree_vmla,iree_vulkan"
    # and add other supported vision models
    for data, model_name, backends in [
        # "cifar10" has toy models with input 32x32, is good for debugging
        ("cifar10", "ResNet50", "tf,iree_vmla,iree_vulkan,iree_llvmjit"),
    ]
]

# it requres access to external URL, so these tests will be run manually
[
    iree_py_test(
        name = "_".join([
            "keras_vision_model",
            data,
            model_name,
            "test",
        ]),
        srcs = [
            "keras_vision_model_test.py",
        ],
        args = [
            "--model=%s" % model_name,
            "--override_backends=%s" % backends,
            "--data=%s" % data,
            "--url=https://storage.googleapis.com/iree_models/",
        ],
        main = "keras_vision_model_test.py",
        python_version = "PY3",
        tags = [
            "external",
            "large",
            "manual",
            "noga",
            "notap",
        ],
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    # TODO(b/150244105) update backends with "tf,iree_vmla,iree_vulkan"
    # and add other supported vision models
    for data, model_name, backends in [
        # "cifar10" has toy models with input 32x32, is good for debugging
        # "imagenet" has real model weights for input 224x224
        ("cifar10", "MobileNet", "tf,iree_vmla"),
        ("imagenet", "ResNet50", "tf"),
        ("imagenet", "MobileNet", "tf"),
        ("imagenet", "MobileNetV2", "tf"),
    ]
]

[
    iree_py_test(
        name = name,
        srcs = [name + ".py"],
        python_version = "PY3",
        # TODO(b/145815906) Get this running in OSS CI.
        tags = ["noga"],
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    for name in [
        "conv_test",
        "linspace_test",
        "math_test",
        # TODO(GH-1620): Re-enable this after fixing the failure on
        # GitHub Actions.
        "keras_lstm_static_test",
    ]
]

# It is used to produce weights for keras vision models with input image size 32x32.
# These models are not optimized for accuracy or latency (they are for debugging only).
# They have the same neural net topology
# with keras vision models trained on imagenet data sets
py_binary(
    name = "train_vision_models_on_cifar",
    srcs = [
        "train_vision_models_on_cifar.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)
