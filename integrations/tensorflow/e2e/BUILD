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
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

[
    py_test(
        name = name,
        srcs = [name + ".py"],
        python_version = "PY3",
        # TODO(b/145815906) Get this running in OSS CI.
        tags = ["nokokoro"],
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//bindings/python/pyiree/tf/support",
        ],
    )
    for name in [
        "control_flow_test",
        "dynamic_mlp_test",
        "exported_names_test",
        "keras_lstm_test",
        "simple_stateful_test",
        "mandelbrot_test",
        "simple_arithmetic_test",
        "batch_norm_test",
    ]
]

py_test(
    name = "vulkan_conv_test",
    srcs = ["vulkan_conv_test.py"],
    python_version = "PY3",
    # TODO(b/145815906) Get this running in OSS CI.
    tags = [
        "noga",
        "nokokoro",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//bindings/python/pyiree/tf/support",
    ],
)
