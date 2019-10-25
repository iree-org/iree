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

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

DEFAULT_COPTS = [
    "-fexceptions",
]

DEFAULT_FEATURES = [
    "-use_header_modules",  # Incompatible with exceptions builds.
]

config_setting(
    name = "enable",
    define_values = {
        "iree_tensorflow": "true",
    },
)

cc_library(
    name = "tensorflow",
    hdrs = [
        "register_tensorflow.h",
    ],
    copts = DEFAULT_COPTS,
    defines = select({
        ":enable": [
            "IREE_TENSORFLOW_ENABLED",
        ],
        "//conditions:default": [],
    }),
    features = DEFAULT_FEATURES,
    deps = select({
        ":enable": [
            ":tensorflow_impl",
        ],
        "//conditions:default": [
            ":tensorflow_disabled",
        ],
    }) + [
        "//bindings/python/pyiree:base",
    ],
)

# Runtime deps needed to compile the tensorflow compiler.
SAVED_MODEL_TF_RUNTIME_DEPS = [
    "@org_tensorflow//tensorflow/core:core_cpu",
    "@org_tensorflow//tensorflow/core:direct_session",
    "@org_tensorflow//tensorflow/core:ops",
]

# Kernels that the current version of the GraphDef/SavedModel importer
# needs. This has been manually curated and may need tweaking. As-is,
# this adds about 7MB to the binary (on win64) vs >300MB if all kernels
# are pulled in. As such, curating is a big enough savings to be worth the
# trouble in this case. Also, there is discussion about severing the
# importer from needing these things.
SAVED_MODEL_REQUIRED_KERNEL_DEPS = [
    "@org_tensorflow//tensorflow/core/kernels:constant_op",
    "@org_tensorflow//tensorflow/core/kernels:io",
    "@org_tensorflow//tensorflow/core/kernels:partitioned_function_ops",
    "@org_tensorflow//tensorflow/core/kernels:identity_op",
    "@org_tensorflow//tensorflow/core/kernels:identity_n_op",
]

cc_library(
    name = "tensorflow_impl",
    srcs = [
        "register_tensorflow.cc",
    ],
    hdrs = [
        "register_tensorflow.h",
    ],
    copts = DEFAULT_COPTS,
    features = DEFAULT_FEATURES,
    visibility = ["//visibility:private"],
    deps = [
        "@local_config_mlir//:IR",
        "@llvm//:support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//bindings/python/pyiree:base",
    ] + SAVED_MODEL_TF_RUNTIME_DEPS + SAVED_MODEL_REQUIRED_KERNEL_DEPS,
)

cc_library(
    name = "tensorflow_disabled",
    srcs = [
        "register_tensorflow_noop.cc",
    ],
    hdrs = [
        "register_tensorflow.h",
    ],
    copts = DEFAULT_COPTS,
    features = DEFAULT_FEATURES,
    deps = [
        "//bindings/python/pyiree:base",
    ],
)
