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
    "PYBIND_COPTS",
    "PYBIND_EXTENSION_COPTS",
    "PYBIND_FEATURES",
    "iree_py_extension",
    "iree_py_library",
    "iree_py_test",
    "pybind_cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_py_library(
    name = "compiler",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    tags = [
        "nokokoro",
    ],
    deps = [
        ":binding",
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "//integrations/tensorflow/bindings/python:pathsetup",  # build_cleaner: keep
    ],
)

iree_py_extension(
    name = "binding",
    srcs = [
        "initialize_module.cc",
    ],
    copts = PYBIND_COPTS + PYBIND_EXTENSION_COPTS,
    features = PYBIND_FEATURES,
    linkstatic = 1,
    tags = [
        "nokokoro",
    ],
    win_def_file = "export.def",
    deps = [
        ":compiler_library",
        "//bindings/python/pyiree/common",
        "//bindings/python/pyiree/compiler:compiler_library",
    ],
)

pybind_cc_library(
    name = "compiler_library",
    srcs = [
        "register_xla.cc",
    ],
    hdrs = [
        "register_xla.h",
    ],
    tags = [
        "nokokoro",
    ],
    deps = [
        "//bindings/python/pyiree/common",
        "//bindings/python/pyiree/compiler:compiler_library",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo_module_importer",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    ],
)

iree_py_test(
    name = "xla_module_proto_test",
    srcs = ["xla_module_proto_test.py"],
    python_version = "PY3",
    tags = [
        "nokokoro",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "@absl_py//absl/testing:absltest",
        "//integrations/tensorflow/bindings/python/pyiree/xla/compiler",
        "@org_tensorflow//tensorflow/compiler/xla/python:xla_client",
    ],
)
