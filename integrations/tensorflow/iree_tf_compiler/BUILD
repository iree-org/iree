# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "importer-binaries",
    srcs = [
        ":iree-import-tf",
        ":iree-import-tflite",
        ":iree-import-xla",
    ],
)

filegroup(
    name = "opt-binaries",
    srcs = [
        ":iree-opt-tflite",
        ":iree-tf-opt",
    ],
)

cc_binary(
    name = "iree-tf-opt",
    srcs = ["iree-tf-opt-main.cpp"],
    deps = [
        "//iree_tf_compiler/MHLO",
        "//iree_tf_compiler/TF",
        "//iree_tf_compiler/dialect/tf_strings/ir:dialect",
        "//iree_tf_compiler/dialect/tf_tensorlist/ir:tf_tensorlist_dialect",
        "@iree//iree/compiler/Dialect/Flow/IR",
        "@iree//iree/compiler/Dialect/HAL/IR",
        "@iree//iree/compiler/Dialect/IREE/IR",
        "@iree//iree/compiler/Dialect/Modules/Strings/IR:Dialect",
        "@iree//iree/compiler/Dialect/Modules/TensorList/IR",
        "@iree//iree/tools:init_xla_dialects",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_legalize_tf",
    ],
)

cc_binary(
    name = "iree-opt-tflite",
    srcs = ["iree-opt-tflite-main.cpp"],
    deps = [
        "//iree_tf_compiler/TFL",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
)

cc_binary(
    name = "iree-import-tf",
    srcs = ["iree-import-tf-main.cpp"],
    deps = [
        "//iree_tf_compiler/TF",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "@org_tensorflow//tensorflow/core/platform:errors",
    ],
)

cc_binary(
    name = "iree-import-tflite",
    srcs = ["iree-import-tflite-main.cpp"],
    deps = [
        "//iree_tf_compiler/TFL",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
    ],
)

cc_binary(
    name = "iree-import-xla",
    srcs = ["iree-import-xla-main.cpp"],
    deps = [
        "//iree_tf_compiler/MHLO",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo:hlo_dialect_registration",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo_to_mlir_hlo",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_parser",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
