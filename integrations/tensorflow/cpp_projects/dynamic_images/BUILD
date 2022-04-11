# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

common_copts = [
    "-fno-exceptions",
] + select({
    "@org_tensorflow//tensorflow:android": [
        "-mfpu=neon",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    "//conditions:default": [
        "-Wno-sign-compare",
    ],
})

common_linkopts = select({
    "@org_tensorflow//tensorflow:android": [
        "-latomic",
        "-ldl",
        "-llog",
        "-lm",
        "-pie",
        "-Wl,--gc-sections",  # Eliminate unused code and data.
        "-Wl,--as-needed",  # Don't link unused libs.
        "-s",
    ],
    "//conditions:default": [],
})

proto_library(
    name = "anchor_layers_proto",
    srcs = [
        "anchor_layers.proto",
    ],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "anchor_layers_cc_proto",
    deps = [":anchor_layers_proto"],
)

cc_library(
    name = "detection_post_process",
    srcs = ["detection_post_process.cc"],
    hdrs = [
        "detection_post_process.h",
    ],
    deps = [
        ":anchor_layers_cc_proto",
    ],
)

filegroup(
    name = "test_data",
    srcs = [
        # Source https://tfhub.dev/iree/lite-model/ssd_mobilenet_v2_100/fp32/default/1
        "test_data/ssd_mobilenet_v2_coco_dynamic.tflite",
        "test_data/anchors_dynamic.pb",
    ] + glob([
        "test_data/images/*.jpg",
    ]),
)

cc_test(
    name = "dynamic_images_test",
    srcs = ["dynamic_images_test.cc"],
    copts = common_copts,
    data = [
        ":test_data",
    ],
    linkopts = common_linkopts,
    linkstatic = True,
    deps = [
        ":detection_post_process",
        "@bazel_tools//tools/cpp/runfiles",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/stages:image_preprocessing_stage",
        "@org_tensorflow_lite_support//tensorflow_lite_support/examples/task/vision/desktop/utils:image_utils",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:lib",
        ],
    }),
)
