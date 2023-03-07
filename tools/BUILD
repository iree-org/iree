# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tools IREE provides for compiling, executing, and benchmarking programs, as
# well as other utilities.
#
# Only binary targets and their associated main files should go in this
# directory. Library targets and header files should be placed in the
# appropriate subtree, e.g. `compiler/src/iree/compiler/Tools/`.
#
# Programs with a dependency on the compiler are not designed to run on device
# platforms (e.g. Android), so they are tagged "hostonly".

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["lit.cfg.py"])

cc_binary(
    name = "iree-benchmark-module",
    srcs = ["iree-benchmark-module-main.cc"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/modules/hal:types",
        "//runtime/src/iree/tooling:context_util",
        "//runtime/src/iree/tooling:device_util",
        "//runtime/src/iree/tooling:vm_util",
        "//runtime/src/iree/vm",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "iree-benchmark-trace",
    srcs = ["iree-benchmark-trace-main.c"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:atomic_slist",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/base/internal:path",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/testing:benchmark",
        "//runtime/src/iree/tooling:device_util",
        "//runtime/src/iree/tooling:trace_replay",
        "//runtime/src/iree/tooling:yaml_util",
        "//runtime/src/iree/vm",
        "@com_github_yaml_libyaml//:yaml",
    ],
)

cc_binary(
    name = "iree-check-module",
    testonly = True,
    srcs = ["iree-check-module-main.cc"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:file_io",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/modules/check",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/tooling:context_util",
        "//runtime/src/iree/tooling:device_util",
        "//runtime/src/iree/tooling:vm_util",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm/bytecode:module",
    ],
)

cc_binary(
    name = "iree-dump-instruments",
    srcs = ["iree-dump-instruments-main.c"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:file_io",
        "//runtime/src/iree/base/internal/flatcc:parsing",
        "//runtime/src/iree/schemas/instruments",
        "//runtime/src/iree/schemas/instruments:dispatch_def_c_fbs",
    ],
)

cc_binary(
    name = "iree-dump-module",
    srcs = ["iree-dump-module-main.c"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:file_io",
        "//runtime/src/iree/base/internal/flatcc:debugging",
        "//runtime/src/iree/schemas:bytecode_module_def_c_fbs",
        "//runtime/src/iree/vm/bytecode:module",
    ],
)

cc_binary(
    name = "iree-opt",
    srcs = ["iree-opt-main.cc"],
    tags = ["hostonly"],
    deps = [
        "//compiler/bindings/c:headers",
        "//compiler/src/iree/compiler/API2:Impl",
    ],
)

cc_binary(
    name = "iree-mlir-lsp-server",
    srcs = ["iree-mlir-lsp-server.cc"],
    deps = [
        "//compiler/bindings/c:headers",
        "//compiler/src/iree/compiler/API2:Impl",
    ],
)

cc_binary(
    name = "iree-run-mlir",
    srcs = ["iree-run-mlir-main.cc"],
    tags = ["hostonly"],
    deps = [
        "//compiler/src/iree/compiler/ConstEval",
        "//compiler/src/iree/compiler/Dialect/HAL/Target",
        "//compiler/src/iree/compiler/Dialect/VM/Target:init_targets",
        "//compiler/src/iree/compiler/Dialect/VM/Target/Bytecode",
        "//compiler/src/iree/compiler/Pipelines",
        "//compiler/src/iree/compiler/Tools:init_passes_and_dialects",
        "//compiler/src/iree/compiler/Tools:init_targets",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/modules/hal:types",
        "//runtime/src/iree/tooling:context_util",
        "//runtime/src/iree/tooling:device_util",
        "//runtime/src/iree/tooling:vm_util",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm/bytecode:module",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)

cc_binary(
    name = "iree-run-module",
    srcs = ["iree-run-module-main.cc"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/modules/hal:types",
        "//runtime/src/iree/tooling:comparison",
        "//runtime/src/iree/tooling:context_util",
        "//runtime/src/iree/tooling:device_util",
        "//runtime/src/iree/tooling:instrument_util",
        "//runtime/src/iree/tooling:vm_util",
        "//runtime/src/iree/vm",
    ],
)

cc_binary(
    name = "iree-run-trace",
    srcs = ["iree-run-trace-main.c"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/base/internal:path",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/tooling:device_util",
        "//runtime/src/iree/tooling:trace_replay",
        "//runtime/src/iree/tooling:yaml_util",
        "//runtime/src/iree/vm",
        "@com_github_yaml_libyaml//:yaml",
    ],
)

cc_binary(
    name = "iree-e2e-matmul-test",
    srcs = ["iree-e2e-matmul-test.c"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/base/internal:path",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/modules/hal",
        "//runtime/src/iree/tooling:device_util",
        "//runtime/src/iree/tooling:trace_replay",
        "//runtime/src/iree/tooling:yaml_util",
        "//runtime/src/iree/vm",
        "@com_github_yaml_libyaml//:yaml",
    ],
)

cc_binary(
    name = "iree-cpuinfo",
    srcs = ["iree-cpuinfo.c"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/schemas:cpu_data",
    ],
)

cc_binary(
    name = "iree-tblgen",
    srcs = [
        "//compiler/src/iree/compiler/Dialect/VM/Tools:GenSrcs",
        "@llvm-project//mlir:tools/mlir-tblgen/mlir-tblgen.cpp",
    ],
    tags = ["hostonly"],
    deps = [
        "//build_tools:default_linkopts",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//mlir:MlirTableGenMain",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TableGen",
    ],
)

cc_binary(
    name = "iree-compile",
    srcs = ["iree-compile-main.cc"],
    tags = ["hostonly"],
    deps = [
        "//compiler/bindings/c:headers",
        "//compiler/src/iree/compiler/API2:Impl",
    ],
)
