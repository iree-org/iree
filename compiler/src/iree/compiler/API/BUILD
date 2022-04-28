# NOTE: This must be BUILD.bazel (not BUILD) because otherwise, you can't
# build python packages in this directory (which creates "build" directories)
# on case-inensitive file systems.
load("//iree:build_defs.oss.bzl", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

# Allow all python sources to be built externally.
exports_files(glob([
    "python/*.cpp",
    "python/*.c",
    "python/**/*.py",
    "unittests/**/*.py",
]))

################################################################################
# Package-by-package filegroup exports.
# Some tools can make use of such non-nested groups.
################################################################################

filegroup(
    name = "python_compiler_tools",
    srcs = glob(["python/iree/compiler/tools/*.py"]),
)

filegroup(
    name = "python_compiler_tools_deprecated_redirects",
    srcs = [
        "python/iree/compiler/__init__.py",
        "python/iree/compiler/tf.py",
        "python/iree/compiler/tflite.py",
        "python/iree/compiler/xla.py",
    ],
)

filegroup(
    name = "python_compiler_api",
    srcs = glob(["python/iree/compiler/transforms/*.py"]),
)

filegroup(
    name = "unittests_tools_testdata",
    srcs = glob([
        "python/test/tools/testdata/*",
    ]),
)

################################################################################
# CAPI
################################################################################

iree_compiler_cc_library(
    name = "CAPI",
    srcs = [
        "Compiler.cpp",
        "Lld.cpp",
        "Tools.cpp",
    ],
    hdrs = [
        "Compiler.h",
        "Tools.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/ConstEval",
        "//compiler/src/iree/compiler/Dialect/VM/IR",
        "//compiler/src/iree/compiler/Dialect/VM/Target/Bytecode",
        "//compiler/src/iree/compiler/InputConversion/MHLO",
        "//compiler/src/iree/compiler/InputConversion/TOSA",
        "//compiler/src/iree/compiler/Pipelines",
        "//compiler/src/iree/compiler/Utils",
        "//iree/tools:init_llvmir_translations",
        "//iree/tools:init_passes_and_dialects",
        "//iree/tools:init_targets",
        "//iree/tools:iree_translate_lib",
        "@llvm-project//lld:COFF",
        "@llvm-project//lld:Common",
        "@llvm-project//lld:ELF",
        "@llvm-project//lld:MachO",
        "@llvm-project//lld:Wasm",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
    ],
)
