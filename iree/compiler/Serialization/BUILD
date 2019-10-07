# Serialization for the VM bytecode.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Serialization",
    srcs = [
        "BytecodeTables.cpp",
        "BytecodeWriter.cpp",
        "VMDeviceTableBuilder.cpp",
        "VMExecutableTableBuilder.cpp",
        "VMFunctionBuilder.cpp",
        "VMFunctionTableBuilder.cpp",
        "VMModuleBuilder.cpp",
        "VMSourceMapBuilder.cpp",
    ],
    hdrs = [
        "BytecodeTables.h",
        "BytecodeWriter.h",
        "VMDeviceTableBuilder.h",
        "VMExecutableTableBuilder.h",
        "VMFunctionBuilder.h",
        "VMFunctionTableBuilder.h",
        "VMModuleBuilder.h",
        "VMSourceMapBuilder.h",
    ],
    deps = [
        "//iree/compiler/IR",
        "//iree/compiler/Utils",
        "//iree/schemas",
        "//iree/schemas/bytecode:bytecode_v0",
        "//iree/schemas/bytecode:interpreter_bytecode_v0",
        "//iree/schemas/bytecode:sequencer_bytecode_v0",
        "//third_party/flatbuffers",
        "//third_party/llvm/llvm:support",
        "//third_party/llvm/llvm/projects/google_mlir:IR",
        "//third_party/llvm/llvm/projects/google_mlir:StandardOps",
        "//third_party/llvm/llvm/projects/google_mlir:Support",
    ],
)
