# Transforms specific to the IREE sequencer.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Sequencer",
    srcs = [
        "AssignExecutableOrdinals.cpp",
        "AssignExecutableWorkloadAttrs.cpp",
        "FoldCompatibleDispatchRegions.cpp",
        "IdentifyDispatchRegions.cpp",
        "IdentifyReductionRegions.cpp",
        "LowerSequencerDialect.cpp",
        "LowerStdToSequencerDialect.cpp",
        "LowerToSequencerDialect.cpp",
        "LowerXLAToSequencerDialect.cpp",
        "OutlineDispatchRegions.cpp",
        "OutlineReductionRegions.cpp",
        "RematerializeDispatchConstants.cpp",
    ],
    hdrs = [
        "Passes.h",
        "Rewrites.h",
    ],
    deps = [
        "//iree/compiler/IR",
        "//iree/compiler/IR/Sequencer",
        "//iree/compiler/Transforms",
        "//iree/compiler/Utils",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:Transforms",
        "@tensorflow///compiler/mlir/xla:hlo",
        "@tensorflow///compiler/mlir/xla:xla_legalize_to_standard",
        "@tensorflow///compiler/mlir/xla:xla_lower_general_dot",
    ],
)
