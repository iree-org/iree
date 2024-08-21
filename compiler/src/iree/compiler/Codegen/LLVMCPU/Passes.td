// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_LLVMCPU_PASSES
#define IREE_CODEGEN_LLVMCPU_PASSES

include "mlir/Pass/PassBase.td"

//------------------------------------------------------------------------------
// LLVMCPU Passes (keep alphabetical)
//------------------------------------------------------------------------------

def ConvertToLLVMPass :
    Pass<"iree-convert-to-llvm", "ModuleOp"> {
  let summary =
      "Perform final conversion from Linalg/HAL/Shape/Vector/Standard to LLVMIR dialect";
  let options = [
    Option<"reassociateFpReductions", "reassociateFpReductions", "bool",
            /*default=*/"false",
           "Specifies if FP add and mult reductions can be reordered">,
    Option<"targetTriple", "target-triple", "std::string", /*default=*/"",
           "Code generation target triple.">,
    Option<"targetDataLayout", "target-data-layout", "std::string",
           /*default=*/"",
           "Code generation target data layout.">,
  ];
}

def ExpandF16OpToF32Pass :
    Pass<"iree-llvmcpu-expand-f16-op-to-f32", ""> {
  let summary =
      "Preform f16 opertaions by expanding them to f32.";
  let description = [{
    Pass to handel F16 bit operations, but converting f16 operands to F32.
    Currently this pass is handeling fmaxf conversion from f16 to f32,
    and then returing a f16 output back after preforming the operation.
    Can handle more operations if required in future.
  }];
}

def LLVMCPUAssignConstantOrdinalsPass :
    Pass<"iree-llvmcpu-assign-constant-ordinals", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Assigns executable constant ordinals across all LLVMCPU variants.";
}

def LLVMCPUAssignImportOrdinalsPass :
    Pass<"iree-llvmcpu-assign-import-ordinals", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Assigns executable import ordinals across all LLVMCPU variants.";
}

def LLVMCPUCheckIRBeforeLLVMConversionPass :
    InterfacePass<"iree-llvmcpu-check-ir-before-llvm-conversion", "mlir::FunctionOpInterface"> {
  let summary = "Checks CPU backend specific IR constraints (like no allocas)";
  let options = [
    Option<"failOnOutOfBounds", "fail-on-out-of-bounds", "bool", "true",
           "Fails if the upper bound of dynamic stack allocation cannot be"
           "resolved or is more than the limit.">
  ];
}

def LLVMCPUEmitVectorizationRemarksPass :
    InterfacePass<"iree-llvmcpu-emit-vectorization-remarks", "mlir::FunctionOpInterface"> {
  let summary = "Emit vectorization remarks on Linalg ops";
}

def LLVMCPULinkExecutablesPass :
    Pass<"iree-llvmcpu-link-executables", "mlir::ModuleOp"> {
  let summary = "Links LLVMCPU HAL executables within the top-level program module.";
}

def LLVMCPULowerExecutableTargetPass :
    InterfacePass<"iree-llvmcpu-lower-executable-target", "mlir::FunctionOpInterface"> {
  let summary =
      "Lower executable target using an IREE::HAL::DispatchLoweringPassPipeline";
  let description = [{
    Pass to lower the module an hal.executable.variant operation to external
    dialect. Currently this pass lowers to LLVM dialect, but could be
    generalized to lower to any "final" dialect like SPIR-V/NVVM, etc.
  }];
}

def LLVMCPUMmt4dVectorLoweringPass
    : InterfacePass<"iree-llvmcpu-mmt4d-vector-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Apply vector lowering logic to vector ops";
  let options = [
    Option<"enableVectorContractCustomKernels", "vector-contract-custom-kernels", "bool",
      /*default=*/"true",
      "Flag to enable or disable vector contract custom kernels.">,
  ];
}

def LLVMCPUPeelPass :
    InterfacePass<"iree-llvmcpu-peel", "mlir::FunctionOpInterface"> {
  let summary = "Pass to perform peeling on non-distributed loops.";
}

def LLVMCPUSelectLoweringStrategyPass :
    Pass<"iree-llvmcpu-select-lowering-strategy", "ModuleOp"> {
  let summary =
      "Select a IREE::HAL::DispatchLoweringPassPipeline for lowering the variant";
  let description = [{
    Pass to select a lowering strategy for a hal.executable.variant operation.
    The variant is annotated with the selected strategies, which are
    subsequently ingested by LLVMCPULowerExecutableTargetPass.
  }];
}

def LLVMCPUSplitReductionPass : InterfacePass<"iree-llvmcpu-split-reduction", "mlir::FunctionOpInterface"> {
  let summary = "Pass to splitReduce linalg operations.";
  let options = [
    Option<"enableFpReductionReordering", "enable-fp-reduction-reordering",
      "bool", /*default=*/"false",
      "Flag to enable reduction reordering on floating points.">,
  ];
}

def LLVMCPUSynchronizeSymbolVisibilityPass :
    Pass<"iree-llvmcpu-synchronize-symbol-visibility", "ModuleOp"> {
  let summary = "Synchronizes LLVM linkage with MLIR symbol visibility";
}

def LLVMCPUTilePass :
    InterfacePass<"iree-llvmcpu-tile", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile TilingInterface operations.";
  let options = [
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">
  ];
}

def LLVMCPUTileAndFusePass :
    InterfacePass<"iree-llvmcpu-tile-and-fuse", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile and fuse TilingInterface operations.";
  let options = [
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">
  ];
}

def LLVMCPUTileRootAndFuseProducerConsumerPass
    : InterfacePass<"iree-llvmcpu-tile-root-and-fuse-producer-consumer",
                    "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile root op and fuse with producer and consumer "
                "TilingInterface ops.";
  let options =
      [Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
              "Use default tiling level used to retrieve the configuration "
              "from lowering_config">,
       Option<"onlyFuseProducerInputOperands",
              "only-fuse-producer-input-operands", "bool",
              /*default=*/"false",
              "Specifies if we only want to fuse producer's input operands. "
              "This is helpful to tile&fuse in case of reduction dimensions.">];
}

def LLVMCPUVerifyVectorSizeLegalityPass :
    InterfacePass<"iree-llvmcpu-verify-vector-size-legality", "mlir::FunctionOpInterface"> {
  let summary =
      "Signals errors when there are large vectors in the IR. I.e., one of"
      "the vector sizes is greater than"
      "maxAllowedNumberOfNativeVectors * native_vector_size. For scalable"
      "vectors, it assumes that the vscale value is always 1. It may be an"
      "underestimate if the runtime larger than 1, but it should still catch"
      "unreasonable vector sizes.";
  let options = [
    Option<"maxAllowedNumberOfNativeVectors", "max-allowed-number-of-native-vectors", "int64_t", /*default=*/"512",
           "The ratio used in the computation of max vector size.">
  ];
}

// Note: This pass is currently only required when targeting Arm SME (which is
// the only target that currently has some concept of 2D scalability).
def LLVMCPU2DScalableTo1DScalablePass :
    InterfacePass<"iree-llvmcpu-2d-scalable-to-1d-scalable", "mlir::FunctionOpInterface"> {
  let summary = "Pass to replace unsupported scalable dimensions with loops.";
  let options = [
    Option<"assumeArmSME", "assume-arm-sme", "bool", /*default=*/"false",
      "Assume the current target is ArmSME (used for testing)">
  ];
}

def LLVMCPUUnfuseFMAOpsPass :
    InterfacePass<"iree-llvmcpu-unfuse-fma-pass", "mlir::FunctionOpInterface"> {
  let summary = "Convert llvm.fma into unfused mulf and addf ops";
}

def LLVMCPUDropVectorUnitDimsPass :
    InterfacePass<"iree-llvmcpu-drop-vector-unit-dims", "mlir::FunctionOpInterface"> {
  let summary = "Pass to drop vector unit dims.";
}

def LLVMCPUVirtualVectorLoweringPass :
    InterfacePass<"iree-llvmcpu-virtual-vector-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower high level vector operations like contract or multidim reduce ops to lower level vector ops.";
  let options = [
    Option<"splitVectorTransfersTo", "split-transfers", "std::string",
      /*default=*/"",
      [{Split vector transfers between slow (masked) and fast "
        "(unmasked) variants. Possible options are:\n"
          "\tnone [default]: keep unsplit vector.transfer and pay the price\n"
          "\tlinalg-copy: use linalg.fill + linalg.generic for the slow path\n"
          "\tvector-transfers: use extra small unmasked vector.transfers for"
          " the slow path\n}]>,
    Option<"enableArmI8mm", "enable-arm-i8mm", "bool",
      /*default=*/ "false",
      "Enables arm i8mm lowering patterns">,
  ];
}

def LLVMCPUVectorTransferLoweringPass :
    InterfacePass<"iree-llvmcpu-vector-transfer-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower transfer ops to simpler ops like `vector.load`, `vector.store`, `vector.broadcast`, and a set of scf ops.";
  let options = [
    Option<"enableScalableLowerings", "enable-scalable-lowerings", "bool",
      /*default=*/"false",
      "Enables scalable vector specific transfer lowerings">,
  ];
}

def LLVMCPUVectorTransposeLoweringPass :
    InterfacePass<"iree-llvmcpu-vector-transpose-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower vector.transpose ops.";
  let options = [
    Option<"lowerVectorTransposeToAVX2", "lower-vector-transpose-to-avx2", "bool",
      /*default=*/"false",
      "Add specific transpose to avx2 lowering patterns.">,
  ];
}

def LLVMCPUVectorShapeCastLoweringPass :
    InterfacePass<"iree-llvmcpu-vector-shape-cast-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower vector.shape_cast ops.";
}

def VectorContractCustomKernelsPass :
    InterfacePass<"iree-llvmcpu-vector-contract-custom-kernels", "mlir::FunctionOpInterface"> {
  let summary = "Enable custom kernels (inline assembly or intrinsics) for some vector.contract ops";
}

def VerifyLinalgTransformLegalityPass :
    InterfacePass<"iree-llvmcpu-verify-linalg-transform-legality", "mlir::FunctionOpInterface"> {
  let summary = "Verify that only supported IR constructs are passed to the compiler.";
}

#endif // IREE_CODEGEN_LLVMCPU_PASSES
