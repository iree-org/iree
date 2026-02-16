//===-- Gemmini.td - Gemmini dialect operation definitions --- tablegen --===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the Gemmini dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GEMMINI_OPS
#define GEMMINI_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Gemmini dialect definition
//===----------------------------------------------------------------------===//

def Gemmini_Dialect : Dialect {
  let name = "gemmini";
  let cppNamespace = "::buddy::gemmini";
  let summary = "Basic dialect to target RISC-V Gemmini extension";
  let description = [{
    Gemmini is an accelerator based on systolic array architecture.
    For more details about Gemmini,
    please see the [Gemmini repo](https://github.com/ucb-bar/gemmini).
  }];
}

//===----------------------------------------------------------------------===//
// Gemmini operation definitions
//===----------------------------------------------------------------------===//
class Gemmini_Op<string mnemonic, list<Trait> traits = []> :
  Op<Gemmini_Dialect, mnemonic, traits> {}

def  FlushOp : Gemmini_Op<"flush"> {
  let summary = "Flush op";
  let description = [{
    Flush operation flushes the TLB.
    If skip = 1, the current TLB request is skipped.
    Otherwise, the current TLB request is repeated.
  }];
  let arguments = (ins I64:$skip);
  let assemblyFormat = "$skip attr-dict `:` type($skip)";
}

def ConfigStOp : Gemmini_Op<"config_st"> {
  let summary = "Config store operation";
  let description = [{
    TODO: consider the attribute type according to Gemmini spec.
  }];
  let arguments = (ins I64:$stride,
                       DefaultValuedAttr<I64Attr, "0">:$activation,
                       DefaultValuedAttr<F32Attr, "1.0">:$scale);  
  let assemblyFormat = "$stride attr-dict `:` type($stride)";
}

def ConfigLdOp : Gemmini_Op<"config_ld"> {
  let summary = "Config load operation";
  let description = [{
    TODO: consider the attribute type according to Gemmini spec.
  }];
  let arguments = (ins I64:$stride,
                       DefaultValuedAttr<F32Attr, "1.0">:$scale,
                       DefaultValuedAttr<BoolAttr, "false">:$shrunk,
                       DefaultValuedAttr<I64Attr, "0">:$id,
                       DefaultValuedAttr<I64Attr, "-1">:$block_mvin_stride,
                       DefaultValuedAttr<I64Attr, "1">:$pixel_repeats);
  let assemblyFormat = "$stride attr-dict `:` type($stride)";
}

def ConfigExOp : Gemmini_Op<"config_ex"> {
  let summary = "ConfigExOp configures the execute pipeline";
  let description = [{
    ConfigExOp configures the execute pipeline.
    - dataflow: output-stationary (0) or weight-stationary (1) dataflow
    - sysAct: activation function relu (1) or no activation function (0)
    - sysShift: the number of bits by which the accumulated result of a matmul 
                is right-shifted when leaving the systolic array.
    - sysAccScale: the scalar value by which we scale the accType output of the 
                   accumulator down to inputType values when reading from the
                   accumulator.
                   (In the default config, rs1[63:32] is of type float32)
    - cStride: TODO
    - aStride: the stride (in scratchpad addresses) by which the rows of A are 
               fed into the systolic array.  "A" in this context refers to the 
               left-hand matrix A in the matmul represented by A * B = C. 
               If this stride is 1, then we feed consecutive rows in the 
               scratchpad, starting from the starting address of A, into the 
               systolic array as the A matrix. If the stride is 2, then we feed 
               every other row into the systolic array instead.
    - aTranspose: transpose A
    - bTranspose: transpose B
    - setOnlyStrides: TODO
  }];
  let arguments = (ins DefaultValuedAttr<I64Attr, "0">:$dataflow,
                       DefaultValuedAttr<I64Attr, "0">:$sysAct,
                       DefaultValuedAttr<I64Attr, "0">:$sysShift,
                       DefaultValuedAttr<F32Attr, "1.0">:$sysAccScale,
                       DefaultValuedAttr<I64Attr, "1">:$cStride,
                       DefaultValuedAttr<I64Attr, "1">:$aStride,
                       DefaultValuedAttr<BoolAttr, "false">:$aTranspose,
                       DefaultValuedAttr<BoolAttr, "false">:$bTranspose,
                       DefaultValuedAttr<BoolAttr, "false">:$setOnlyStrides);
  let assemblyFormat = "attr-dict";
}

def ConfigNormOp : Gemmini_Op<"config_norm"> {
  let summary = "ConfigNormOp configures normalize pipeline";
  let description = [{
    ConfigNormOp configures normalize pipeline
    -qConst: A constant value used for quantization during normalization.
    -qConstType: Defines the type of the qConst.
    -setStatsIdOnly: A flag to indicate if only the StatsId should be set.
    -actMsg: A message related to the normalization activity.
    -StatsId: An identifier associated with the statistics or metrics of the normalization process.
    -igeluQb: A parameter related to the IGELU function for quantization. Specifies the 'b' value.
    -igeluQc: Another parameter related to the IGELU function for quantization. Specifies the 'c' value.
  }];
  let arguments = (ins DefaultValuedAttr<I64Attr, "0">:$qConst,
                       DefaultValuedAttr<I64Attr, "0">:$qConstType,
                       DefaultValuedAttr<I64Attr, "0">:$setStatsIdOnly,
                       DefaultValuedAttr<I64Attr, "0">:$actMsb,
                       DefaultValuedAttr<I64Attr, "0">:$StatsId,
                       DefaultValuedAttr<I64Attr, "0">:$igeluQb,
                       DefaultValuedAttr<I64Attr, "0">:$igeluQc);
  let assemblyFormat = "attr-dict";
}

def MvinOp : Gemmini_Op<"mvin"> {
  let summary = "Load operation";
  let description = [{
    Move data from main memory to scratchpad
    - MemRef to load in.
      (including DRAM address, number of columns, number of rows)
    - Local scratchpad or accumulator address.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input, I64:$addr);
  let assemblyFormat = "$input $addr attr-dict `:` type($input) type($addr)";
}

def Mvin2Op : Gemmini_Op<"mvin2"> {
  let summary = "Load operation";
  let description = [{
      Similar to Mvin
          Move data from main memory to scratchpad
      - MemRef to load in.
        (including DRAM address, number of columns, number of rows)
      - Local scratchpad or accumulator address.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input, I64:$addr);
  let assemblyFormat = "$input $addr attr-dict `:` type($input) type($addr)";
}

def Mvin3Op : Gemmini_Op<"mvin3"> {
  let summary = "Load operation";
  let description = [{
      Similar to Mvin and Mvin2
          Move data from main memory to scratchpad
      - MemRef to load in.
        (including DRAM address, number of columns, number of rows)
      - Local scratchpad or accumulator address.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input, I64:$addr);
  let assemblyFormat = "$input $addr attr-dict `:` type($input) type($addr)";
}

def MvoutOp : Gemmini_Op<"mvout"> {
  let summary = "Store operation";
  let description = [{
    Move data from scratchpad to L2/DRAM
    - MemRef to store from scratchpad.
      (including DRAM address, number of columns, number of rows)
    - Local scratchpad address.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$output, I64:$addr);
  let assemblyFormat = "$output $addr attr-dict `:` type($output) type($addr)";
}

def PrintOp : Gemmini_Op<"print"> {
  let summary = "Print memref value.";
  let arguments = (ins AnyTypeOf<[I8MemRef, I32MemRef, F32MemRef, F64MemRef]>:$input); 
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PreloadZerosOp : Gemmini_Op<"preload_zeros"> {
  let summary = "Preload zeros in scratchpad";
  let description = [{
    - Local scratchpad address.
    - Number of rows
    - Number of columns.
  }];
  let arguments = (ins I64:$addr, I64:$cRows, I64:$cCols);
  let assemblyFormat = [{
    $addr $cRows $cCols attr-dict `:` type($addr) type($cRows) type($cCols)
  }];
}

def PreloadOp : Gemmini_Op<"preload"> {
  let summary = "Preload matrix in scratchpad";
  let description = [{
    - Local scratchpad address of D matrix (when output-stationary),
      or B matrix (when weight-stationary)
    - Local scratchpad address of C matrix. If this is set to all high bits,
      then C will not be written to the scratchpad or accumulator.
    - Number of rows of D/B matrix.
    - Number of columns of D/B matrix.
    - Number of rows of C matrix.
    - Number of columns of C matrix.
  }];
  let arguments = (ins I64:$bdAddr, I64:$cAddr, I64:$bdRows,
                       I64:$bdCols, I64:$cRows, I64:$cCols);
  let assemblyFormat = [{
    $bdAddr $cAddr $bdRows $bdCols $cRows $cCols  attr-dict `:` type($bdAddr) 
    type($cAddr) type($bdRows) type($bdCols) type($cRows) type($cCols)
  }];
}

def ComputePreloadedOp : Gemmini_Op<"compute_preloaded"> {
  let summary = "compute operation";
  let description = [{
    Explicitly Preloaded
    - Local scratchpad address(systolic array single-axis addressed) of A matrix
    - local scratchpad address(systolic array single-axis addressed) of B matrix
      (when output-stationary), or D matrix (when weight-stationary)
    - Number of rows of A matrix
    - Number of columns of A matrix
    - Number of rows of B/D matrix
    - Number of columns of B/D matrix
  }];
  let arguments = (ins I64:$aAddr, I64:$bdAddr, I64:$aRows,
                       I64:$aCols, I64:$bdRows, I64:$bdCols);
  let assemblyFormat = [{
    $aAddr $bdAddr $aRows $aCols $bdRows $bdCols attr-dict `:` type($aAddr)
    type($bdAddr) type($aRows) type($aCols) type($bdRows) type($bdCols)
  }];
}

def ComputeAccumulatedOp : Gemmini_Op<"compute_accumulated"> {
  let summary = "compute accumulated opertion";
  let description = [{
    - Local scratchpad address(systolic array single-axis addressed) of A matrix
    - local scratchpad address(systolic array single-axis addressed) of B matrix
      (when output-stationary), or D matrix (when weight-stationary)
    - Number of rows of A matrix
    - Number of columns of A matrix
    - Number of rows of B/D matrix
    - Number of columns of B/D matrix
  }];
  let arguments = (ins I64:$aAddr, I64:$bdAddr, I64:$aRows,
                       I64:$aCols, I64:$bdRows, I64:$bdCols);
  let assemblyFormat = [{
    $aAddr $bdAddr $aRows $aCols $bdRows $bdCols attr-dict `:` type($aAddr)
    type($bdAddr) type($aRows) type($aCols) type($bdRows) type($bdCols)
  }];
}

def TileMatMulOp : Gemmini_Op<"tile_matmul"> {
  let summary = "Perform matrix multiplication.";
  let description = [{
    TODO: consider the attribute type according to Gemmini spec.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aArray,
                       MemRefRankOf<[AnyType], [2]>:$bArray,
                       MemRefRankOf<[AnyType], [2]>:$cArray,
                       MemRefRankOf<[AnyType], [2]>:$dArray,
                       DefaultValuedAttr<F32Attr, "1.0">:$aScaleFactor,
                       DefaultValuedAttr<F32Attr, "1.0">:$bScaleFactor,
                       DefaultValuedAttr<F32Attr, "1.0">:$dScaleFactor,
                       DefaultValuedAttr<I64Attr, "0">:$act,
                       DefaultValuedAttr<F32Attr, "1.0">:$accScale,
                       DefaultValuedAttr<F32Attr, "0.0">:$bertScale,
                       DefaultValuedAttr<BoolAttr, "false">:$repeatingBias,
                       DefaultValuedAttr<BoolAttr, "false">:$aTranspose,
                       DefaultValuedAttr<BoolAttr, "false">:$bTranspose,
                       DefaultValuedAttr<BoolAttr, "false">:$fullC,
                       DefaultValuedAttr<BoolAttr, "false">:$lowD,
                       DefaultValuedAttr<I64Attr, "0">:$weightA,
                       DefaultValuedAttr<I64Attr, "1">:$dataflow);
  let assemblyFormat = [{
    $aArray $bArray $cArray $dArray attr-dict `:`
    type($aArray) type($bArray) type($cArray) type($dArray)
  }];
}

def TileConvOp : Gemmini_Op<"tile_conv"> {
  let summary = "Perform convolution.";
  let description = [{
    TODO: consider the attribute type according to Gemmini spec.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [4]>:$input,
                       MemRefRankOf<[AnyType], [2]>:$weights,
                       MemRefRankOf<[AnyType], [1]>:$bias,
                       MemRefRankOf<[AnyType], [2]>:$output,
                       I64:$outRowDim, I64:$outColDim, I64:$kernelDim,
                       DefaultValuedAttr<F32Attr, "1.0">:$scale,
                       DefaultValuedAttr<I64Attr, "1">:$stride,
                       DefaultValuedAttr<I64Attr, "1">:$inputDilation, 
                       DefaultValuedAttr<I64Attr, "1">:$kernelDilation,
                       DefaultValuedAttr<I64Attr, "0">:$padding, 
                       DefaultValuedAttr<BoolAttr, "false">:$wrot180,
                       DefaultValuedAttr<BoolAttr, "false">:$transOutput1203, 
                       DefaultValuedAttr<BoolAttr, "false">:$transInput3120,
                       DefaultValuedAttr<BoolAttr, "false">:$transWeight1203, 
                       DefaultValuedAttr<BoolAttr, "false">:$transWeight0132,
                       DefaultValuedAttr<I64Attr, "0">:$act,
                       DefaultValuedAttr<I64Attr, "0">:$poolSize,
                       DefaultValuedAttr<I64Attr, "0">:$poolStride, 
                       DefaultValuedAttr<I64Attr, "0">:$poolPadding);
  let assemblyFormat = [{
    $input $weights $bias $output $outRowDim $outColDim $kernelDim attr-dict `:` type($input)
    type($weights) type($bias) type($output) type($outRowDim) type($outColDim) type($kernelDim)
  }];
}

//===----------------------------------------------------------------------===//
// Gemmini intrinsic operation definitions
//===----------------------------------------------------------------------===//

class Gemmini_IntrOpBase<string mnemonic, list<Trait> traits = []> : 
  LLVM_IntrOpBase</*Dialect dialect=*/Gemmini_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[], 
                  /*list<int> overloadedOperands=*/[], 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/0>;

def Gemmini_Mvin_IntrOp : Gemmini_IntrOpBase<"mvin">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_Mvin2_IntrOp : Gemmini_IntrOpBase<"mvin2">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_Mvin3_IntrOp : Gemmini_IntrOpBase<"mvin3">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_Mvout_IntrOp : Gemmini_IntrOpBase<"mvout">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_Flush_IntrOp : Gemmini_IntrOpBase<"flush">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ConifgLd_IntrOp : Gemmini_IntrOpBase<"config_ld">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ConfigSt_IntrOp : Gemmini_IntrOpBase<"config_st">, 
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ConfigEX_IntrOp : Gemmini_IntrOpBase<"config_ex">, 
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ConfigNorm_IntrOp : Gemmini_IntrOpBase<"config_norm">, 
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_Preload_IntrOp : Gemmini_IntrOpBase<"preload">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ComputePreloaded_IntrOp: Gemmini_IntrOpBase<"compute_preloaded">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ComputeAccumulated_IntrOp : Gemmini_IntrOpBase<"compute_accumulated">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWsConfigBounds_IntrOp : Gemmini_IntrOpBase<"loop_ws_config_bounds">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWsConfigAddrsAB_IntrOp : Gemmini_IntrOpBase<"loop_ws_config_addrs_ab">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWsConfigAddrsDC_IntrOp : Gemmini_IntrOpBase<"loop_ws_config_addrs_dc">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWsConfigStridesAB_IntrOp : Gemmini_IntrOpBase<"loop_ws_config_strides_ab">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWsConfigStridesDC_IntrOp : Gemmini_IntrOpBase<"loop_ws_config_strides_dc">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWs_IntrOp : Gemmini_IntrOpBase<"loop_ws">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopConvWs_IntrOp : Gemmini_IntrOpBase<"loop_conv_ws">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopConvWsConfig1_IntrOp : Gemmini_IntrOpBase<"loop_conv_ws_config1">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopConvWsConfig2_IntrOp : Gemmini_IntrOpBase<"loop_conv_ws_config2">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopConvWsConfig3_IntrOp : Gemmini_IntrOpBase<"loop_conv_ws_config3">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopConvWsConfig4_IntrOp : Gemmini_IntrOpBase<"loop_conv_ws_config4">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopConvWsConfig5_IntrOp : Gemmini_IntrOpBase<"loop_conv_ws_config5">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopConvWsConfig6_IntrOp : Gemmini_IntrOpBase<"loop_conv_ws_config6">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif 
