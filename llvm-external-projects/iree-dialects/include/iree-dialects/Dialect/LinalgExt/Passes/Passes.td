// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_PASSES
#define IREE_DIALECT_LINALGEXT_PASSES

include "mlir/Pass/PassBase.td"

def LinalgExtToLoops :
    Pass<"iree-linalg-ext-to-loops", "func::FuncOp"> {
  let summary = "Convert LinalgExt ops to loops and Linalg ops.";
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::createLinalgExtToLoopsPass()";
}

def TilingInterfaceTiling :
    Pass<"iree-linalg-ext-tile", "func::FuncOp"> {
  let summary = "Test pass for tiling using TiledOpInterface";
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::createTilingInterfaceTilingPass()";
}

def MaterializeEncoding :
    Pass<"iree-linalg-ext-materialize-encoding", "func::FuncOp"> {
  let summary = "Test pass to materialize ops with tensor encoding into ops with explicit data movement";
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::createMaterializeEncodingPass()";
}

def FoldIntoPackAndUnpackOps :
    Pass<"iree-linalg-ext-fold-into-pack-unpack-ops", "func::FuncOp"> {
  let summary = "Test pass to fold operations into pack and unpacl operations";
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::createFoldIntoPackAndUnpackOps()";
}

def PadContractionToBlockSize :
    Pass<"iree-linalg-pad-contraction-to-block-size", ""> {
  let summary = "Pads contraction (matmul) ops to next multiple of block size";
  let description = [{
    This pass will apply padding to any supported linalg contractions:
      * Row-major matmul:
          Padded to <rowAlignment x columnAlignment>

    Both rowAlignment and columnAlignment must be power-of-two values. If an
    op is already statically padded properly, no change will be made. However,
    if dynamic dimensions exist, padding will be applied regardless. Because
    of the dynamic case, applying this pass multiple times can result in
    mutation on each run.
  }];
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::createPadContractionToBlockSizePass()";
  let options = [
    Option<"rowAlignment", "rowAlignment", "int", /*default=*/"16",
           "The row-wise output block size">,
    Option<"columnAlignment", "columnAlignment", "int", /*default=*/"16",
           "The column-wise output block size">,
  ];
}

def TopkSplitReduction:
    Pass<"iree-linalg-ext-topk-split-reduction", "func::FuncOp"> {
  let summary = "Topk split reduction pass.";
  let description = [{
    Produces a "map-reduce" style of parallelizing a Topk Op. The op is split
    into two, on containing reducitons in parallel and the other contianing the
    combination of the parallel reductions into a final result.
  }];
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::createTopkSplitReductionPass()";
  let options = [
    ListOption<"splitRatios", "split-ratios", "int",
           "List of split reduction ratios">,
  ];
}

def TileAndDecomposeWinogradTransform :
    Pass<"iree-linalg-ext-tile-and-decompose-winograd", "func::FuncOp"> {
  let summary =
      "Tiles and decomposes winograd transform ops into linalg ops";
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::"
                    "createTileAndDecomposeWinogradTransformPass()";
}

def ConvertConv2DToWinograd :
    Pass<"iree-linalg-ext-convert-conv2d-to-winograd", ""> {
  let summary = "Convert linalg convolution ops to winograd based implementation";
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::createConvertConv2DToWinogradPass()";
}

def DecomposeSoftmax :
    Pass<"iree-linalg-ext-decompose-softmax", "func::FuncOp"> {
  let summary =
      "Decomposes softmax op into a sequence of linalg ops";
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::"
                    "createDecomposeSoftmaxPass()";
}

def TileAndDecomposeAttention :
    Pass<"iree-linalg-ext-tile-and-decompose-attention", "func::FuncOp"> {
  let summary =
      "Tiles and decomposes attention op into a sequence of linalg ops";
  let constructor = "mlir::iree_compiler::IREE::LinalgExt::"
                    "createTileAndDecomposeAttentionPass()";
}

//===---------------------------------------------------------------------====//
// Codegen Strategy passes moved into IREE
// TODO: Deprecate all this.
//===---------------------------------------------------------------------====//

def LinalgStrategyTilePass
    : Pass<"iree-linalg-strategy-tile-pass", "func::FuncOp"> {
  let summary = "Configurable pass to apply pattern-based linalg tiling.";
  let constructor = "createLinalgStrategyTilePass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,
  ];
}

// TODO: if/when we need finer control add an anchorOp option.
def LinalgStrategyDecomposePass
    : Pass<"iree-linalg-strategy-decompose-pass", "func::FuncOp"> {
  let summary = "Configurable pass to apply pattern-based generalization.";
  let constructor = "createLinalgStrategyDecomposePass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
  ];
}

def LinalgStrategyVectorizePass
    : Pass<"iree-linalg-strategy-vectorize-pass", "func::FuncOp"> {
  let summary = "Configurable pass to apply pattern-based linalg vectorization.";
  let constructor = "createLinalgStrategyVectorizePass()";
  let dependentDialects = ["linalg::LinalgDialect", "vector::VectorDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,
    Option<"vectorizePadding", "vectorize-padding", "bool", "false",
      "Enable vectorization of padding ops.">,
  ];
}

def LinalgStrategyEnablePass
    : Pass<"iree-linalg-strategy-enable-pass", "func::FuncOp"> {
  let summary = "Configurable pass to enable the application of other "
    "pattern-based linalg passes.";
  let constructor = "createLinalgStrategyEnablePass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
  ];
}

def LinalgStrategyLowerVectorsPass
    : Pass<"iree-linalg-strategy-lower-vectors-pass", "func::FuncOp"> {
  let summary = "Configurable pass to lower vector operations.";
  let constructor = "createLinalgStrategyLowerVectorsPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
  ];
}

def LinalgStrategyRemoveMarkersPass
    : Pass<"iree-linalg-strategy-remove-markers-pass", "func::FuncOp"> {
  let summary = "Cleanup pass that drops markers.";
  let constructor = "createLinalgStrategyRemoveMarkersPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
  ];
}

#endif  // IREE_DIALECT_LINALGEXT_PASSES
