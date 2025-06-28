// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_COMMON_CPU_PASSES
#define IREE_CODEGEN_COMMON_CPU_PASSES

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// Common Passes used for CPU-like backends (keep alphabetical)
//===---------------------------------------------------------------------===//

def CPULowerToUKernelsPass :
    Pass<"iree-codegen-cpu-lower-to-ukernels", ""> {
  let summary =
      "Separate out parts of the IR that lower to a micro-kernel";
  let options = [
    Option<"skipIntermediateRoundings", "skip-intermediate-roundings",
      "bool", /*default=*/"true",
      "Allow skipping intermediate roundings, e.g. in f16 ukernels internally doing f32 arithmetic.">,
  ];
}

def CPUPrepareUkernelsPass :
    InterfacePass<"iree-codegen-cpu-prepare-ukernels", "mlir::FunctionOpInterface"> {
  let summary = "Rank reduce operations to fit existing ukernels requirements."
                "For example, batch_mmt4d ops are decomposed to mmt4d ops";
}

def CPUTileAndDistributeToWorkgroupsPass :
    InterfacePass<"iree-codegen-cpu-tile-and-distribute-to-workgroups",
                  "mlir::FunctionOpInterface"> {
  let summary = "Tile and distribute operations to workgroups using scf.forall op.";
  let description = [{
    Tile and distribute operations to workgroups using scf.forall op. Different
    from the shared TileAndDistributeToWorkgroupsUsingForallOpPass, the CPU
    version starts the tiling with the root op.
  }];

  let dependentDialects = [
    "affine::AffineDialect",
    "IREE::Codegen::IREECodegenDialect",
    "IREE::LinalgExt::IREELinalgExtDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
  ];
}


#endif  // IREE_CODEGEN_COMMON_CPU_PASSES
