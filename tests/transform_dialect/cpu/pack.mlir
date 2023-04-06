func.func @static_pack_simple() -> tensor<2x2x2x2xi32> {
  %iree_input = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %init = tensor.empty() : tensor<2x2x2x2xi32>
  %pack = tensor.pack %iree_input inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %init
      : tensor<4x4xi32> -> tensor<2x2x2x2xi32>
  return %pack : tensor<2x2x2x2xi32>
}

// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu \
// RUN: --iree-codegen-llvmcpu-use-transform-dialect=%p/pack_spec.mlir | \
// RUN: iree-run-module --function=static_pack_simple | \
// RUN: FileCheck %s --check-prefixes=EXEC

// EXEC:      2x2x2x2xi32
// EXEC-SAME:   [0 1][4 5]
// EXEC-SAME:   [2 3][6 7]
// EXEC-SAME:   [8 9][12 13]
// EXEC-SAME:   [10 11][14 15]
