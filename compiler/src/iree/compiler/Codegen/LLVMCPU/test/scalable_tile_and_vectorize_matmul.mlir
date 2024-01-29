// RUN: iree-opt --split-input-file \
// RUN: --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-and-fuse{tiling-level=1},iree-llvmcpu-tile{tiling-level=2},iree-codegen-generic-vectorization{enable-vector-masking}))" %s | FileCheck %s

// -----

/// This test checks we successfully tile the matmul and invoke the linalg vectorizer,
/// and produce scalable vector ops.

/// Simple scalable lowering config, derived from the default for SVE on AArch64.
#scalable_lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [1, [32], 0], [0, 0, 1], [0, 0, 0]]>

func.func @scalable_matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32>{
  %1 = linalg.matmul {lowering_config = #scalable_lowering_config} ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @scalable_matmul(
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK:       %[[VSCALE:.*]] = vector.vscale
// CHECK-DAG:   %[[SCALABLE_TILE_SIZE:.*]] = arith.muli %[[VSCALE]], %[[C32]] : index
// CHECK:       scf.for
// CHECK-SAME:      step %[[C1]]
// CHECK:         scf.for
// CHECK-SAME:        step %[[SCALABLE_TILE_SIZE]]
// CHECK:           scf.for
// CHECK-SAME:          step %[[C1]]
// CHECK:             vector.create_mask {{.*}} : vector<1x[32]xi1>
// CHECK:             vector.mask
// CHECK-SAME:          vector.contract
// CHECK-SAME:          vector<1x1xf32>, vector<1x[32]xf32> into vector<1x[32]xf32>
