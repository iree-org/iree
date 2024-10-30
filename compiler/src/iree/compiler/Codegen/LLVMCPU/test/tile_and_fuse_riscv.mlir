// RUN: iree-opt --split-input-file \
// RUN: --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-and-fuse{tiling-level=1}, iree-llvmcpu-tile-and-fuse{tiling-level=3}, iree-llvmcpu-peel))" %s | FileCheck %s

// -----

/// This test checks we successfully tile the matmul to 7 x m4 RVV tile_size.

/// Simple RVV lowering config, derived from the VLEN=512 for RVV on RISC-V 64.
#rvv_lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 0], [64, 128, 0], [0, 0, 0], [7, 64, 0], [0, 0, 1], [0, 0, 0]]>

func.func @f32_rvv_matmul(%A: tensor<384x1024xf32>, %B: tensor<1024x512xf32>, %C: tensor<384x512xf32>) -> tensor<384x512xf32>{
  %1 = linalg.matmul {lowering_config = #rvv_lowering_config} ins(%A, %B: tensor<384x1024xf32>, tensor<1024x512xf32>)
            outs(%C: tensor<384x512xf32>) -> tensor<384x512xf32>
  return %1 : tensor<384x512xf32>
}
// CHECK-LABEL: func.func @f32_rvv_matmul(
// CHECK:     %[[c7:.+]] = arith.constant 7 : index
// CHECK:     %[[c128:.+]] = arith.constant 128 : index
// CHECK:     %[[c512:.+]] = arith.constant 512 : index
// CHECK:     %[[c64:.+]] = arith.constant 64 : index
// CHECK:     scf.for {{.*}} step %[[c64]]
// CHECK:       scf.for {{.*}} step %[[c128]]
// CHECK:         scf.for {{.*}} step %[[c7]]
// CHECK:           scf.for {{.*}} step %[[c64]]
// CHECK:             linalg.matmul{{.*}}: tensor<7x1024xf32>, tensor<1024x64xf32>) outs({{.*}} : tensor<7x64xf32>) -> tensor<7x64xf32>
// CHECK:         scf.for {{.*}} step %[[c64]]
// CHECK:           linalg.matmul{{.*}}: tensor<1x1024xf32>, tensor<1024x64xf32>) outs({{.*}} : tensor<1x64xf32>) -> tensor<1x64xf32>
