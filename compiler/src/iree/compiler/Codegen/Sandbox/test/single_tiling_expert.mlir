// RUN: iree-opt --pass-pipeline="builtin.module(func.func(linalg-single-tiling-expert-driver{tiling-level=0 vectorize}))" --split-input-file %s | FileCheck %s

func.func @matmul(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[10, 20, 30]]>}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func.func @matmul(
//      CHECK:   scf.for
// CHECK-SAME:   {
//      CHECK:     scf.for
// CHECK-SAME:     {
//      CHECK:       scf.for
// CHECK-SAME:       {
//      CHECK:         linalg.matmul
//      CHECK:       }
//      CHECK:     }
//      CHECK:   }

// -----

 func.func @matmul_static(%arg0 : tensor<20x60xf32>, %arg1 : tensor<60x80xf32>, %arg2 : tensor<20x80xf32>) -> tensor<20x80xf32> {
  %0 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[10, 20, 30]]>}
      ins(%arg0, %arg1 : tensor<20x60xf32>, tensor<60x80xf32>)
      outs(%arg2 : tensor<20x80xf32>) -> tensor<20x80xf32>
  return %0 : tensor<20x80xf32>
}
//      CHECK: func.func @matmul_static(
//      CHECK:   scf.for
// CHECK-SAME:   {
//      CHECK:     scf.for
// CHECK-SAME:     {
//      CHECK:       scf.for
// CHECK-SAME:       {
//  CHECK-NOT:         linalg.matmul
//      CHECK:         vector.contract
//  CHECK-NOT:         linalg.matmul
//      CHECK:       }
//      CHECK:     }
//      CHECK:   }
