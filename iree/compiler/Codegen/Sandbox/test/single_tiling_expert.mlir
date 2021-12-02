// RUN: iree-opt -pass-pipeline="builtin.func(linalg-single-tiling-expert-driver{anchor-func=matmul anchor-op=linalg.matmul tile-sizes=10,20,30 vectorize})" -split-input-file %s | IreeFileCheck %s

func @matmul(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func @matmul(
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

 func @matmul_static(%arg0 : tensor<20x60xf32>, %arg1 : tensor<60x80xf32>, %arg2 : tensor<20x80xf32>) -> tensor<20x80xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<20x60xf32>, tensor<60x80xf32>)
      outs(%arg2 : tensor<20x80xf32>) -> tensor<20x80xf32>
  return %0 : tensor<20x80xf32>
}
//      CHECK: func @matmul_static(
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
