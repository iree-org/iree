// RUN: iree-opt -iree-codegen-fold-tensor-extract-op %s | IreeFileCheck %s

func @fold_tensor_extract(%arg0 : memref<2x3xi32>) -> i32
{
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0 = memref.tensor_load %arg0 : memref<2x3xi32>
  %1 = tensor.extract %0[%c1, %c2] : tensor<2x3xi32>
  return %1 : i32
}
//      CHECK: func @fold_tensor_extract
// CHECK-SAME:   %[[ARG0:.+]]: memref<2x3xi32>
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//      CHECK:   %[[SCALAR:.+]] = memref.load %[[ARG0]][%[[C1]], %[[C2]]]
//      CHECK:   return %[[SCALAR]]
