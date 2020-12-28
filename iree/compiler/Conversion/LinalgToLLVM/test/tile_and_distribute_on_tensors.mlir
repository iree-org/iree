// RUN: iree-opt -split-input-file -verify-diagnostics -iree-codegen-llvm-linalg-tile-and-distribute-on-tensors=tile-sizes="1,2" %s | IreeFileCheck %s

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0) -> (2, -d0 + 4)>

// CHECK-LABEL: func @tensor
func @tensor() -> tensor<2x4xf32> {
  %A = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %B = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32>
  %C = iree.unfoldable_constant dense<1000.0> : tensor<2x4xf32>

  //  CHECK-DAG: %[[C1:.*]] = constant 1 : index
  //  CHECK-DAG: %[[C2:.*]] = constant 2 : index
  //  CHECK-DAG: %[[C4:.*]] = constant 4 : index
  //  CHECK-DAG: %[[bix:.*]] = hal.interface.workgroup.id[0] : index
  //  CHECK-DAG: %[[bdx:.*]] = hal.interface.workgroup.count[0] : index
  //  CHECK-DAG: %[[biy:.*]] = hal.interface.workgroup.id[1] : index
  //  CHECK-DAG: %[[bdy:.*]] = hal.interface.workgroup.count[1] : index
  //      CHECK: %{{.*}} = scf.for %[[I:.*]] = %[[biy]] to %[[C2]] step %[[bdy]] iter_args(%arg1 = %2) -> (tensor<2x4xf32>) {
  // CHECK-NEXT:   %[[bix_scaled:.*]] = muli %[[bix]], %[[C2]] : index
  // CHECK-NEXT:   %[[bdx_scaled:.*]] = muli %[[bdx]], %[[C2]] : index
  // CHECK-NEXT:   %{{.*}} = scf.for %[[J:.*]] = %[[bix_scaled]] to %[[C4]] step %[[bdx_scaled]] iter_args(%arg3 = %arg1) -> (tensor<2x4xf32>) {
  // CHECK-NEXT:     subtensor %{{.*}}[%[[I]], 0] [1, 3] [1, 1] : tensor<2x3xf32> to tensor<1x3xf32>
  //
  // Canonicalizations not yet powerful enough here.
  // CHECK-NEXT:     %[[J_slice_1:.*]] = affine.min #[[$MAP]](%[[J]])
  // CHECK-NEXT:     subtensor %1[0, %[[J]]] [3, %[[J_slice_1]]] [1, 1] : tensor<3x4xf32> to tensor<3x?xf32>
  //
  // Canonicalizations not yet powerful enough here.
  // CHECK-NEXT:     %[[J_slice_2:.*]] = affine.min #[[$MAP]](%[[J]])
  // CHECK-NEXT:     subtensor %arg3[%[[I]], %[[J]]] [1, %[[J_slice_2]]] [1, 1] : tensor<2x4xf32> to tensor<1x?xf32>
  // CHECK-NEXT:     linalg.matmul
  // CHECK-NEXT:     subtensor_insert {{.*}} : tensor<1x?xf32> into tensor<2x4xf32>
  // CHECK-NEXT:     scf.yield %{{.*}} : tensor<2x4xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT:   scf.yield %{{.*}} : tensor<2x4xf32>
  %E = linalg.matmul ins(%A, %B: tensor<2x3xf32>, tensor<3x4xf32>)
                    outs(%C: tensor<2x4xf32>) -> tensor<2x4xf32>
  return %E : tensor<2x4xf32>
}
