// RUN: iree-run-mlir %s -iree-hal-target-backends=vmla -input-value="3x2x2xi32=[[1, 2] [3, 4]] [[5, 6] [7, 8]] [[9, 10] [11, 12]]" -input-value="2xi32=[0, 1]" | IreeFileCheck %s
// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir %s -iree-hal-target-backends=llvm-ir -input-value="3x2x2xi32=[[1, 2] [3, 4]] [[5, 6] [7, 8]] [[9, 10] [11, 12]]" -input-value="2xi32=[0, 1]" | IreeFileCheck %s)

// CHECK-LABEL: EXEC @torch_index_select1
func @torch_index_select1(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?x?xi32> attributes {iree.module.export} {
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {batch_dims = 0 : i64, dim = 1 : i64} : (tensor<?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}

// CHECK: 3x2x2xi32=[
// CHECK-SAME:   [1 2][3 4]
// CHECK-SAME: ][
// CHECK-SAME:   [5 6][7 8]
// CHECK-SAME: ][
// CHECK-SAME:   [9 10][11 12]
// CHECK-SAME: ]

// CHECK-LABEL: EXEC @torch_index_select2
func @torch_index_select2(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?x?xi32> attributes {iree.module.export} {
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}

// CHECK: 2x2x2xi32=[
// CHECK-SAME:   [1 2][3 4]
// CHECK-SAME: ][
// CHECK-SAME:   [5 6][7 8]
// CHECK-SAME: ]

