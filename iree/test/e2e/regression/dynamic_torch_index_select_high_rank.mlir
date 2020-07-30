// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir %s -iree-hal-target-backends=llvm-ir -input-value="2x2xi32=[6, 7] [8, 9]" -input-value="2x2x2x2xi32=[[[0, 1] [1, 0]] [[0, 0] [1, 1]]] [[[1, 1] [0, 0]] [[0, 1] [1, 0]]]" | IreeFileCheck %s)

// CHECK-LABEL: EXEC @torch_index_select1
func @torch_index_select1(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> attributes {iree.module.export} {
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {batch_dims = 1 : i64, dim = 1 : i64} : (tensor<?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %0 : tensor<?x?x?x?xi32>
}

// CHECK: 2x2x2x2xi32=[
// CHECK-SAME:   [
// CHECK-SAME:     [6 7][7 6]
// CHECK-SAME:   ][
// CHECK-SAME:     [6 6][7 7]
// CHECK-SAME:   ]
// CHECK-SAME: ][
// CHECK-SAME:   [
// CHECK-SAME:     [9 9][8 8]
// CHECK-SAME:   ][
// CHECK-SAME:     [8 9][9 8]
// CHECK-SAME:   ]
// CHECK-SAME: ]

// CHECK-LABEL: EXEC @torch_index_select2
func @torch_index_select2(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?x?xi32> attributes {iree.module.export} {
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?x?xi32>
  return %0 : tensor<?x?x?x?x?xi32>
}

// CHECK: 2x2x2x2x2xi32=[
// CHECK-SAME:   [
// CHECK-SAME:     [
// CHECK-SAME:       [6 7][8 9]
// CHECK-SAME:     ][
// CHECK-SAME:       [8 9][6 7]
// CHECK-SAME:     ]
// CHECK-SAME:   ][
// CHECK-SAME:     [
// CHECK-SAME:       [6 7][6 7]
// CHECK-SAME:     ][
// CHECK-SAME:       [8 9][8 9]
// CHECK-SAME:     ]
// CHECK-SAME:   ]
// CHECK-SAME: ][
// CHECK-SAME:   [
// CHECK-SAME:     [
// CHECK-SAME:       [8 9][8 9]
// CHECK-SAME:     ][
// CHECK-SAME:       [6 7][6 7]
// CHECK-SAME:     ]
// CHECK-SAME:   ][
// CHECK-SAME:     [
// CHECK-SAME:       [6 7][8 9]
// CHECK-SAME:     ][
// CHECK-SAME:       [8 9][6 7]
// CHECK-SAME:     ]
// CHECK-SAME:   ]
// CHECK-SAME: ]

