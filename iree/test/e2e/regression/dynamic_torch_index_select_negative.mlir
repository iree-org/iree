// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir %s -iree-hal-target-backends=llvm-ir -input-value="2x2x2xi32=[[100, 101] [110, 111]] [[200, 201] [210, 211]]" -input-value="2x2x2xi32=[[0, 1] [1, 0]] [[0, 0] [1, 1]]" | IreeFileCheck %s)

// CHECK-LABEL: EXEC @torch_index_select1
func @torch_index_select1(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> attributes {iree.module.export} {
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {batch_dims = -1 : i64, dim = -1 : i64} : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}

// CHECK: 2x2x2xi32=[
// CHECK-SAME:   [100 101][111 110]
// CHECK-SAME: ][
// CHECK-SAME:   [200 200][211 211]
// CHECK-SAME: ]

