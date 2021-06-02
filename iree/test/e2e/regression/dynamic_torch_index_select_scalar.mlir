// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir %s --iree-input-type=mhlo -iree-hal-target-backends=dylib-llvm-aot -function-input="5x1x5xi32=[[1,2,3,4,5]] [[6,7,8,9,10]] [[11,12,13,14,15]] [[16,17,18,19,20]] [[21,22,23,24,25]]" -function-input="i32=0" | IreeFileCheck %s)

// CHECK-LABEL: EXEC @torch_index_select1
func @torch_index_select1(%arg0: tensor<?x?x?xi32>, %arg1: tensor<i32>) -> tensor<?x?xi32> attributes {iree.module.export} {
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// CHECK: 1x5xi32=[1 2 3 4 5]

// CHECK-LABEL: EXEC @torch_index_select2
func @torch_index_select2(%arg0: tensor<?x?x?xi32>, %arg1: tensor<i32>) -> tensor<?x?xi32> attributes {iree.module.export} {
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {batch_dims = 0 : i64, dim = 1 : i64} : (tensor<?x?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// CHECK: 5x5xi32=[1 2 3 4 5][6 7 8 9 10][11 12 13 14 15][16 17 18 19 20][21 22 23 24 25]
