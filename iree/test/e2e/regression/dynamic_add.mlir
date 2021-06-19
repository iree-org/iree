// RUN: iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=vmvx -function-input="2x4xf32=[[1.0,2.0,3.0,4.0],[-1.0,-2.0,-3.0,-4.0]]" -function-input="2x4xf32=[[5.0,6.0,7.0,8.0],[-5.0,-6.0,-7.0,-8.0]]" %s | IreeFileCheck %s
// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=dylib-llvm-aot -function-input="2x4xf32=[[1.0,2.0,3.0,4.0],[-1.0,-2.0,-3.0,-4.0]]" -function-input="2x4xf32=[[5.0,6.0,7.0,8.0],[-5.0,-6.0,-7.0,-8.0]]" %s | IreeFileCheck %s)

// CHECK: EXEC @main
// CHECK: 2x4xf32=[6 8 10 12][-6 -8 -10 -12]

func @main(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  return %0: tensor<?x4xf32>
}
