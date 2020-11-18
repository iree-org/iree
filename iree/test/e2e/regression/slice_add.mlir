// RUN: iree-run-mlir -export-all -iree-hal-target-backends=vmla -function-input="3x4xi32=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]" -function-input="1x2xi32=10" %s | IreeFileCheck %s
// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir -export-all -iree-hal-target-backends=llvm-ir -function-input="3x4xi32=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]" -function-input="1x2xi32=10" %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -export-all -iree-hal-target-backends=vulkan-spirv -function-input="3x4xi32=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]" -function-input="1x2xi32=10" %s | IreeFileCheck %s)

// CHECK: EXEC @slice_stride_part
// CHECK: 1x2xi32=[16 17]
func @slice_stride_part(%arg0: tensor<3x4xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  %1 = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 1]> : tensor<2xi64>,
    limit_indices = dense<[2, 3]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  %2 = mhlo.add %1, %arg1 : tensor<1x2xi32>
  return %2 : tensor<1x2xi32>
}
