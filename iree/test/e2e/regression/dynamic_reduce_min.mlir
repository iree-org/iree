// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=dylib-llvm-aot -function-input="2x4xf32=[[1.0,2.0,3.0,4.0],[-1.0,-2.0,-3.0,-4.0]]" %s | IreeFileCheck %s)
// RUN: [[ $IREE_VMVX_DISABLE == 1 ]] || (iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=vmvx -function-input="2x4xf32=[[1.0,2.0,3.0,4.0],[-1.0,-2.0,-3.0,-4.0]]" %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=vulkan-spirv -function-input="2x4xf32=[[1.0,2.0,3.0,4.0],[-1.0,-2.0,-3.0,-4.0]]" %s | IreeFileCheck %s)

func @reduce_min(%arg0: tensor<?x?xf32>) -> tensor<f32> {
  %0 = mhlo.constant dense<0x7F800000> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.minimum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK: EXEC @reduce_min
// CHECK: f32=-4
