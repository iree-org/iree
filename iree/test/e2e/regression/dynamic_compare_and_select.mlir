// RUN: iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=dylib-llvm-aot -function-input="10xi32=[0,1,2,3,4,5,6,7,8,9]" -function-input="10xi32=[0,1,2,3,4,5,6,7,8,9]" -function-input="10xi32=[0,1,2,3,4,5,6,7,8,9]" -function-input="10xi32=[9,8,7,6,5,4,3,2,1,0]" %s | FileCheck %s
// RUN: [[ $IREE_VMVX_DISABLE == 1 ]] || (iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=vmvx -function-input="10xi32=[0,1,2,3,4,5,6,7,8,9]" -function-input="10xi32=[0,1,2,3,4,5,6,7,8,9]" -function-input="10xi32=[0,1,2,3,4,5,6,7,8,9]" -function-input="10xi32=[9,8,7,6,5,4,3,2,1,0]" %s | FileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=vulkan-spirv -function-input="10xi32=[0,1,2,3,4,5,6,7,8,9]" -function-input="10xi32=[0,1,2,3,4,5,6,7,8,9]" -function-input="10xi32=[0,1,2,3,4,5,6,7,8,9]" -function-input="10xi32=[9,8,7,6,5,4,3,2,1,0]" %s | FileCheck %s)

// CHECK: EXEC @main
// CHECK: 10xi32=9 8 7 6 5 4 3 2 1 0

func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<?xi32>, %arg3: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
    %2 = "mhlo.select"(%1, %arg2, %arg3) : (tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %2 : tensor<?xi32>
}
