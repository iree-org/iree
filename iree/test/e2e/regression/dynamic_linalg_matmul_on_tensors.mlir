// RUN: iree-run-mlir %s -iree-hal-target-backends=dylib-llvm-aot -iree-flow-dispatch-linalg-on-tensors-tile-sizes="1,1" -function-input="2x3xf32=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"  -function-input="3x4xf32=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]" -function-input="2x4xf32=[[1000.0, 1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0, 1000.0]]" | FileCheck %s
// RUN: [[ $IREE_VMVX_DISABLE == 1 ]] || (iree-run-mlir %s -iree-hal-target-backends=vmvx -iree-flow-dispatch-linalg-on-tensors-tile-sizes="1,1" -function-input="2x3xf32=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"  -function-input="3x4xf32=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]" -function-input="2x4xf32=[[1000.0, 1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0, 1000.0]]" | FileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir %s -iree-hal-target-backends=vulkan-spirv -iree-flow-dispatch-linalg-on-tensors-tile-sizes="1,1" -function-input="2x3xf32=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"  -function-input="3x4xf32=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]" -function-input="2x4xf32=[[1000.0, 1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0, 1000.0]]" | FileCheck %s)

// CHECK: EXEC @main
// CHECK: 2x4xf32=[1038 1044 1050 1056][1083 1098 1113 1128]
func.func @main(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> tensor<?x?xf32> attributes {iree.module.export}
{
  %D = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %D: tensor<?x?xf32>
}
