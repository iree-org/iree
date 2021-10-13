// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir %s -iree-hal-target-backends=dylib-llvm-aot -iree-flow-dispatch-linalg-on-tensors-enable-fusion -iree-flow-dispatch-linalg-on-tensors-tile-sizes="1,1" -function-input="2x3xf32=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"  -function-input="3x4xf32=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]"  -function-input="2x4xf32=[[1000.0, 1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0, 1000.0]]" | IreeFileCheck %s)
// RUN: [[ $IREE_VMVX_DISABLE == 1 ]] || (iree-run-mlir %s -iree-hal-target-backends=vmvx -iree-flow-dispatch-linalg-on-tensors-enable-fusion -iree-flow-dispatch-linalg-on-tensors-tile-sizes="1,1" -function-input="2x3xf32=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"  -function-input="3x4xf32=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]"  -function-input="2x4xf32=[[1000.0, 1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0, 1000.0]]" | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir %s -iree-hal-target-backends=vulkan-spirv -iree-flow-dispatch-linalg-on-tensors-enable-fusion -iree-flow-dispatch-linalg-on-tensors-tile-sizes="1,1" -function-input="2x3xf32=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"  -function-input="3x4xf32=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]"  -function-input="2x4xf32=[[1000.0, 1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0, 1000.0]]" | IreeFileCheck %s)

// CHECK: EXEC @main
// CHECK: 2x4xf32=[985 982 979 976][985 982 979 976]
func @main(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> tensor<?x?xf32> attributes {iree.module.export}
{
  %AA = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"] }
    outs(%A : tensor<?x?xf32>) {
    ^bb0(%b: f32):
      // TODO(nicolasvasilache): The constant ends up being hoisted and turns
      // into a pushconstant. But pushconstants must be integers, so we use
      // sitofp to temporarily circumvent the problem.
      %im1 = arith.constant -1 : i32
      %fm1 = arith.sitofp %im1: i32 to f32
      linalg.yield %fm1 : f32
    } -> tensor<?x?xf32>
  %D = linalg.matmul ins(%AA, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %D: tensor<?x?xf32>
}
