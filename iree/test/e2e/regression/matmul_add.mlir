// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir -export-all -iree-hal-target-backends=llvm-ir -iree-enable-matmul-fusion %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -export-all -iree-hal-target-backends=vulkan-spirv -iree-enable-matmul-fusion %s | IreeFileCheck %s)

func @matmul_add() -> tensor<2x4xf32> {
  %0 = iree.unfoldable_constant dense<[
    [1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0],
    [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]> : tensor<2x8xf32>
  %1 = iree.unfoldable_constant dense<[
    [ 1.0,  2.0,  3.0,  4.0],
    [ 5.0,  6.0,  7.0,  8.0],
    [ 9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
    [17.0, 18.0, 19.0, 20.0],
    [21.0, 22.0, 23.0, 24.0],
    [25.0, 26.0, 27.0, 28.0],
    [29.0, 30.0, 31.0, 32.0]]> : tensor<8x4xf32>
  %2 = iree.unfoldable_constant dense<[
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  %3 = "mhlo.dot"(%0, %1) : (tensor<2x8xf32>, tensor<8x4xf32>) -> tensor<2x4xf32>
  %4 = mhlo.add %3, %2 : tensor<2x4xf32>
  return %4 : tensor<2x4xf32>
}

// CHECK: EXEC @matmul_add
// CHECK: 2x4xf32=[709 746 783 820][1673 1774 1875 1976]
