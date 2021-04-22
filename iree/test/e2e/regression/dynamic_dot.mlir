// RUN: iree-run-mlir -export-all -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir -export-all -iree-hal-target-backends=dylib-llvm-aot %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @dynamic_dot
func @dynamic_dot() -> tensor<?x?xf32> attributes { iree.module.export } {
  %lhs = iree.dynamic_shape_constant dense<[
    [15.0, 14.0, 13.0],
    [12.0, 11.0, 10.0],
    [09.0, 08.0, 07.0],
    [06.0, 05.0, 04.0],
    [03.0, 02.0, 01.0]]> : tensor<5x3xf32> -> tensor<?x?xf32>
  %rhs = iree.dynamic_shape_constant dense<[
    [15.0, 14.0, 13.0, 12.0, 11.0],
    [10.0, 09.0, 08.0, 07.0, 06.0],
    [05.0, 04.0, 03.0, 02.0, 01.0]]> : tensor<3x5xf32> -> tensor<?x?xf32>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %res : tensor<?x?xf32>
}

// CHECK: 5x5xf32=[430 388 346 304 262][340 307 274 241 208][250 226 202 178 154][160 145 130 115 100][70 64 58 52 46]
