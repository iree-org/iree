// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s -input-value="2x3xf32=[[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]" | IreeFileCheck %s

// CHECK-LABEL: EXEC @dynamic_tensor
func @dynamic_tensor(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32>
      attributes { iree.module.export } {
  %0 = "xla_hlo.abs"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK: 2x3xf32=[1 2 3][4 5 6]
