// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values="2x3xi32=[1 2 3 4 5 6]" --output_types=i | FileCheck %s --dump-input=fail

// CHECK-LABEL: EXEC @pad
func @pad(%arg : tensor<2x3xi32>) -> tensor<3x6xi32> {
  %pad_val = constant dense<0> : tensor<i32>
  %result = "xla_hlo.pad"(%arg, %pad_val) {interior_padding = dense<0> : tensor<2xi64>, edge_padding_low = dense<[0,1]> : tensor<2xi64>, edge_padding_high = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<3x6xi32>
  return %result : tensor<3x6xi32>
}
// CHECK-NEXT: 3x6xi32=
// CHECK-SAME:     [0 1 2 3 0 0]
// CHECK-SAME:     [0 4 5 6 0 0]
// CHECK-SAME:     [0 0 0 0 0 0]
