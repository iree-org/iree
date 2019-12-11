// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values="2x6xf32= 1 2 3 4 5 6 7 8 9 10 11 12" | IreeFileCheck %s
// RUN: iree-run-mlir --target_backends=vulkan-spirv %s --input_values="2x6xf32= 1 2 3 4 5 6 7 8 9 10 11 12" | IreeFileCheck %s

// CHECK-LABEL: EXEC @reshape_3D_2D
func @reshape_3D_2D(%arg : tensor<2x6xf32>) -> tensor<2x1x6xf32> {
  %result = "xla_hlo.reshape"(%arg) : (tensor<2x6xf32>) -> tensor<2x1x6xf32>
  return %result : tensor<2x1x6xf32>
}
// CHECK: 2x1x6xf32={{\[}}[1 2 3 4 5 6]{{\]}}{{\[}}[7 8 9 10 11 12]{{\]}}
