// RUN: iree-opt --lower-xla-to-iree-interpreter %s --split-input-file | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @const
func @const() -> tensor<3xi32> {
  // CHECK: [[CONST:%.+]] = iree.constant dense<[1, 2, 3]> : tensor<3xi32>
  %0 = "xla_hlo.constant"() {value = dense<[1, 2, 3]> : tensor<3xi32>} : () -> tensor<3xi32>

  // CHECK: [[RES_TENSOR:%.+]] = iree.memref_to_tensor([[CONST]]
  // CHECK: return [[RES_TENSOR]]
  return %0 : tensor<3xi32>
}
