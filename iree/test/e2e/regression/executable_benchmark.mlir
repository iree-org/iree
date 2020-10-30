// Only checks registered benchmarks.
// RUN: iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-benchmark-vm-bytecode-module %s -o ${TEST_TMPDIR?}/bc.module && iree-benchmark-module --driver=vmla -module_file=${TEST_TMPDIR?}/bc.module --benchmark_list_tests=true | IreeFileCheck %s

func @two_dispatch() -> (tensor<5x5xf32>, tensor<3x5xf32>) attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<1.0> : tensor<5x3xf32>
  %1 = iree.unfoldable_constant dense<0.4> : tensor<3x5xf32>
  %2 = "mhlo.dot"(%0, %1) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
  %3 = "mhlo.dot"(%1, %2) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
  return %2, %3 : tensor<5x5xf32>, tensor<3x5xf32>
}
// CHECK: BM_two_dispatch_ex_dispatch_0_entry
// CHECK: BM_two_dispatch_ex_dispatch_1_entry
// CHECK: BM_two_dispatch
