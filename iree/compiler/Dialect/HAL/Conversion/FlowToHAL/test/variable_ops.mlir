// RUN: iree-opt -split-input-file -iree-convert-flow-to-hal %s | IreeFileCheck %s

// CHECK-LABEL: hal.variable @var_i32 mutable : !ireex.ref<!hal.buffer>
flow.variable @var_i32 mutable : tensor<i32>
func @fn() {
  // CHECK: [[V:%.+]] = hal.variable.load @var_i32 : !ireex.ref<!hal.buffer>
  %0 = flow.variable.load @var_i32 : tensor<i32>
  // CHECK-NEXT: hal.variable.store [[V]], @var_i32 : !ireex.ref<!hal.buffer>
  flow.variable.store %0, @var_i32 : tensor<i32>
  return
}

// -----

// CHECK-LABEL: hal.variable @var_i1 mutable : !ireex.ref<!hal.buffer>
flow.variable @var_i1 mutable : tensor<i1>
func @fn() {
  // CHECK: [[V:%.+]] = hal.variable.load @var_i1 : !ireex.ref<!hal.buffer>
  %0 = flow.variable.load @var_i1 : tensor<i1>
  // CHECK-NEXT: hal.variable.store [[V]], @var_i1 : !ireex.ref<!hal.buffer>
  flow.variable.store %0, @var_i1 : tensor<i1>
  return
}
