// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: func @inc_rgn_dispatch_0
// CHECK-SAME: (%[[INTERFACE:.+]]: !vmla.interface
func @inc_rgn_dispatch_0() attributes {iree.module.export} {
  // CHECK-DAG: %[[C0:.+]] = constant 0
  // CHECK-DAG: %[[C4:.+]] = constant 4
  %c0 = constant 0 : index
  // CHECK-DAG: %[[CST1:.+]] = vmla.constant dense<1.000000e+00> : tensor<f32> -> !vmla.buffer
  %cst = constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[SET0BINDING0:.+]] = vmla.interface.binding %[[INTERFACE]] {binding = 0 : i32, set = 0 : i32} : !vmla.buffer
  // CHECK-NEXT: %[[ARG0:.+]] = vmla.buffer.view %[[SET0BINDING0]][%[[C0]]], byte_length = %[[C4]] : !vmla.buffer
  %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<f32>
  // CHECK-NEXT: %[[TEMP:.+]] = vmla.buffer.alloc byte_length = %[[C4]] : !vmla.buffer
  // CHECK-NEXT: vmla.add %[[ARG0]], %[[CST1]], out %[[TEMP]] : f32
  %1 = xla_hlo.add %0, %cst : tensor<f32>
  // CHECK-NEXT: %[[SET0BINDING1:.+]] = vmla.interface.binding %[[INTERFACE]] {binding = 1 : i32, set = 0 : i32} : !vmla.buffer
  // CHECK-NEXT: vmla.buffer.copy %[[TEMP]][%[[C0]]], out %[[SET0BINDING1]][%[[C0]]], byte_length = %[[C4]]
  hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<f32>
  return
}
func @inc_rgn_dispatch_0_impl(%arg0: tensor<f32>) -> tensor<f32> attributes {iree.module.export, sym_visibility = "private"} {
  %cst = constant dense<1.000000e+00> : tensor<f32>
  %0 = xla_hlo.add %arg0, %cst : tensor<f32>
  return %0 : tensor<f32>
}
hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
