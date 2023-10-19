// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-flow-insert-debug-target-at-ordinal{break-debug-target=@target_func:1 trace-debug-target=@target_func:1})" %s | FileCheck %s --check-prefixes=CHECK,ORDINAL
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-flow-insert-debug-target-at-ordinal{break-debug-target=@target_func:0 trace-debug-target=@target_func:0})" %s | FileCheck %s --check-prefixes=ORDINAL_0
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-flow-insert-debug-target-at-symbol{break-debug-target=dispatch_1 trace-debug-target=dispatch_1[^0-9]})" %s | FileCheck %s --check-prefixes=CHECK,SYMBOL

// Multiple functions.

// CHECK-LABEL: func.func @target_func
// ORDINAL_0-LABEL: func.func @target_func
func.func @target_func(%arg0: tensor<4xf32>) -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  // CHECK: %[[D0:.+]] = flow.dispatch @dispatch_0::@dispatch_0_entry
  //      ORDINAL_0: flow.tensor.trace "dispatch_0::dispatch_0_entry::0 inputs"
  // ORDINAL_0-NEXT: %[[D0:.+]] = flow.dispatch @dispatch_0::@dispatch_0_entry
  %0 = flow.dispatch @dispatch_0::@dispatch_0_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // ORDINAL_0-NEXT: flow.tensor.trace "dispatch_0::dispatch_0_entry::0 outputs"
  // CHECK: %[[D1:.+]] = flow.dispatch @dispatch_1::@dispatch_1_entry
  %1 = flow.dispatch @dispatch_1::@dispatch_1_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = flow.dispatch @dispatch_2::@dispatch_2_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = hal.tensor.export %2 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: %[[EXPORT:.+]] = hal.tensor.export %[[D1]] : tensor<4xf32> -> !hal.buffer_view
  // CHECK: return %[[EXPORT]] : !hal.buffer_view
  return %3 : !hal.buffer_view
}

// CHECK-LABEL: func.func @other_func
func.func @other_func(%arg0: tensor<4xf32>) -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  // CHECK: %[[D3:.+]] = flow.dispatch @dispatch_3::@dispatch_3_entry
  %0 = flow.dispatch @dispatch_3::@dispatch_3_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>

  // CHECK: %[[D4:.+]] = flow.dispatch @dispatch_4::@dispatch_4_entry
  %1 = flow.dispatch @dispatch_4::@dispatch_4_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %[[D5:.+]] = flow.dispatch @dispatch_5::@dispatch_5_entry
  %2 = flow.dispatch @dispatch_5::@dispatch_5_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>

  // ORDINAL: %[[ORIGINAL_EXPORT:.+]] = hal.tensor.export %[[D5]] : tensor<4xf32> -> !hal.buffer_view
  // SYMBOL:  %[[BREAK_EXPORT:.+]] = hal.tensor.export %[[D5]] : tensor<4xf32> -> !hal.buffer_view
  %3 = hal.tensor.export %2 : tensor<4xf32> -> !hal.buffer_view

  // Only break on the symbol as the ordinal specifies a different function.
  // SYMBOL:  return %[[BREAK_EXPORT]] : !hal.buffer_view
  // ORDINAL: return %[[ORIGINAL_EXPORT]] : !hal.buffer_view
  return %3 : !hal.buffer_view
}

// -----

// Break on a dispatch with a different number of results.

// CHECK-LABEL: func.func @target_func
func.func @target_func(%arg0: tensor<4xf32>) -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  // CHECK: %[[D0:.+]] = flow.dispatch @dispatch_0::@dispatch_0_entry
  %0 = flow.dispatch @dispatch_0::@dispatch_0_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %[[D1:.+]]:2 = flow.dispatch @dispatch_1::@dispatch_1_entry
  %1:2 = flow.dispatch @dispatch_1::@dispatch_1_entry[%c4] (%arg0) : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  %2 = flow.dispatch @dispatch_2::@dispatch_2_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = hal.tensor.export %2 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: %[[EXPORT_0:.+]] = hal.tensor.export %[[D1]]#0 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: %[[EXPORT_1:.+]] = hal.tensor.export %[[D1]]#1 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: return %[[EXPORT_0]], %[[EXPORT_1]] : !hal.buffer_view
  return %3 : !hal.buffer_view
}

// -----

// Break/trace on a dispatch not found in the target function should do nothing.

// CHECK-LABEL: func.func @target_func
func.func @target_func(%arg0: tensor<4xf32>) -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  // CHECK: %[[D0:.+]] = flow.dispatch @dispatch_0::@dispatch_0_entry
  %0 = flow.dispatch @dispatch_0::@dispatch_0_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %[[D1:.+]] = hal.tensor.export %[[D0]] : tensor<4xf32> -> !hal.buffer_view
  %1 = hal.tensor.export %0 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: return %[[D1]] : !hal.buffer_view
  return %1 : !hal.buffer_view
}

// -----

// Combines tracing and breaking on the same dispatch.

// CHECK-LABEL: func.func @target_func
// CHECK-SAME:       %[[ARG0:.+]]: tensor<4xf32>
func.func @target_func(%arg0: tensor<4xf32>) -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  // CHECK: %[[D0:.+]] = flow.dispatch @dispatch_0::@dispatch_0_entry
  %0 = flow.dispatch @dispatch_0::@dispatch_0_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>

  // ORDINAL: flow.tensor.trace "dispatch_1::dispatch_1_entry::1 inputs" = [%[[ARG0]] : tensor<4xf32>]
  // SYMBOL:  flow.tensor.trace "dispatch_1::dispatch_1_entry inputs" = [%[[ARG0]] : tensor<4xf32>]
  // CHECK: %[[D1:.+]] = flow.dispatch @dispatch_1::@dispatch_1_entry
  %1 = flow.dispatch @dispatch_1::@dispatch_1_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // ORDINAL: flow.tensor.trace "dispatch_1::dispatch_1_entry::1 outputs" = [%[[D1]] : tensor<4xf32>]
  // SYMBOL:  flow.tensor.trace "dispatch_1::dispatch_1_entry outputs" = [%[[D1]] : tensor<4xf32>]

  %2 = flow.dispatch @dispatch_2::@dispatch_2_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = hal.tensor.export %2 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: %[[EXPORT:.+]] = hal.tensor.export %[[D1]] : tensor<4xf32> -> !hal.buffer_view
  // CHECK: return %[[EXPORT]] : !hal.buffer_view
  return %3 : !hal.buffer_view
}


// -----

// Checks regex matching on a dispatch symbol.

// CHECK-LABEL: func.func @target_func
func.func @target_func(%arg0: tensor<4xf32>) -> !hal.buffer_view {
  %c4 = arith.constant 4 : index

  // SYMBOL: flow.tensor.trace "dispatch_1::dispatch_1_entry inputs"
  // CHECK:  flow.dispatch @dispatch_1::@dispatch_1_entry
  %0 = flow.dispatch @dispatch_1::@dispatch_1_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // SYMBOL: flow.tensor.trace "dispatch_1::dispatch_1_entry outputs"

  // SYMBOL-NOT: flow.tensor.trace "dispatch_11::dispatch_11_entry inputs"
  // CHECK:      flow.dispatch @dispatch_11::@dispatch_11_entry
  %1 = flow.dispatch @dispatch_11::@dispatch_11_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // SYMBOL-NOT: flow.tensor.trace "dispatch_11::dispatch_11_entry outputs"

  %2 = hal.tensor.export %1 : tensor<4xf32> -> !hal.buffer_view
  return %2 : !hal.buffer_view
}
