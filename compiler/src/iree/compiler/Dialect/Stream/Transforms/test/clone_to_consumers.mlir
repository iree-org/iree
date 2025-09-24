// RUN: iree-opt --split-input-file --iree-stream-clone-to-consumers %s | FileCheck %s

// Tests that splats (which are cloneable ops) are cloned for every user.

// CHECK-LABEL: @splatOp
util.func private @splatOp() -> (tensor<1xi32>, tensor<1xi32>) {
  %splat_value = arith.constant 123 : i32
  //      CHECK: %[[SPLAT_A:.+]] = flow.tensor.splat
  %splat = flow.tensor.splat %splat_value : tensor<1xi32>
  // CHECK-NEXT: %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[SPLAT_A]]
  %transfer_a = flow.tensor.transfer %splat : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK-NEXT: %[[SPLAT_B:.+]] = flow.tensor.splat
  // CHECK-NEXT: %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[SPLAT_B]]
  %transfer_b = flow.tensor.transfer %splat : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK-NEXT: util.return %[[TRANSFER_A]], %[[TRANSFER_B]]
  util.return %transfer_a, %transfer_b : tensor<1xi32>, tensor<1xi32>
}

// -----

// Tests that a cloneable op with an explicit affinity assigned is not cloned.
// This allows users to avoid the clones when they know they want a transfer.

// CHECK-LABEL: @pinnedSplatOp
util.func private @pinnedSplatOp() -> (tensor<1xi32>, tensor<1xi32>) {
  %splat_value = arith.constant 123 : i32
  //      CHECK: %[[SPLAT_A:.+]] = flow.tensor.splat
  %splat_a = flow.tensor.splat %splat_value : tensor<1xi32> attributes {stream.affinity = #hal.device.promise<@dev_a>}
  // CHECK-NEXT: %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[SPLAT_A]]
  %transfer_a = flow.tensor.transfer %splat_a : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK-NEXT: %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[SPLAT_A]]
  %transfer_b = flow.tensor.transfer %splat_a : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK-NEXT: util.return %[[TRANSFER_A]], %[[TRANSFER_B]]
  util.return %transfer_a, %transfer_b : tensor<1xi32>, tensor<1xi32>
}

// -----

// Tests that pure ops (here arith.select) also get cloned. Note that we clone
// the entire use-def chain up to the root splat.

// CHECK-LABEL: @selectOp
util.func private @selectOp(%cond: i1) -> (tensor<1xi32>, tensor<1xi32>) {
  // CHECK-DAG: %[[SPLAT0_VALUE:.+]] = arith.constant 123
  %splat0_value = arith.constant 123 : i32
  // CHECK-DAG: %[[SPLAT1_VALUE:.+]] = arith.constant 456
  %splat1_value = arith.constant 456 : i32
  // CHECK-DAG: %[[SPLAT0_A:.+]] = flow.tensor.splat %[[SPLAT0_VALUE]]
  %splat0 = flow.tensor.splat %splat0_value : tensor<1xi32>
  // CHECK-DAG: %[[SPLAT1_A:.+]] = flow.tensor.splat %[[SPLAT1_VALUE]]
  %splat1 = flow.tensor.splat %splat1_value : tensor<1xi32>
  // CHECK-DAG: %[[SELECT_A:.+]] = arith.select {{.+}}, %[[SPLAT0_A]], %[[SPLAT1_A]]
  %select = arith.select %cond, %splat0, %splat1 : tensor<1xi32>
  // CHECK-NEXT: %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[SELECT_A]]
  %transfer_a = flow.tensor.transfer %select : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK-DAG: %[[SPLAT0_B:.+]] = flow.tensor.splat %[[SPLAT0_VALUE]]
  // CHECK-DAG: %[[SPLAT1_B:.+]] = flow.tensor.splat %[[SPLAT1_VALUE]]
  // CHECK-DAG: %[[SELECT_B:.+]] = arith.select {{.+}}, %[[SPLAT0_B]], %[[SPLAT1_B]]
  // CHECK-NEXT: %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[SELECT_B]]
  %transfer_b = flow.tensor.transfer %select : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK-NEXT: util.return %[[TRANSFER_A]], %[[TRANSFER_B]]
  util.return %transfer_a, %transfer_b : tensor<1xi32>, tensor<1xi32>
}

// -----

// Tests that a dispatch that consumes nothing is cloned for every user.
// This is a special case where a dispatch is acting like a fancy splat that can
// arise when generating masks, random numbers, etc.

// CHECK-LABEL: @splatLikeDispatchOp
util.func private @splatLikeDispatchOp() -> (tensor<1xi32>, tensor<1xi32>) {
  %splat_value = arith.constant 123 : i32
  //      CHECK: %[[DISPATCH_A:.+]] = flow.dispatch
  %dispatch = flow.dispatch @some::@splat_like(%splat_value) : (i32) -> tensor<1xi32>
  // CHECK-NEXT: %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[DISPATCH_A]]
  %transfer_a = flow.tensor.transfer %dispatch : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK-NEXT: %[[DISPATCH_B:.+]] = flow.dispatch
  // CHECK-NEXT: %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[DISPATCH_B]]
  %transfer_b = flow.tensor.transfer %dispatch : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK-NEXT: util.return %[[TRANSFER_A]], %[[TRANSFER_B]]
  util.return %transfer_a, %transfer_b : tensor<1xi32>, tensor<1xi32>
}

// -----

// Tests that tied ops that do not have affinity of their own are cloned in
// order to fully flatten out the dependency chain. Here we expect both the
// splat dispatch and the reshape to be cloned for each target. Note that
// reshapes move past transfers because the pass applies canonicalization
// patterns.

// CHECK-LABEL: @reshapedDispatchOp
util.func private @reshapedDispatchOp() -> (tensor<1x4xi32>, tensor<1x4xi32>) {
  %splat_value = arith.constant 123 : i32
  //      CHECK: %[[DISPATCH_A:.+]] = flow.dispatch
  // CHECK-NEXT: %[[RESHAPE_A:.+]] = flow.tensor.reshape %[[DISPATCH_A]]
  // CHECK-NEXT: %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[RESHAPE_A]]
  %splat = flow.dispatch @some::@splat_like(%splat_value) : (i32) -> tensor<4x1xi32>
  %reshape = flow.tensor.reshape %splat : tensor<4x1xi32> -> tensor<1x4xi32>
  %transfer_a = flow.tensor.transfer %reshape : tensor<1x4xi32> to #hal.device.promise<@dev_a>
  // CHECK-NEXT: %[[DISPATCH_B:.+]] = flow.dispatch
  // CHECK-NEXT: %[[RESHAPE_B:.+]] = flow.tensor.reshape %[[DISPATCH_B]]
  // CHECK-NEXT: %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[RESHAPE_B]]
  %transfer_b = flow.tensor.transfer %reshape : tensor<1x4xi32> to #hal.device.promise<@dev_b>
  // CHECK-NEXT: util.return %[[TRANSFER_A]], %[[TRANSFER_B]]
  util.return %transfer_a, %transfer_b : tensor<1x4xi32>, tensor<1x4xi32>
}

// -----

// Tests that a dispatch with multiple results is cloned even if some results
// are not used on all devices - they'll end up unused and that's (probably) ok.
// At this level of the stack there's not much we can do to elide the unneeded
// results. We should only produce one clone per affinity instead of one per use
// but the analysis required is expensive.

// CHECK-LABEL: @uniformMultiResultDispatchOp
util.func private @uniformMultiResultDispatchOp() -> (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) {
  %dispatch_value = arith.constant 123 : i32
  //      CHECK: %[[DISPATCH_A:.+]]:2 = flow.dispatch
  %dispatch:2 = flow.dispatch @some::@multi_splat_like(%dispatch_value) : (i32) -> (tensor<1xi32>, tensor<1xi32>)
  // CHECK-NEXT: %[[TRANSFER0_A:.+]] = flow.tensor.transfer %[[DISPATCH_A]]#0
  %transfer0_a = flow.tensor.transfer %dispatch#0 : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK-NEXT: %[[DISPATCH0_B:.+]]:2 = flow.dispatch
  // CHECK-NEXT: %[[TRANSFER0_B:.+]] = flow.tensor.transfer %[[DISPATCH0_B]]#0
  %transfer0_b = flow.tensor.transfer %dispatch#0 : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK-NEXT: %[[DISPATCH1_B:.+]]:2 = flow.dispatch
  // CHECK-NEXT: %[[TRANSFER1_B:.+]] = flow.tensor.transfer %[[DISPATCH1_B]]#1
  %transfer1_b = flow.tensor.transfer %dispatch#1 : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK-NEXT: util.return %[[TRANSFER0_A]], %[[TRANSFER0_B]], %[[TRANSFER1_B]]
  util.return %transfer0_a, %transfer0_b, %transfer1_b : tensor<1xi32>, tensor<1xi32>, tensor<1xi32>
}

// -----

// Tests that takes the same value multiple times does not clone the producers
// more than once.

// CHECK-LABEL: @multipleUses
util.func private @multipleUses() -> (tensor<1xi32>, tensor<1xi32>) {
  %splat_value = arith.constant 123 : i32
  //      CHECK: %[[SPLAT_A:.+]] = flow.tensor.splat
  %splat = flow.tensor.splat %splat_value : tensor<1xi32>
  // CHECK-NEXT: %[[DISPATCH_A:.+]] = flow.dispatch @ex::@a(%[[SPLAT_A]], %[[SPLAT_A]])
  %dispatch = flow.dispatch @ex::@a(%splat, %splat) {stream.affinity = #hal.device.promise<@dev_a>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[DISPATCH_A]]
  %transfer_a = flow.tensor.transfer %dispatch : tensor<1xi32> to #hal.device.promise<@dev_a>
  // CHECK-NEXT: %[[SPLAT_B:.+]] = flow.tensor.splat
  // CHECK-NEXT: %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[SPLAT_B]]
  %transfer_b = flow.tensor.transfer %splat : tensor<1xi32> to #hal.device.promise<@dev_b>
  // CHECK-NEXT: util.return %[[TRANSFER_A]], %[[TRANSFER_B]]
  util.return %transfer_a, %transfer_b : tensor<1xi32>, tensor<1xi32>
}
