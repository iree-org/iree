// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-stream-pack-constants))' %s | FileCheck %s

// Tests that PackConstants correctly materializes util.buffer.constant ops for
// parameter scope and key strings when converting NamedParameterAttr values to
// cmd.parameter.load / cmd.parameter.gather ops.

// Verifies a single named parameter with scope produces util.buffer.constant
// ops for both scope and key, passed to stream.cmd.parameter.load.

// CHECK-LABEL: @parameterLoadScoped
util.func public @parameterLoadScoped() -> (!stream.resource<constant>, !stream.timepoint) {
  %c40 = arith.constant 40 : index
  // CHECK-DAG: %[[IMMEDIATE:.+]] = stream.timepoint.immediate
  // CHECK-DAG: %[[KEY:.+]] = util.buffer.constant : !util.buffer = "weights"
  // CHECK-DAG: %[[SCOPE:.+]] = util.buffer.constant : !util.buffer = "model"
  // CHECK: stream.cmd.parameter.load await(%[[IMMEDIATE]]) => {
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY]]
  // CHECK-SAME:     : !stream.resource<constant>
  // CHECK-NEXT: } =>
  %0:2 = stream.resource.constants :
    !stream.resource<constant>{%c40} = #stream.parameter.named<"model"::"weights"> : tensor<10xf32>
    => !stream.timepoint
  util.return %0#0, %0#1 : !stream.resource<constant>, !stream.timepoint
}

// -----

// Verifies that multiple parameters from the same scope are batched into a
// single stream.cmd.parameter.load, sharing one scope buffer constant.

// CHECK-LABEL: @parameterLoadBatchSameScope
util.func public @parameterLoadBatchSameScope() -> (!stream.resource<constant>, !stream.resource<constant>, !stream.timepoint) {
  %c40 = arith.constant 40 : index
  %c80 = arith.constant 80 : index
  // CHECK-DAG: %[[KEY_W:.+]] = util.buffer.constant : !util.buffer = "weights"
  // CHECK-DAG: %[[KEY_B:.+]] = util.buffer.constant : !util.buffer = "bias"
  // CHECK-DAG: %[[SCOPE:.+]] = util.buffer.constant : !util.buffer = "model"
  // CHECK: stream.cmd.parameter.load
  // CHECK-SAME: {
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY_W]]
  // CHECK-SAME:     : !stream.resource<constant>
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY_B]]
  // CHECK-SAME:     : !stream.resource<constant>
  // CHECK-NEXT: } =>
  %0:3 = stream.resource.constants :
    !stream.resource<constant>{%c40} = #stream.parameter.named<"model"::"weights"> : tensor<10xf32>,
    !stream.resource<constant>{%c80} = #stream.parameter.named<"model"::"bias"> : tensor<20xf32>
    => !stream.timepoint
  util.return %0#0, %0#1, %0#2 : !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

// Verifies that a parameter without a scope produces no scope buffer constant
// and the cmd.parameter.load entry has only the key (no :: separator).

// CHECK-LABEL: @parameterLoadNoScope
util.func public @parameterLoadNoScope() -> (!stream.resource<constant>, !stream.timepoint) {
  %c40 = arith.constant 40 : index
  // CHECK-DAG: %[[KEY:.+]] = util.buffer.constant : !util.buffer = "weights"
  // CHECK: stream.cmd.parameter.load
  // CHECK-SAME: {
  // CHECK-NEXT:   %[[KEY]]
  // CHECK-NOT:    ::
  // CHECK-SAME:   : !stream.resource<constant>
  // CHECK-NEXT: } =>
  %0:2 = stream.resource.constants :
    !stream.resource<constant>{%c40} = #stream.parameter.named<"weights"> : tensor<10xf32>
    => !stream.timepoint
  util.return %0#0, %0#1 : !stream.resource<constant>, !stream.timepoint
}

// -----

// Verifies that variable-lifetime parameters are packed and gathered (not
// loaded), allocating a storage buffer and using stream.cmd.parameter.gather.

// CHECK-LABEL: @parameterGatherVariables
util.func public @parameterGatherVariables() -> (!stream.resource<variable>, !stream.resource<variable>, !stream.timepoint) {
  %c40 = arith.constant 40 : index
  %c80 = arith.constant 80 : index
  // CHECK-DAG: %[[KEY_W:.+]] = util.buffer.constant : !util.buffer = "weights"
  // CHECK-DAG: %[[KEY_B:.+]] = util.buffer.constant : !util.buffer = "bias"
  // CHECK-DAG: %[[SCOPE:.+]] = util.buffer.constant : !util.buffer = "scope"
  // CHECK-DAG: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<variable>
  // CHECK: stream.cmd.parameter.gather
  // CHECK-SAME: {
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY_W]]
  // CHECK-SAME:   -> %[[ALLOC]]
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY_B]]
  // CHECK-SAME:   -> %[[ALLOC]]
  // CHECK-NEXT: } =>
  %0:3 = stream.resource.constants :
    !stream.resource<variable>{%c40} = #stream.parameter.named<"scope"::"weights"> : tensor<10xf32>,
    !stream.resource<variable>{%c80} = #stream.parameter.named<"scope"::"bias"> : tensor<20xf32>
    => !stream.timepoint
  util.return %0#0, %0#1, %0#2 : !stream.resource<variable>, !stream.resource<variable>, !stream.timepoint
}

// -----

// Verifies that variable parameters from different scopes produce separate
// gather operations (one per scope), joined with stream.timepoint.join.

// CHECK-LABEL: @parameterGatherMixedScopes
util.func public @parameterGatherMixedScopes() -> (!stream.resource<variable>, !stream.resource<variable>, !stream.timepoint) {
  %c40 = arith.constant 40 : index
  %c80 = arith.constant 80 : index
  // CHECK-DAG: %[[KEY_W:.+]] = util.buffer.constant : !util.buffer = "weights"
  // CHECK-DAG: %[[SCOPE_A:.+]] = util.buffer.constant : !util.buffer = "scope_a"
  // CHECK: stream.cmd.parameter.gather
  // CHECK-NEXT:   %[[SCOPE_A]]::%[[KEY_W]]
  // CHECK-DAG: %[[KEY_B:.+]] = util.buffer.constant : !util.buffer = "bias"
  // CHECK-DAG: %[[SCOPE_B:.+]] = util.buffer.constant : !util.buffer = "scope_b"
  // CHECK: stream.cmd.parameter.gather
  // CHECK-NEXT:   %[[SCOPE_B]]::%[[KEY_B]]
  // CHECK: stream.timepoint.join max
  %0:3 = stream.resource.constants :
    !stream.resource<variable>{%c40} = #stream.parameter.named<"scope_a"::"weights"> : tensor<10xf32>,
    !stream.resource<variable>{%c80} = #stream.parameter.named<"scope_b"::"bias"> : tensor<20xf32>
    => !stream.timepoint
  util.return %0#0, %0#1, %0#2 : !stream.resource<variable>, !stream.resource<variable>, !stream.timepoint
}
