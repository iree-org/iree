// RUN: iree-opt --iree-stream-verify-affinities --split-input-file %s --verify-diagnostics | FileCheck %s

// Tests that affinities on ops are checked.

// CHECK-LABEL: @affinityOnOp
util.func public @affinityOnOp(%size: index) {
  // CHECK: stream.async.alloca
  %0 = stream.async.alloca on(#hal.device.promise<@device>) : !stream.resource<transient>{%size}
  util.return
}

// -----

// Tests that affinities on ancestor ops are allowed.

// CHECK-LABEL: @affinityOnAncestorOp
util.func public @affinityOnAncestorOp(%size: index) attributes {
  stream.affinity = #hal.device.promise<@device>
} {
  // CHECK: stream.async.alloca
  %0 = stream.async.alloca : !stream.resource<transient>{%size}
  util.return
}

// -----

// Tests that ops with no affinities fail.

util.func public @missingAffinity(%size: index) {
  // expected-error @+1 {{does not have an affinity assigned}}
  %0 = stream.async.alloca : !stream.resource<transient>{%size}
  util.return
}
