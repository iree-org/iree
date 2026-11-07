// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-stream-schedule-allocation{bound-transients=true})" %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-stream-schedule-allocation %s | FileCheck %s --check-prefix=DEFAULT
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-stream-schedule-allocation{bound-transients=true},iree-stream-layout-slices)" %s | FileCheck %s --check-prefix=LAYOUT

// A transient whose size has a proven upper bound packs at that bound, aligned
// to the 64-byte default range alignment; the fills keep the exact sizes.
// CHECK-LABEL: @bounded_transient
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[OTHER:.+]]: index)
// CHECK: %[[EXACT:.+]] = util.assume.int %[[SIZE]]<umax = 256> : index
// CHECK: %[[OTHER_EXACT:.+]] = util.assume.int %[[OTHER]]<umin = 16, umax = 96> : index
// CHECK-DAG: %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
// CHECK: %[[SLICES:.+]]:3 = stream.resource.pack slices({
// CHECK-NEXT:   [0, 0] = %[[C256]],
// CHECK-NEXT:   [1, 1] = %[[C128]]
// CHECK-NEXT: })
// CHECK: stream.cmd.fill %c254_i32, %{{.+}}[%[[SLICES]]#1 for %[[EXACT]]]
// CHECK: stream.cmd.fill %c255_i32, %{{.+}}[%[[SLICES]]#2 for %[[OTHER_EXACT]]]
// With every slice bounded the slab is static.
// LAYOUT-LABEL: @bounded_transient
// LAYOUT: stream.resource.alloca uninitialized : !stream.resource<transient>{%c256{{(_[0-9]+)?}}}
// DEFAULT-LABEL: @bounded_transient
// DEFAULT: stream.resource.pack slices({
// DEFAULT-NEXT:   [0, 0] = %{{.+}},
// DEFAULT-NOT: arith.constant 256
util.func public @bounded_transient(%size: index, %other: index) -> !stream.timepoint {
  %c254_i32 = arith.constant 254 : i32
  %c255_i32 = arith.constant 255 : i32
  %exact = util.assume.int %size<umax = 256> : index
  %other_exact = util.assume.int %other<umin = 16, umax = 96> : index
  %timepoint = stream.async.execute with() {
    %0 = stream.async.splat %c254_i32 : i32 -> !stream.resource<transient>{%exact}
    %1 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%other_exact}
    stream.yield
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @unbounded_transient
// CHECK-SAME: (%[[SIZE:.+]]: index)
// CHECK: stream.resource.pack slices({
// CHECK-NEXT:   [0, 0] = %[[SIZE]]
// CHECK-NEXT: })
util.func public @unbounded_transient(%size: index) -> !stream.timepoint {
  %c254_i32 = arith.constant 254 : i32
  %timepoint = stream.async.execute with() {
    %0 = stream.async.splat %c254_i32 : i32 -> !stream.resource<transient>{%size}
    stream.yield
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// Bounded and unbounded transients share one slab; the bounded one is constant.
// LAYOUT-LABEL: @mixed_transients
// LAYOUT: stream.resource.alloca uninitialized : !stream.resource<transient>{%{{[0-9]+}}}
// CHECK-LABEL: @mixed_transients
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[OTHER:.+]]: index)
// CHECK: %[[C64:.+]] = arith.constant 64 : index
// CHECK: stream.resource.pack slices({
// CHECK-NEXT:   [0, 0] = %[[C64]],
// CHECK-NEXT:   [1, 1] = %[[OTHER]]
// CHECK-NEXT: })
util.func public @mixed_transients(%size: index, %other: index) -> !stream.timepoint {
  %c254_i32 = arith.constant 254 : i32
  %c255_i32 = arith.constant 255 : i32
  %exact = util.assume.int %size<umax = 64> : index
  %timepoint = stream.async.execute with() {
    %0 = stream.async.splat %c254_i32 : i32 -> !stream.resource<transient>{%exact}
    %1 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%other}
    stream.yield
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// Capacities beyond the largest allocation keep the exact sizes.
// CHECK-LABEL: @oversized_bound
// CHECK-SAME: (%[[SIZE:.+]]: index)
// CHECK: %[[EXACT:.+]] = util.assume.int
// CHECK: stream.resource.pack slices({
// CHECK-NEXT:   [0, 0] = %[[EXACT]]
// CHECK-NEXT: })
util.func public @oversized_bound(%size: index) -> !stream.timepoint {
  %c254_i32 = arith.constant 254 : i32
  %exact = util.assume.int %size<umax = 4611686018427387904> : index
  %timepoint = stream.async.execute with() {
    %0 = stream.async.splat %c254_i32 : i32 -> !stream.resource<transient>{%exact}
    stream.yield
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// A transient that is always empty packs at 0.
// CHECK-LABEL: @zero_bound
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: stream.resource.pack slices({
// CHECK-NEXT:   [0, 0] = %[[C0]]
// CHECK-NEXT: })
util.func public @zero_bound(%size: index) -> !stream.timepoint {
  %c254_i32 = arith.constant 254 : i32
  %exact = util.assume.int %size<umax = 0> : index
  %timepoint = stream.async.execute with() {
    %0 = stream.async.splat %c254_i32 : i32 -> !stream.resource<transient>{%exact}
    stream.yield
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}
