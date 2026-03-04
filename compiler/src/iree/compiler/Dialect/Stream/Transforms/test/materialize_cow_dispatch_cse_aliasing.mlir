// Tests the dual-scatter buffer aliasing bug caused by CSE.
//
// Pattern:
//   A dispatch with non-tied resource results (zeros_kernel) is used as the
//   tied destination for two independent scatter dispatches. The flow:
//
//   1. MaterializeCopyOnWrite inserts clone(zeros_dispatch) for each scatter.
//   2. Canonicalize (PropagateCloneableOps) replaces clone(X) with fresh X
//      at each use site because preferCloneToConsumers() == true for
//      dispatches with no resource inputs.
//   3. After step 2 there are two independent zeros dispatches.
//
// FIX: AsyncDispatchOp::getEffects() reports MemoryEffects::Allocate when the
// dispatch has any non-tied resource result (a fresh allocation), preventing
// CSE from merging the two freshly rematerialized zeros dispatches.
//
// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(util.func(iree-stream-materialize-copy-on-write,canonicalize,cse))' \
// RUN:   %s | FileCheck %s
// RUN: iree-opt --split-input-file --cse %s \
// RUN:   | FileCheck %s --check-prefix=CSE

// After COW + canonicalize each scatter should have its own independent zeros
// buffer.  CSE must NOT merge them.  The output order is interleaved:
//   zeros0, scatter0, zeros1, scatter1
//
// CHECK-LABEL: @moe_scatter_aliasing
// CHECK:      %[[ZEROS0:.+]] = stream.async.dispatch @ex::@zeros_kernel
// CHECK:      stream.async.dispatch @ex::@scatter{{.*}}(%[[ZEROS0]]
// CHECK-SAME:   -> %[[ZEROS0]]
// CHECK:      %[[ZEROS1:.+]] = stream.async.dispatch @ex::@zeros_kernel
// CHECK:      stream.async.dispatch @ex::@scatter{{.*}}(%[[ZEROS1]]
// CHECK-SAME:   -> %[[ZEROS1]]
util.func private @moe_scatter_aliasing(
    %data0: !stream.resource<*>, %data1: !stream.resource<*>
) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index

  // zeros_kernel produces a non-tied resource result (fresh allocation).
  // COW inserts clone(zeros) for each tied-operand consumer, then
  // canonicalize replaces clone(X) with a fresh X.  CSE then merges
  // the two fresh dispatches (both identical) into one -> BUG.
  %zeros = stream.async.dispatch @ex::@zeros_kernel[%c1, %c1, %c1](%c64) :
      (index) -> !stream.resource<*>{%c64}

  %scatter0 = stream.async.dispatch @ex::@scatter[%c1, %c1, %c1](
      %zeros[%c0 to %c64 for %c64], %data0[%c0 to %c16 for %c16]) :
      (!stream.resource<*>{%c64}, !stream.resource<*>{%c16}) -> %zeros{%c64}

  %scatter1 = stream.async.dispatch @ex::@scatter[%c1, %c1, %c1](
      %zeros[%c0 to %c64 for %c64], %data1[%c0 to %c16 for %c16]) :
      (!stream.resource<*>{%c64}, !stream.resource<*>{%c16}) -> %zeros{%c64}

  util.return %scatter0, %scatter1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// Verify that CSE still merges identical dispatches when all results are tied
// (pure in-place transformation, no fresh allocation).
// This test uses the CSE-only RUN line (no COW/canonicalize).

// CSE-LABEL: @tied_dispatches_cse_allowed
// CSE:      %[[D:.+]] = stream.async.dispatch @ex::@transform
// CSE-NOT:  stream.async.dispatch @ex::@transform
// CSE:      util.return %[[D]], %[[D]]
util.func private @tied_dispatches_cse_allowed(
    %input: !stream.resource<*>
) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // Both dispatches are identical and produce tied results (in-place).
  // getEffects() returns no effects -> CSE merges them into one.
  %d0 = stream.async.dispatch @ex::@transform[%c1, %c1, %c1](
      %input[%c0 to %c64 for %c64]) :
      (!stream.resource<*>{%c64}) -> %input{%c64}

  %d1 = stream.async.dispatch @ex::@transform[%c1, %c1, %c1](
      %input[%c0 to %c64 for %c64]) :
      (!stream.resource<*>{%c64}) -> %input{%c64}

  util.return %d0, %d1 : !stream.resource<*>, !stream.resource<*>
}
