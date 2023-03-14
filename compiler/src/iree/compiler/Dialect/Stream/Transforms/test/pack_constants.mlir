// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-stream-pack-constants))' %s | FileCheck %s

// This is a high level test of the structure emitted by the pass.
// Subsequent tests focus on individual components.

// Constants get packed into composite attributes.
//      CHECK: #composite_of_192b = #util.composite<192xi8, [
// CHECK-NEXT:   dense<100> : tensor<1xi32>,
// CHECK-NEXT:   dense<0> : vector<60xi8>,
// CHECK-NEXT:   dense<[101, 102]> : tensor<2xi32>,
// CHECK-NEXT:   dense<0> : vector<56xi8>,
// CHECK-NEXT:   dense_resource<__elided__> : tensor<3x4xf32>,
// CHECK-NEXT:   dense<0> : vector<16xi8>,
// CHECK-NEXT: ]>

// CHECK-LABEL: @resourceConstants
func.func @resourceConstants() -> (!stream.resource<constant>, !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c48 = arith.constant 48 : index

  // Fetch the read-only host data containing the constants.
  // CHECK: %[[RODATA:.+]] = util.buffer.constant {alignment = 64 : index} : !util.buffer = #composite_of_192b
  %0:4 = stream.resource.constants :
    !stream.resource<constant>{%c4} = dense<100> : tensor<1xi32>,
    !stream.resource<constant>{%c8} = dense<[101, 102]> : tensor<2xi32>,
    !stream.resource<constant>{%c48} = dense_resource<__elided__> : tensor<3x4xf32>
    => !stream.timepoint

  // Try first to map the memory directly into a usable resource. If this
  // succeeds we are done and can avoid allocation/complete immediately.
  // CHECK: %[[DID_MAP:.+]], %[[TRY_MAP:.+]] = stream.resource.try_map %[[RODATA]][%c0] :
  // CHECK-SAME: !util.buffer -> i1, !stream.resource<constant>{%c192}
  //      CHECK: %[[IF:.+]]:2 = scf.if %[[DID_MAP]] -> (!stream.resource<constant>, !stream.timepoint) {
  // CHECK-NEXT:   %[[IMMEDIATE:.+]] = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NEXT:   scf.yield %[[TRY_MAP]], %[[IMMEDIATE]]
  // CHECK-NEXT: } else {

  // If the mapping fails we need to perform an upload via a staging buffer.
  // CHECK: %[[STAGING:.+]] = stream.resource.map %[[RODATA]][%c0] : !util.buffer -> !stream.resource<staging>{%c192}
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<constant>{%c192}
  // CHECK: %[[EXEC_TIMEPOINT:.+]] = stream.cmd.execute
  // CHECK-SAME: with(%[[STAGING]] as %[[STAGING_CAPTURE:.+]]: !stream.resource<staging>{%c192},
  // CHECK-SAME:      %[[ALLOC]] as %[[ALLOC_CAPTURE:.+]]: !stream.resource<constant>{%c192}) {
  // CHECK:   stream.cmd.copy %[[STAGING_CAPTURE]][%c0], %[[ALLOC_CAPTURE]][%c0], %c192 : !stream.resource<staging>{%c192} -> !stream.resource<constant>{%c192}
  // CHECK: } => !stream.timepoint
  // CHECK: scf.yield %[[ALLOC]], %[[EXEC_TIMEPOINT]]

  // Get subviews pointing to the subresources within the packed resource.
  // CHECK: %[[RES0:.+]] = stream.resource.subview %[[IF]]#0[%c0] : !stream.resource<constant>{%c192} -> !stream.resource<constant>{%c4}
  // CHECK: %[[RES1:.+]] = stream.resource.subview %[[IF]]#0[%c64] : !stream.resource<constant>{%c192} -> !stream.resource<constant>{%c8}
  // CHECK: %[[RES2:.+]] = stream.resource.subview %[[IF]]#0[%c128] : !stream.resource<constant>{%c192} -> !stream.resource<constant>{%c48}

  // CHECK: return %[[RES0]], %[[RES1]], %[[RES2]], %[[IF]]#1
  return %0#0, %0#1, %0#2, %0#3 : !stream.resource<constant>, !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

// Tests that if we exceed the maximum allowed allocation size the constants get
// partitioned into multiple buckets each within the required bounds. This test
// produces the same logic as above but doubled.

#splitResourceConstantsConfig = #stream.resource_config<{
  max_allocation_size = 16,
  min_buffer_offset_alignment = 16,
  max_buffer_range = 1073741824,
  min_buffer_range_alignment = 16,
  index_bits = 32
}>

// CHECK: #composite_of_16b = #util.composite<16xi8, [
// CHECK:     dense<100> : tensor<1xi32>,
// CHECK:     dense<0> : vector<12xi8>,
// CHECK: ]>
// CHECK: #composite_of_16b1 = #util.composite<16xi8, [
// CHECK:     dense<[101, 102]> : tensor<2xi32>,
// CHECK:     dense<0> : vector<8xi8>,
// CHECK: ]>

// CHECK-LABEL: @splitResourceConstants
func.func @splitResourceConstants() -> (!stream.resource<constant>, !stream.resource<constant>, !stream.timepoint)
    attributes {stream.resources = #splitResourceConstantsConfig} {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // CHECK: %[[RODATA0:.+]] = util.buffer.constant {alignment = 16 : index} : !util.buffer = #composite_of_16b
  // CHECK: %[[RODATA1:.+]] = util.buffer.constant {alignment = 16 : index} : !util.buffer = #composite_of_16b1
  %0:3 = stream.resource.constants :
    !stream.resource<constant>{%c4} = dense<100> : tensor<1xi32>,
    !stream.resource<constant>{%c8} = dense<[101, 102]> : tensor<2xi32>
    => !stream.timepoint

  // NOTE: we fall back for all even if only one fails; this is just for
  // simplicity in the pass today but we could only fallback for the ones that
  // failed if we wanted.
  // CHECK: %[[DID_MAP0:.+]], %[[TRY_MAP0:.+]] = stream.resource.try_map %[[RODATA0]][%c0] : !util.buffer -> i1, !stream.resource<constant>{%c16}
  // CHECK: %[[DID_MAP1:.+]], %[[TRY_MAP1:.+]] = stream.resource.try_map %[[RODATA1]][%c0] : !util.buffer -> i1, !stream.resource<constant>{%c16}
  // CHECK: %[[BOTH_MAPPED:.+]] = arith.andi %[[DID_MAP0]], %[[DID_MAP1]] : i1
  // CHECK: %[[IF:.+]]:3 = scf.if %[[BOTH_MAPPED]]
  // CHECK:    scf.yield %[[TRY_MAP0]], %[[TRY_MAP1]]
  // CHECK: } else {

  // CHECK: stream.resource.map %[[RODATA0]]
  // CHECK: stream.resource.alloc
  // CHECK: stream.resource.map %[[RODATA1]]
  // CHECK: stream.resource.alloc
  // CHECK: stream.cmd.execute
  // CHECK-NEXT: stream.cmd.copy
  // CHECK-NEXT: stream.cmd.copy

  // CHECK: %[[RES0:.+]] = stream.resource.subview %[[IF]]#0[%c0] : !stream.resource<constant>{%c16} -> !stream.resource<constant>{%c4}
  // CHECK: %[[RES1:.+]] = stream.resource.subview %[[IF]]#1[%c0] : !stream.resource<constant>{%c16} -> !stream.resource<constant>{%c8}

  // CHECK: return %[[RES0]], %[[RES1]], %[[IF]]#2
  return %0#0, %0#1, %0#2 : !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

// Tests that resources with varying lifetimes get split and processed
// independently. This allows for fast-path constants while allowing variable
// initializers to go the normal staging route. We expect to end up with two
// constant storage buffers, two uploads, and a join for the final timepoint.

// CHECK-LABEL: @mixedResourceConstants
func.func @mixedResourceConstants() -> (!stream.resource<constant>, !stream.resource<variable>, !stream.timepoint) {
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index

  // CHECK: %[[CONSTANT_HOST:.+]] = util.buffer.constant {{.+}} = #composite_of_1024b
  // CHECK: %[[CONSTANT_IF:.+]]:2 = scf.if {{.+}} -> (!stream.resource<constant>, !stream.timepoint)
  // CHECK: %[[CONSTANT_VIEW:.+]] = stream.resource.subview %[[CONSTANT_IF]]#0

  // CHECK: %[[VARIABLE_HOST:.+]] = util.buffer.constant {{.+}} = #composite_of_64b
  // CHECK: %[[VARIABLE_BUFFER:.+]] = stream.resource.alloc {{.+}} : !stream.resource<variable>{%c64}
  // CHECK: %[[VARIABLE_EXEC:.+]] = stream.cmd.execute {{.+}} %[[VARIABLE_BUFFER]]
  // CHECK: %[[VARIABLE_VIEW:.+]] = stream.resource.subview %[[VARIABLE_BUFFER]]

  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[CONSTANT_IF]]#1, %[[VARIABLE_EXEC]])
  %0:3 = stream.resource.constants :
    !stream.resource<constant>{%c1024} = dense<100> : tensor<256xi32>,
    !stream.resource<variable>{%c8} = dense<[101, 102]> : tensor<2xi32>
    => !stream.timepoint

  // CHECK: return %[[CONSTANT_VIEW]], %[[VARIABLE_VIEW]], %[[JOIN]]
  return %0#0, %0#1, %0#2 : !stream.resource<constant>, !stream.resource<variable>, !stream.timepoint
}
