// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module( util.func(iree-stream-pack-constants))' %s | FileCheck %s

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
util.func public @resourceConstants() -> (!stream.resource<constant>, !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c48 = arith.constant 48 : index

  // CHECK-DAG: %[[IMMEDIATE:.+]] = stream.timepoint.immediate => !stream.timepoint

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
  //      CHECK: %[[IF:.+]]:2 = scf.if %[[DID_MAP]] -> (!stream.timepoint, !stream.resource<constant>) {
  // CHECK-NEXT:   scf.yield %[[IMMEDIATE]], %[[TRY_MAP]]
  // CHECK-NEXT: } else {

  // If the mapping fails we need to perform an upload via a staging buffer.
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<constant>{%c192}
  // CHECK: %[[FILE:.+]] = stream.file.constant %[[RODATA]][%c0 for %c192] : !util.buffer{%c192} -> !stream.file
  // CHECK: %[[READ_TIMEPOINT:.+]] = stream.file.read await(%[[IMMEDIATE]]) => %[[FILE]][%c0_i64], %[[ALLOC]][%c0], %c192 : !stream.file -> !stream.resource<constant>{%c192} => !stream.timepoint
  // CHECK: scf.yield %[[READ_TIMEPOINT]], %[[ALLOC]]

  // Get subviews pointing to the subresources within the packed resource.
  // CHECK: %[[RES0:.+]] = stream.resource.subview %[[IF]]#1[%c0] : !stream.resource<constant>{%c192} -> !stream.resource<constant>{%c4}
  // CHECK: %[[RES1:.+]] = stream.resource.subview %[[IF]]#1[%c64] : !stream.resource<constant>{%c192} -> !stream.resource<constant>{%c8}
  // CHECK: %[[RES2:.+]] = stream.resource.subview %[[IF]]#1[%c128] : !stream.resource<constant>{%c192} -> !stream.resource<constant>{%c48}

  // CHECK: util.return %[[RES0]], %[[RES1]], %[[RES2]], %[[IF]]#0
  util.return %0#0, %0#1, %0#2, %0#3 : !stream.resource<constant>, !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

// Tests variables which always need copies so that they can be mutated.

// CHECK: #composite_of_1088b = #util.composite<1088xi8, [
// CHECK:     dense<100> : tensor<256xi32>,
// CHECK:     dense<[101, 102]> : tensor<2xi32>,
// CHECK:     dense<0> : vector<56xi8>,
// CHECK: ]>

// CHECK-LABEL: @resourceVariables
util.func public @resourceVariables() -> (!stream.resource<variable>, !stream.resource<variable>, !stream.timepoint) {
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index

  // CHECK-DAG: %[[IMMEDIATE:.+]] = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[RODATA:.+]] = util.buffer.constant {alignment = 64 : index} : !util.buffer = #composite_of_1088b
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<variable>{%c1088}
  // CHECK: %[[FILE:.+]] = stream.file.constant %[[RODATA]][%c0 for %c1088] : !util.buffer{%c1088} -> !stream.file
  // CHECK: %[[READ_TIMEPOINT:.+]] = stream.file.read await(%[[IMMEDIATE]]) => %[[FILE]][%c0_i64], %[[ALLOC]][%c0], %c1088 : !stream.file -> !stream.resource<variable>{%c1088} => !stream.timepoint
  // CHECK: %[[RES0:.+]] = stream.resource.subview %[[ALLOC]][%c0] : !stream.resource<variable>{%c1088} -> !stream.resource<variable>{%c1024}
  // CHECK: %[[RES1:.+]] = stream.resource.subview %[[ALLOC]][%c1024] : !stream.resource<variable>{%c1088} -> !stream.resource<variable>{%c8}

  %0:3 = stream.resource.constants :
    !stream.resource<variable>{%c1024} = dense<100> : tensor<256xi32>,
    !stream.resource<variable>{%c8} = dense<[101, 102]> : tensor<2xi32>
    => !stream.timepoint

  // CHECK: util.return %[[RES0]], %[[RES1]], %[[READ_TIMEPOINT]]
  util.return %0#0, %0#1, %0#2 : !stream.resource<variable>, !stream.resource<variable>, !stream.timepoint
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
util.func public @splitResourceConstants() -> (!stream.resource<constant>, !stream.resource<constant>, !stream.timepoint)
    attributes {stream.resources = #splitResourceConstantsConfig} {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // CHECK-DAG: %[[IMMEDIATE:.+]] = stream.timepoint.immediate => !stream.timepoint

  // CHECK: %[[RODATA0:.+]] = util.buffer.constant {alignment = 16 : index} : !util.buffer = #composite_of_16b
  // CHECK: %[[DID_MAP0:.+]], %[[TRY_MAP0:.+]] = stream.resource.try_map %[[RODATA0]]
  // CHECK: %[[IF0:.+]]:2 = scf.if %[[DID_MAP0]]
  // CHECK: %[[FILE0:.+]] = stream.file.constant %[[RODATA0]]
  // CHECK: stream.file.read await(%[[IMMEDIATE]]) => %[[FILE0]]
  // CHECK: %[[RES0:.+]] = stream.resource.subview %[[IF0]]#1[%c0] : !stream.resource<constant>{%c16} -> !stream.resource<constant>{%c4}

  // CHECK: %[[RODATA1:.+]] = util.buffer.constant {alignment = 16 : index} : !util.buffer = #composite_of_16b1
  // CHECK: %[[DID_MAP1:.+]], %[[TRY_MAP1:.+]] = stream.resource.try_map %[[RODATA1]]
  // CHECK: %[[IF1:.+]]:2 = scf.if %[[DID_MAP1]]
  // CHECK: %[[FILE1:.+]] = stream.file.constant %[[RODATA1]]
  // CHECK: stream.file.read await(%[[IF0]]#0) => %[[FILE1]]
  // CHECK: %[[RES1:.+]] = stream.resource.subview %[[IF1]]#1[%c0] : !stream.resource<constant>{%c16} -> !stream.resource<constant>{%c8}

  %0:3 = stream.resource.constants :
    !stream.resource<constant>{%c4} = dense<100> : tensor<1xi32>,
    !stream.resource<constant>{%c8} = dense<[101, 102]> : tensor<2xi32>
    => !stream.timepoint

  // CHECK: util.return %[[RES0]], %[[RES1]], %[[IF1]]#0
  util.return %0#0, %0#1, %0#2 : !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}
