// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_60 --pass-pipeline='builtin.module(func.func(iree-codegen-expand-gpu-ops))' %s | FileCheck %s

// Basic scan with cluster_size=4, stride=1 (default), f32 addf.
// Expects 2 shuffle steps (offsets 1, 2) with predicates and selects.
// When stride=1, the divui-by-1 is folded away, so lanePos = remui(laneId, 4).

#translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [32, 1, 1] subgroup_size = 32>
func.func @scan_add_f32(%x: f32) -> f32 attributes {translation_info = #translation_info} {
  %scan = iree_gpu.subgroup_scan(%x) cluster(size = 4) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    iree_gpu.yield %add : f32
  } : f32
  return %scan : f32
}

// CHECK-LABEL: func.func @scan_add_f32
// CHECK-SAME:    (%[[X:.+]]: f32)
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : i32
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : i32
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : i32
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : i32
// CHECK:         %[[LANE:.+]] = gpu.lane_id
// CHECK:         %[[LANE_I32:.+]] = arith.index_cast %[[LANE]] : index to i32
// CHECK:         %[[POS:.+]] = arith.remui %[[LANE_I32]], %[[C4]]
//
// Step 0: offset=1, threshold=1
// CHECK:         %[[SRC0:.+]] = arith.subi %[[LANE_I32]], %[[C1]]
// CHECK:         %[[SHUF0:.+]], %{{.+}} = gpu.shuffle idx %[[X]], %[[SRC0]], %[[C32]]
// CHECK:         %[[ADD0:.+]] = arith.addf %[[SHUF0]], %[[X]]
// CHECK:         %[[PRED0:.+]] = arith.cmpi uge, %[[POS]], %[[C1]]
// CHECK:         %[[SEL0:.+]] = arith.select %[[PRED0]], %[[ADD0]], %[[X]]
//
// Step 1: offset=2, threshold=2
// CHECK:         %[[SRC1:.+]] = arith.subi %[[LANE_I32]], %[[C2]]
// CHECK:         %[[SHUF1:.+]], %{{.+}} = gpu.shuffle idx %[[SEL0]], %[[SRC1]], %[[C32]]
// CHECK:         %[[ADD1:.+]] = arith.addf %[[SHUF1]], %[[SEL0]]
// CHECK:         %[[PRED1:.+]] = arith.cmpi uge, %[[POS]], %[[C2]]
// CHECK:         %[[SEL1:.+]] = arith.select %[[PRED1]], %[[ADD1]], %[[SEL0]]
//
// CHECK:         return %[[SEL1]]

// -----

// Clustered scan with cluster_size=4, stride=16.
// Expects shuffle offsets of 16, 32. lanePos = (laneId / 16) % 4.

#translation_info2 = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>
func.func @scan_add_f32_stride16(%x: f32) -> f32 attributes {translation_info = #translation_info2} {
  %scan = iree_gpu.subgroup_scan(%x) cluster(size = 4, stride = 16) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    iree_gpu.yield %add : f32
  } : f32
  return %scan : f32
}

// CHECK-LABEL: func.func @scan_add_f32_stride16
// CHECK-SAME:    (%[[X:.+]]: f32)
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : i32
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : i32
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : i32
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : i32
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : i32
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : i32
// CHECK:         %[[LANE:.+]] = gpu.lane_id
// CHECK:         %[[LANE_I32:.+]] = arith.index_cast %[[LANE]] : index to i32
// CHECK:         %[[DIV:.+]] = arith.divui %[[LANE_I32]], %[[C16]]
// CHECK:         %[[POS:.+]] = arith.remui %[[DIV]], %[[C4]]
//
// Step 0: offset=16
// CHECK:         %[[SRC0:.+]] = arith.subi %[[LANE_I32]], %[[C16]]
// CHECK:         %[[SHUF0:.+]], %{{.+}} = gpu.shuffle idx %[[X]], %[[SRC0]], %[[C64]]
// CHECK:         %[[ADD0:.+]] = arith.addf %[[SHUF0]], %[[X]]
// CHECK:         %[[PRED0:.+]] = arith.cmpi uge, %[[POS]], %[[C1]]
// CHECK:         %[[SEL0:.+]] = arith.select %[[PRED0]], %[[ADD0]], %[[X]]
//
// Step 1: offset=32
// CHECK:         %[[SRC1:.+]] = arith.subi %[[LANE_I32]], %[[C32]]
// CHECK:         %[[SHUF1:.+]], %{{.+}} = gpu.shuffle idx %[[SEL0]], %[[SRC1]], %[[C64]]
// CHECK:         %[[ADD1:.+]] = arith.addf %[[SHUF1]], %[[SEL0]]
// CHECK:         %[[PRED1:.+]] = arith.cmpi uge, %[[POS]], %[[C2]]
// CHECK:         %[[SEL1:.+]] = arith.select %[[PRED1]], %[[ADD1]], %[[SEL0]]
//
// CHECK:         return %[[SEL1]]

// -----

// i32 scan: no bitcast needed.

#translation_info3 = #iree_codegen.translation_info<pipeline = None workgroup_size = [32, 1, 1] subgroup_size = 32>
func.func @scan_add_i32(%x: i32) -> i32 attributes {translation_info = #translation_info3} {
  %scan = iree_gpu.subgroup_scan(%x) cluster(size = 4) {
  ^bb0(%lhs: i32, %rhs: i32):
    %add = arith.addi %lhs, %rhs : i32
    iree_gpu.yield %add : i32
  } : i32
  return %scan : i32
}

// CHECK-LABEL: func.func @scan_add_i32
// No bitcast or extui/trunci for i32.
// CHECK-NOT:     arith.bitcast
// CHECK-NOT:     arith.extui
// CHECK-NOT:     arith.trunci
// CHECK:         gpu.shuffle idx
// CHECK:         arith.addi
// CHECK:         arith.select
// CHECK:         gpu.shuffle idx
// CHECK:         arith.addi
// CHECK:         arith.select

// -----

// f16 scan: requires bitcast to i32 for shuffle.

#translation_info4 = #iree_codegen.translation_info<pipeline = None workgroup_size = [32, 1, 1] subgroup_size = 32>
func.func @scan_add_f16(%x: f16) -> f16 attributes {translation_info = #translation_info4} {
  %scan = iree_gpu.subgroup_scan(%x) cluster(size = 4) {
  ^bb0(%lhs: f16, %rhs: f16):
    %add = arith.addf %lhs, %rhs : f16
    iree_gpu.yield %add : f16
  } : f16
  return %scan : f16
}

// CHECK-LABEL: func.func @scan_add_f16
// Step 0: pack to i32, shuffle, unpack
// CHECK:         arith.bitcast %{{.+}} : f16 to i16
// CHECK:         arith.extui %{{.+}} : i16 to i32
// CHECK:         gpu.shuffle idx
// CHECK:         arith.trunci %{{.+}} : i32 to i16
// CHECK:         arith.bitcast %{{.+}} : i16 to f16
// CHECK:         arith.addf
// CHECK:         arith.select
// Step 1: pack to i32, shuffle, unpack
// CHECK:         arith.bitcast %{{.+}} : f16 to i16
// CHECK:         arith.extui %{{.+}} : i16 to i32
// CHECK:         gpu.shuffle idx
// CHECK:         arith.trunci %{{.+}} : i32 to i16
// CHECK:         arith.bitcast %{{.+}} : i16 to f16
// CHECK:         arith.addf
// CHECK:         arith.select

// -----

// Full subgroup scan (no cluster clause) with subgroup_size=32.
// Expects 5 shuffle steps (offsets 1, 2, 4, 8, 16).

#translation_info5 = #iree_codegen.translation_info<pipeline = None workgroup_size = [32, 1, 1] subgroup_size = 32>
func.func @scan_full_subgroup(%x: f32) -> f32 attributes {translation_info = #translation_info5} {
  %scan = iree_gpu.subgroup_scan(%x) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    iree_gpu.yield %add : f32
  } : f32
  return %scan : f32
}

// CHECK-LABEL: func.func @scan_full_subgroup
// 5 steps for subgroup_size=32: offsets 1, 2, 4, 8, 16
// CHECK:         gpu.shuffle idx
// CHECK:         arith.select
// CHECK:         gpu.shuffle idx
// CHECK:         arith.select
// CHECK:         gpu.shuffle idx
// CHECK:         arith.select
// CHECK:         gpu.shuffle idx
// CHECK:         arith.select
// CHECK:         gpu.shuffle idx
// CHECK:         arith.select
// CHECK-NOT:     gpu.shuffle
