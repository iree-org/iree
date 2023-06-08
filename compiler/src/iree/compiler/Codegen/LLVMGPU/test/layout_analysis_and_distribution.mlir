// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule -cse -split-input-file --verify-diagnostics | FileCheck %s

builtin.module {
  func.func @matmul_dispatch_0_matmul_16x8x16() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x16xf16>
    memref.assume_alignment %1, 64 : memref<8x16xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = vector.transfer_read %0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %4 = vector.transfer_read %1[%c0, %c0], %cst_0 {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : memref<8x16xf16>, vector<8x16xf16>
    %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %cst : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    vector.transfer_write %5, %2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_dispatch_0_matmul_16x8x16() {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x16xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<8x16xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK-DAG:    %[[D3:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D4:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D6:.+]] = affine.apply #[[MAP]](%[[D3]], %[[D4]], %[[D5]])
// CHECK-DAG:    %[[D7:.+]] = affine.apply #[[MAP1]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D8:.+]] = arith.addi %[[D6]], %[[C0]] : index
// CHECK:        %[[D9:.+]] = arith.addi %[[D7]], %[[C0]] : index
// CHECK:        %[[D10:.+]] = memref.load %[[D0]][%[[D8]], %[[D9]]] : memref<16x16xf16>
// CHECK:        %[[D11:.+]] = vector.broadcast %[[D10]] : f16 to vector<1xf16>
// CHECK:        %[[D12:.+]] = vector.insert_strided_slice %[[D11]], %[[CST_0]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D13:.+]] = affine.apply #[[MAP2]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D14:.+]] = arith.addi %[[D13]], %[[C0]] : index
// CHECK:        %[[D15:.+]] = memref.load %[[D0]][%[[D8]], %[[D14]]] : memref<16x16xf16>
// CHECK:        %[[D16:.+]] = vector.broadcast %[[D15]] : f16 to vector<1xf16>
// CHECK:        %[[D17:.+]] = vector.insert_strided_slice %[[D16]], %[[D12]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D18:.+]] = affine.apply #[[MAP3]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D19:.+]] = arith.addi %[[D18]], %[[C0]] : index
// CHECK:        %[[D20:.+]] = memref.load %[[D0]][%[[D8]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D21:.+]] = vector.broadcast %[[D20]] : f16 to vector<1xf16>
// CHECK:        %[[D22:.+]] = vector.insert_strided_slice %[[D21]], %[[D17]] {offsets = [0, 0, 2, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D23:.+]] = affine.apply #[[MAP4]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D24:.+]] = arith.addi %[[D23]], %[[C0]] : index
// CHECK:        %[[D25:.+]] = memref.load %[[D0]][%[[D8]], %[[D24]]] : memref<16x16xf16>
// CHECK:        %[[D26:.+]] = vector.broadcast %[[D25]] : f16 to vector<1xf16>
// CHECK:        %[[D27:.+]] = vector.insert_strided_slice %[[D26]], %[[D22]] {offsets = [0, 0, 2, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D28:.+]] = affine.apply #[[MAP5]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D29:.+]] = arith.addi %[[D28]], %[[C0]] : index
// CHECK:        %[[D30:.+]] = memref.load %[[D0]][%[[D29]], %[[D9]]] : memref<16x16xf16>
// CHECK:        %[[D31:.+]] = vector.broadcast %[[D30]] : f16 to vector<1xf16>
// CHECK:        %[[D32:.+]] = vector.insert_strided_slice %[[D31]], %[[D27]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D33:.+]] = memref.load %[[D0]][%[[D29]], %[[D14]]] : memref<16x16xf16>
// CHECK:        %[[D34:.+]] = vector.broadcast %[[D33]] : f16 to vector<1xf16>
// CHECK:        %[[D35:.+]] = vector.insert_strided_slice %[[D34]], %[[D32]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D36:.+]] = memref.load %[[D0]][%[[D29]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D37:.+]] = vector.broadcast %[[D36]] : f16 to vector<1xf16>
// CHECK:        %[[D38:.+]] = vector.insert_strided_slice %[[D37]], %[[D35]] {offsets = [0, 0, 3, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D39:.+]] = memref.load %[[D0]][%[[D29]], %[[D24]]] : memref<16x16xf16>
// CHECK:        %[[D40:.+]] = vector.broadcast %[[D39]] : f16 to vector<1xf16>
// CHECK:        %[[D41:.+]] = vector.insert_strided_slice %[[D40]], %[[D38]] {offsets = [0, 0, 3, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D42:.+]] = affine.apply #[[MAP6]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D43:.+]] = arith.addi %[[D42]], %[[C0]] : index
// CHECK:        %[[D44:.+]] = memref.load %[[D1]][%[[D9]], %[[D43]]] : memref<8x16xf16>
// CHECK:        %[[D45:.+]] = vector.broadcast %[[D44]] : f16 to vector<1xf16>
// CHECK:        %[[D46:.+]] = vector.insert_strided_slice %[[D45]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D47:.+]] = memref.load %[[D1]][%[[D14]], %[[D43]]] : memref<8x16xf16>
// CHECK:        %[[D48:.+]] = vector.broadcast %[[D47]] : f16 to vector<1xf16>
// CHECK:        %[[D49:.+]] = vector.insert_strided_slice %[[D48]], %[[D46]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D50:.+]] = memref.load %[[D1]][%[[D19]], %[[D43]]] : memref<8x16xf16>
// CHECK:        %[[D51:.+]] = vector.broadcast %[[D50]] : f16 to vector<1xf16>
// CHECK:        %[[D52:.+]] = vector.insert_strided_slice %[[D51]], %[[D49]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D53:.+]] = memref.load %[[D1]][%[[D24]], %[[D43]]] : memref<8x16xf16>
// CHECK:        %[[D54:.+]] = vector.broadcast %[[D53]] : f16 to vector<1xf16>
// CHECK:        %[[D55:.+]] = vector.insert_strided_slice %[[D54]], %[[D52]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D56:.+]] = vector.extract %[[CST]][0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D57:.+]] = vector.extract %[[D41]][0, 0] : vector<1x1x4x2xf16>
// CHECK:        %[[D58:.+]] = vector.extract %[[D55]][0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D59:.+]] = nvgpu.mma.sync(%[[D57]], %[[D58]], %[[D56]]) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>,
// CHECK-SAME:     vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D60:.+]] = vector.insert %[[D59]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D61:.+]] = vector.extract %[[D60]][0, 0, 0, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D61]], %[[D2]][%[[D8]], %[[D9]]] : memref<16x8xf16>
// CHECK:        %[[D62:.+]] = vector.extract %[[D60]][0, 0, 0, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D62]], %[[D2]][%[[D8]], %[[D14]]] : memref<16x8xf16>
// CHECK:        %[[D63:.+]] = vector.extract %[[D60]][0, 0, 1, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D63]], %[[D2]][%[[D29]], %[[D9]]] : memref<16x8xf16>
// CHECK:        %[[D64:.+]] = vector.extract %[[D60]][0, 0, 1, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D64]], %[[D2]][%[[D29]], %[[D14]]] : memref<16x8xf16>
// CHECK:        return
// CHECK:      }

// -----

builtin.module {
  func.func @matmul_reduction() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %init = arith.constant dense<-1.000000e+04> : vector<16xf16>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x16xf16>
    memref.assume_alignment %1, 64 : memref<8x16xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = vector.transfer_read %0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %4 = vector.transfer_read %1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<8x16xf16>, vector<8x16xf16>
    %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %cst : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    %6 = vector.multi_reduction <maxf>, %5, %init [1] : vector<16x8xf16> to vector<16xf16>
    %7 = vector.broadcast %6 : vector<16xf16> to vector<8x16xf16>
    %8 = vector.transpose %7, [1, 0] : vector<8x16xf16> to vector<16x8xf16>
    vector.transfer_write %8, %2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_reduction() {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<-1.000000e+04> : vector<1x1x2x2xf16>
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x16xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<8x16xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK-DAG:    %[[D3:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D4:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D6:.+]] = affine.apply #[[MAP]](%[[D3]], %[[D4]], %[[D5]])
// CHECK-DAG:    %[[D7:.+]] = affine.apply #[[MAP1]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D8:.+]] = arith.addi %[[D6]], %[[C0]] : index
// CHECK:        %[[D9:.+]] = arith.addi %[[D7]], %[[C0]] : index
// CHECK:        %[[D10:.+]] = memref.load %[[D0]][%[[D8]], %[[D9]]] : memref<16x16xf16>
// CHECK:        %[[D11:.+]] = vector.broadcast %[[D10]] : f16 to vector<1xf16>
// CHECK:        %[[D12:.+]] = vector.insert_strided_slice %[[D11]], %[[CST_1]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D13:.+]] = affine.apply #[[MAP2]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D14:.+]] = arith.addi %[[D13]], %[[C0]] : index
// CHECK:        %[[D15:.+]] = memref.load %[[D0]][%[[D8]], %[[D14]]] : memref<16x16xf16>
// CHECK:        %[[D16:.+]] = vector.broadcast %[[D15]] : f16 to vector<1xf16>
// CHECK:        %[[D17:.+]] = vector.insert_strided_slice %[[D16]], %[[D12]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D18:.+]] = affine.apply #[[MAP3]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D19:.+]] = arith.addi %[[D18]], %[[C0]] : index
// CHECK:        %[[D20:.+]] = memref.load %[[D0]][%[[D8]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D21:.+]] = vector.broadcast %[[D20]] : f16 to vector<1xf16>
// CHECK:        %[[D22:.+]] = vector.insert_strided_slice %[[D21]], %[[D17]] {offsets = [0, 0, 2, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D23:.+]] = affine.apply #[[MAP4]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D24:.+]] = arith.addi %[[D23]], %[[C0]] : index
// CHECK:        %[[D25:.+]] = memref.load %[[D0]][%[[D8]], %[[D24]]] : memref<16x16xf16>
// CHECK:        %[[D26:.+]] = vector.broadcast %[[D25]] : f16 to vector<1xf16>
// CHECK:        %[[D27:.+]] = vector.insert_strided_slice %[[D26]], %[[D22]] {offsets = [0, 0, 2, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D28:.+]] = affine.apply #[[MAP5]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D29:.+]] = arith.addi %[[D28]], %[[C0]] : index
// CHECK:        %[[D30:.+]] = memref.load %[[D0]][%[[D29]], %[[D9]]] : memref<16x16xf16>
// CHECK:        %[[D31:.+]] = vector.broadcast %[[D30]] : f16 to vector<1xf16>
// CHECK:        %[[D32:.+]] = vector.insert_strided_slice %[[D31]], %[[D27]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D33:.+]] = memref.load %[[D0]][%[[D29]], %[[D14]]] : memref<16x16xf16>
// CHECK:        %[[D34:.+]] = vector.broadcast %[[D33]] : f16 to vector<1xf16>
// CHECK:        %[[D35:.+]] = vector.insert_strided_slice %[[D34]], %[[D32]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D36:.+]] = memref.load %[[D0]][%[[D29]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D37:.+]] = vector.broadcast %[[D36]] : f16 to vector<1xf16>
// CHECK:        %[[D38:.+]] = vector.insert_strided_slice %[[D37]], %[[D35]] {offsets = [0, 0, 3, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D39:.+]] = memref.load %[[D0]][%[[D29]], %[[D24]]] : memref<16x16xf16>
// CHECK:        %[[D40:.+]] = vector.broadcast %[[D39]] : f16 to vector<1xf16>
// CHECK:        %[[D41:.+]] = vector.insert_strided_slice %[[D40]], %[[D38]] {offsets = [0, 0, 3, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D42:.+]] = affine.apply #[[MAP6]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D43:.+]] = arith.addi %[[D42]], %[[C0]] : index
// CHECK:        %[[D44:.+]] = memref.load %[[D1]][%[[D43]], %[[D9]]] : memref<8x16xf16>
// CHECK:        %[[D45:.+]] = vector.broadcast %[[D44]] : f16 to vector<1xf16>
// CHECK:        %[[D46:.+]] = vector.insert_strided_slice %[[D45]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D47:.+]] = memref.load %[[D1]][%[[D43]], %[[D14]]] : memref<8x16xf16>
// CHECK:        %[[D48:.+]] = vector.broadcast %[[D47]] : f16 to vector<1xf16>
// CHECK:        %[[D49:.+]] = vector.insert_strided_slice %[[D48]], %[[D46]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D50:.+]] = memref.load %[[D1]][%[[D43]], %[[D19]]] : memref<8x16xf16>
// CHECK:        %[[D51:.+]] = vector.broadcast %[[D50]] : f16 to vector<1xf16>
// CHECK:        %[[D52:.+]] = vector.insert_strided_slice %[[D51]], %[[D49]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D53:.+]] = memref.load %[[D1]][%[[D43]], %[[D24]]] : memref<8x16xf16>
// CHECK:        %[[D54:.+]] = vector.broadcast %[[D53]] : f16 to vector<1xf16>
// CHECK:        %[[D55:.+]] = vector.insert_strided_slice %[[D54]], %[[D52]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D56:.+]] = vector.extract %[[CST]][0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D57:.+]] = vector.extract %[[D41]][0, 0] : vector<1x1x4x2xf16>
// CHECK:        %[[D58:.+]] = vector.extract %[[D55]][0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D59:.+]] = nvgpu.mma.sync(%[[D57]], %[[D58]], %[[D56]]) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>,
// CHECK-SAME:     vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D60:.+]] = vector.insert %[[D59]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D61:.+]] = vector.extract %[[CST_0]][0, 0, 0, 0] : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant dense<0.000000e+00> : vector<2xf16>
// CHECK:        %[[D62:.+]] = vector.extract %[[D60]][0, 0, 0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D63:.+]] = vector.insert %[[D62]], %[[CST_2]] [0] : f16 into vector<2xf16>
// CHECK:        %[[D64:.+]] = vector.extract %[[D60]][0, 0, 0, 1] : vector<1x1x2x2xf16>
// CHECK:        %[[D65:.+]] = vector.insert %[[D64]], %[[D63]] [1] : f16 into vector<2xf16>
// CHECK:        %[[D66:.+]] = vector.bitcast %[[D65]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D67:.+]] = vector.extract %[[D66]][0] : vector<1xi32>
// CHECK-DAG:    %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-DAG:    %[[C32_I32:.+]] = arith.constant 32 : i32
// CHECK:        %[[SHUFFLERESULT:.+]], %[[VALID:.+]] = gpu.shuffle  xor %[[D67]], %[[C1_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D68:.+]] = vector.broadcast %[[SHUFFLERESULT]] : i32 to vector<1xi32>
// CHECK:        %[[D69:.+]] = vector.bitcast %[[D68]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D70:.+]] = arith.maxf %[[D69]], %[[D65]] : vector<2xf16>
// CHECK:        %[[D71:.+]] = vector.bitcast %[[D70]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D72:.+]] = vector.extract %[[D71]][0] : vector<1xi32>
// CHECK-DAG:    %[[C2_I32:.+]] = arith.constant 2 : i32
// CHECK:        %[[SHUFFLERESULT_3:.+]], %[[VALID_4:.+]] = gpu.shuffle  xor %[[D72]], %[[C2_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D73:.+]] = vector.broadcast %[[SHUFFLERESULT_3]] : i32 to vector<1xi32>
// CHECK:        %[[D74:.+]] = vector.bitcast %[[D73]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D75:.+]] = arith.maxf %[[D74]], %[[D70]] : vector<2xf16>
// CHECK:        %[[D76:.+]] = vector.extract %[[D75]][0] : vector<2xf16>
// CHECK:        %[[D77:.+]] = vector.extract %[[D75]][1] : vector<2xf16>
// CHECK:        %[[D78:.+]] = arith.maxf %[[D76]], %[[D77]] : f16
// CHECK:        %[[D79:.+]] = arith.maxf %[[D78]], %[[D61]] : f16
// CHECK:        %[[D80:.+]] = vector.insert %[[D79]], %[[CST]] [0, 0, 0, 0] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D81:.+]] = vector.insert %[[D79]], %[[D80]] [0, 0, 0, 1] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D82:.+]] = vector.extract %[[CST_0]][0, 0, 1, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D83:.+]] = vector.extract %[[D60]][0, 0, 1, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D84:.+]] = vector.insert %[[D83]], %[[CST_2]] [0] : f16 into vector<2xf16>
// CHECK:        %[[D85:.+]] = vector.extract %[[D60]][0, 0, 1, 1] : vector<1x1x2x2xf16>
// CHECK:        %[[D86:.+]] = vector.insert %[[D85]], %[[D84]] [1] : f16 into vector<2xf16>
// CHECK:        %[[D87:.+]] = vector.bitcast %[[D86]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D88:.+]] = vector.extract %[[D87]][0] : vector<1xi32>
// CHECK:        %[[SHUFFLERESULT_5:.+]], %[[VALID_6:.+]] = gpu.shuffle  xor %[[D88]], %[[C1_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D89:.+]] = vector.broadcast %[[SHUFFLERESULT_5]] : i32 to vector<1xi32>
// CHECK:        %[[D90:.+]] = vector.bitcast %[[D89]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D91:.+]] = arith.maxf %[[D90]], %[[D86]] : vector<2xf16>
// CHECK:        %[[D92:.+]] = vector.bitcast %[[D91]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D93:.+]] = vector.extract %[[D92]][0] : vector<1xi32>
// CHECK:        %[[SHUFFLERESULT_7:.+]], %[[VALID_8:.+]] = gpu.shuffle  xor %[[D93]], %[[C2_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D94:.+]] = vector.broadcast %[[SHUFFLERESULT_7]] : i32 to vector<1xi32>
// CHECK:        %[[D95:.+]] = vector.bitcast %[[D94]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D96:.+]] = arith.maxf %[[D95]], %[[D91]] : vector<2xf16>
// CHECK:        %[[D97:.+]] = vector.extract %[[D96]][0] : vector<2xf16>
// CHECK:        %[[D98:.+]] = vector.extract %[[D96]][1] : vector<2xf16>
// CHECK:        %[[D99:.+]] = arith.maxf %[[D97]], %[[D98]] : f16
// CHECK:        %[[D100:.+]] = arith.maxf %[[D99]], %[[D82]] : f16
// CHECK:        %[[D101:.+]] = vector.insert %[[D100]], %[[D81]] [0, 0, 1, 0] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D102:.+]] = vector.insert %[[D100]], %[[D101]] [0, 0, 1, 1] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D103:.+]] = vector.extract %[[D102]][0, 0, 0, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D103]], %[[D2]][%[[D8]], %[[D9]]] : memref<16x8xf16>
// CHECK:        %[[D104:.+]] = vector.extract %[[D102]][0, 0, 0, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D104]], %[[D2]][%[[D8]], %[[D14]]] : memref<16x8xf16>
// CHECK:        %[[D105:.+]] = vector.extract %[[D102]][0, 0, 1, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D105]], %[[D2]][%[[D29]], %[[D9]]] : memref<16x8xf16>
// CHECK:        %[[D106:.+]] = vector.extract %[[D102]][0, 0, 1, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D106]], %[[D2]][%[[D29]], %[[D14]]] : memref<16x8xf16>
// CHECK:        return
// CHECK:      }

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0)>
#map3 = affine_map<(d0) -> (d0 * 16)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module {
  func.func @matmul_scf() {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x64xf16>
    memref.assume_alignment %0, 64 : memref<16x64xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x64xf16>
    memref.assume_alignment %1, 64 : memref<8x64xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %3, 64 : memref<16x8xf16>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %4 = affine.apply #map1()[%workgroup_id_x]
    %5 = affine.apply #map2(%c0)[%4]
    %6 = vector.transfer_read %2[%5, %c0], %cst {in_bounds = [true, true]} : memref<16x8xf16>, vector<16x8xf16>
    %7 = scf.for %arg0 = %c0 to %c4 step %c1 iter_args(%arg1 = %6) -> (vector<16x8xf16>) {
      %9 = affine.apply #map3(%arg0)
      %10 = affine.apply #map2(%c0)[%9]
      %11 = vector.transfer_read %0[%c0, %10], %cst {in_bounds = [true, true]} : memref<16x64xf16>, vector<16x16xf16>
      %13 = vector.transfer_read %1[%c0, %10], %cst {in_bounds = [true, true]} : memref<8x64xf16>, vector<8x16xf16>
      %14 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %13, %arg1 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
      scf.yield %14 : vector<16x8xf16>
    }
    %8 = affine.apply #map2(%c0)[%4]
    vector.transfer_write %7, %3[%8, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG:  #[[MAP7:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP8:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP9:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_scf() {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x64xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x64xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x64xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<8x64xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<16x8xf16>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG:    %[[D4:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_X]]]
// CHECK-DAG:    %[[D5:.+]] = affine.apply #[[MAP1]](%[[C0]])[%[[D4]]]
// CHECK-DAG:    %[[D6:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D7:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D8:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D9:.+]] = affine.apply #[[MAP2]](%[[D6]], %[[D7]], %[[D8]])
// CHECK-DAG:    %[[D10:.+]] = affine.apply #[[MAP3]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:        %[[D11:.+]] = arith.addi %[[D9]], %[[D5]] : index
// CHECK:        %[[D12:.+]] = arith.addi %[[D10]], %[[C0]] : index
// CHECK:        %[[D13:.+]] = memref.load %[[D2]][%[[D11]], %[[D12]]] : memref<16x8xf16>
// CHECK:        %[[D14:.+]] = vector.broadcast %[[D13]] : f16 to vector<1xf16>
// CHECK:        %[[D15:.+]] = vector.insert_strided_slice %[[D14]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D16:.+]] = affine.apply #[[MAP4]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:        %[[D17:.+]] = arith.addi %[[D16]], %[[C0]] : index
// CHECK:        %[[D18:.+]] = memref.load %[[D2]][%[[D11]], %[[D17]]] : memref<16x8xf16>
// CHECK:        %[[D19:.+]] = vector.broadcast %[[D18]] : f16 to vector<1xf16>
// CHECK:        %[[D20:.+]] = vector.insert_strided_slice %[[D19]], %[[D15]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D21:.+]] = affine.apply #[[MAP5]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:        %[[D22:.+]] = arith.addi %[[D21]], %[[D5]] : index
// CHECK:        %[[D23:.+]] = memref.load %[[D2]][%[[D22]], %[[D12]]] : memref<16x8xf16>
// CHECK:        %[[D24:.+]] = vector.broadcast %[[D23]] : f16 to vector<1xf16>
// CHECK:        %[[D25:.+]] = vector.insert_strided_slice %[[D24]], %[[D20]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D26:.+]] = memref.load %[[D2]][%[[D22]], %[[D17]]] : memref<16x8xf16>
// CHECK:        %[[D27:.+]] = vector.broadcast %[[D26]] : f16 to vector<1xf16>
// CHECK:        %[[D28:.+]] = vector.insert_strided_slice %[[D27]], %[[D25]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<16x8xf16>
// CHECK:        %[[D29:.+]]:2 = scf.for %[[ARG0:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG1:[a-zA-Z0-9_]+]] = %[[CST_0]], %[[ARG2:[a-zA-Z0-9_]+]] = %[[D28]]) ->
// CHECK-SAME:     (vector<16x8xf16>, vector<1x1x2x2xf16>) {
// CHECK-DAG:      %[[D34:.+]] = affine.apply #[[MAP6]](%[[ARG0]])
// CHECK-DAG:      %[[D35:.+]] = affine.apply #[[MAP1]](%[[C0]])[%[[D34]]]
// CHECK-DAG:      %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK:          %[[D36:.+]] = arith.addi %[[D9]], %[[C0]] : index
// CHECK:          %[[D37:.+]] = arith.addi %[[D10]], %[[D35]] : index
// CHECK:          %[[D38:.+]] = memref.load %[[D0]][%[[D36]], %[[D37]]] : memref<16x64xf16>
// CHECK:          %[[D39:.+]] = vector.broadcast %[[D38]] : f16 to vector<1xf16>
// CHECK:          %[[D40:.+]] = vector.insert_strided_slice %[[D39]], %[[CST_1]] {offsets = [0, 0, 0, 0], strides =
// CHECK-SAME:       [1]} : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D41:.+]] = arith.addi %[[D16]], %[[D35]] : index
// CHECK:          %[[D42:.+]] = memref.load %[[D0]][%[[D36]], %[[D41]]] : memref<16x64xf16>
// CHECK:          %[[D43:.+]] = vector.broadcast %[[D42]] : f16 to vector<1xf16>
// CHECK:          %[[D44:.+]] = vector.insert_strided_slice %[[D43]], %[[D40]] {offsets = [0, 0, 0, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D45:.+]] = affine.apply #[[MAP7]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:          %[[D46:.+]] = arith.addi %[[D45]], %[[D35]] : index
// CHECK:          %[[D47:.+]] = memref.load %[[D0]][%[[D36]], %[[D46]]] : memref<16x64xf16>
// CHECK:          %[[D48:.+]] = vector.broadcast %[[D47]] : f16 to vector<1xf16>
// CHECK:          %[[D49:.+]] = vector.insert_strided_slice %[[D48]], %[[D44]] {offsets = [0, 0, 2, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D50:.+]] = affine.apply #[[MAP8]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:          %[[D51:.+]] = arith.addi %[[D50]], %[[D35]] : index
// CHECK:          %[[D52:.+]] = memref.load %[[D0]][%[[D36]], %[[D51]]] : memref<16x64xf16>
// CHECK:          %[[D53:.+]] = vector.broadcast %[[D52]] : f16 to vector<1xf16>
// CHECK:          %[[D54:.+]] = vector.insert_strided_slice %[[D53]], %[[D49]] {offsets = [0, 0, 2, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D55:.+]] = arith.addi %[[D21]], %[[C0]] : index
// CHECK:          %[[D56:.+]] = memref.load %[[D0]][%[[D55]], %[[D37]]] : memref<16x64xf16>
// CHECK:          %[[D57:.+]] = vector.broadcast %[[D56]] : f16 to vector<1xf16>
// CHECK:          %[[D58:.+]] = vector.insert_strided_slice %[[D57]], %[[D54]] {offsets = [0, 0, 1, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D59:.+]] = memref.load %[[D0]][%[[D55]], %[[D41]]] : memref<16x64xf16>
// CHECK:          %[[D60:.+]] = vector.broadcast %[[D59]] : f16 to vector<1xf16>
// CHECK:          %[[D61:.+]] = vector.insert_strided_slice %[[D60]], %[[D58]] {offsets = [0, 0, 1, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D62:.+]] = memref.load %[[D0]][%[[D55]], %[[D46]]] : memref<16x64xf16>
// CHECK:          %[[D63:.+]] = vector.broadcast %[[D62]] : f16 to vector<1xf16>
// CHECK:          %[[D64:.+]] = vector.insert_strided_slice %[[D63]], %[[D61]] {offsets = [0, 0, 3, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D65:.+]] = memref.load %[[D0]][%[[D55]], %[[D51]]] : memref<16x64xf16>
// CHECK:          %[[D66:.+]] = vector.broadcast %[[D65]] : f16 to vector<1xf16>
// CHECK:          %[[D67:.+]] = vector.insert_strided_slice %[[D66]], %[[D64]] {offsets = [0, 0, 3, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D68:.+]] = affine.apply #[[MAP9]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:          %[[D69:.+]] = arith.addi %[[D68]], %[[C0]] : index
// CHECK:          %[[D70:.+]] = memref.load %[[D1]][%[[D69]], %[[D37]]] : memref<8x64xf16>
// CHECK:          %[[D71:.+]] = vector.broadcast %[[D70]] : f16 to vector<1xf16>
// CHECK:          %[[D72:.+]] = vector.insert_strided_slice %[[D71]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D73:.+]] = memref.load %[[D1]][%[[D69]], %[[D41]]] : memref<8x64xf16>
// CHECK:          %[[D74:.+]] = vector.broadcast %[[D73]] : f16 to vector<1xf16>
// CHECK:          %[[D75:.+]] = vector.insert_strided_slice %[[D74]], %[[D72]] {offsets = [0, 0, 0, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D76:.+]] = memref.load %[[D1]][%[[D69]], %[[D46]]] : memref<8x64xf16>
// CHECK:          %[[D77:.+]] = vector.broadcast %[[D76]] : f16 to vector<1xf16>
// CHECK:          %[[D78:.+]] = vector.insert_strided_slice %[[D77]], %[[D75]] {offsets = [0, 0, 1, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D79:.+]] = memref.load %[[D1]][%[[D69]], %[[D51]]] : memref<8x64xf16>
// CHECK:          %[[D80:.+]] = vector.broadcast %[[D79]] : f16 to vector<1xf16>
// CHECK:          %[[D81:.+]] = vector.insert_strided_slice %[[D80]], %[[D78]] {offsets = [0, 0, 1, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D82:.+]] = vector.extract %[[ARG2]][0, 0] : vector<1x1x2x2xf16>
// CHECK:          %[[D83:.+]] = vector.extract %[[D67]][0, 0] : vector<1x1x4x2xf16>
// CHECK:          %[[D84:.+]] = vector.extract %[[D81]][0, 0] : vector<1x1x2x2xf16>
// CHECK:          %[[D85:.+]] = nvgpu.mma.sync(%[[D83]], %[[D84]], %[[D82]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:          %[[D86:.+]] = vector.insert %[[D85]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:          scf.yield %[[CST_0]], %[[D86]] : vector<16x8xf16>, vector<1x1x2x2xf16>
// CHECK:        }
// CHECK:        %[[D30:.+]] = vector.extract %[[D29]]#[[D1:.+]][0, 0, 0, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D30]], %[[D3]][%[[D11]], %[[D12]]] : memref<16x8xf16>
// CHECK:        %[[D31:.+]] = vector.extract %[[D29]]#[[D1]][0, 0, 0, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D31]], %[[D3]][%[[D11]], %[[D17]]] : memref<16x8xf16>
// CHECK:        %[[D32:.+]] = vector.extract %[[D29]]#[[D1]][0, 0, 1, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D32]], %[[D3]][%[[D22]], %[[D12]]] : memref<16x8xf16>
// CHECK:        %[[D33:.+]] = vector.extract %[[D29]]#[[D1]][0, 0, 1, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D33]], %[[D3]][%[[D22]], %[[D17]]] : memref<16x8xf16>
// CHECK:        return
// CHECK:      }

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0)>
#map3 = affine_map<(d0) -> (d0 * 16)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module {
  func.func @matmul_scf() {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x64xf16>
    memref.assume_alignment %0, 64 : memref<16x64xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x64xf16>
    memref.assume_alignment %1, 64 : memref<8x64xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %3, 64 : memref<16x8xf16>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %4 = affine.apply #map1()[%workgroup_id_x]
    %5 = affine.apply #map2(%c0)[%4]
    %cst_2 = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %7 = scf.for %arg0 = %c0 to %c4 step %c1 iter_args(%arg1 = %cst_2) -> (vector<16x8xf16>) {
      %9 = affine.apply #map3(%arg0)
      %10 = affine.apply #map2(%c0)[%9]
      %11 = vector.transfer_read %0[%c0, %10], %cst {in_bounds = [true, true]} : memref<16x64xf16>, vector<16x16xf16>
      %13 = vector.transfer_read %1[%c0, %10], %cst {in_bounds = [true, true]} : memref<8x64xf16>, vector<8x16xf16>
      %14 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %13, %arg1 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
      scf.yield %14 : vector<16x8xf16>
    }
    %8 = affine.apply #map2(%c0)[%4]
    vector.transfer_write %7, %3[%8, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP7:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP8:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP9:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_scf() {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x64xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x64xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x64xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<8x64xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<16x8xf16>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG:    %[[D4:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_X]]]
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<16x8xf16>
// CHECK:        %[[D5:.+]]:2 = scf.for %[[ARG0:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG1:[a-zA-Z0-9_]+]] = %[[CST_0]], %[[ARG2:[a-zA-Z0-9_]+]] = %[[CST]]) ->
// CHECK-SAME:     (vector<16x8xf16>, vector<1x1x2x2xf16>) {
// CHECK-DAG:      %[[D22:.+]] = affine.apply #[[MAP1]](%[[ARG0]])
// CHECK-DAG:      %[[D23:.+]] = affine.apply #[[MAP2]](%[[C0]])[%[[D22]]]
// CHECK-DAG:      %[[D24:.+]] = gpu.thread_id  x
// CHECK-DAG:      %[[D25:.+]] = gpu.thread_id  y
// CHECK-DAG:      %[[D26:.+]] = gpu.thread_id  z
// CHECK-DAG:      %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D27:.+]] = affine.apply #[[MAP3]](%[[D24]], %[[D25]], %[[D26]])
// CHECK-DAG:      %[[D28:.+]] = affine.apply #[[MAP4]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D29:.+]] = arith.addi %[[D27]], %[[C0]] : index
// CHECK:          %[[D30:.+]] = arith.addi %[[D28]], %[[D23]] : index
// CHECK:          %[[D31:.+]] = memref.load %[[D0]][%[[D29]], %[[D30]]] : memref<16x64xf16>
// CHECK:          %[[D32:.+]] = vector.broadcast %[[D31]] : f16 to vector<1xf16>
// CHECK:          %[[D33:.+]] = vector.insert_strided_slice %[[D32]], %[[CST_1]] {offsets = [0, 0, 0, 0], strides =
// CHECK-SAME:       [1]} : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D34:.+]] = affine.apply #[[MAP5]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D35:.+]] = arith.addi %[[D34]], %[[D23]] : index
// CHECK:          %[[D36:.+]] = memref.load %[[D0]][%[[D29]], %[[D35]]] : memref<16x64xf16>
// CHECK:          %[[D37:.+]] = vector.broadcast %[[D36]] : f16 to vector<1xf16>
// CHECK:          %[[D38:.+]] = vector.insert_strided_slice %[[D37]], %[[D33]] {offsets = [0, 0, 0, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D39:.+]] = affine.apply #[[MAP6]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D40:.+]] = arith.addi %[[D39]], %[[D23]] : index
// CHECK:          %[[D41:.+]] = memref.load %[[D0]][%[[D29]], %[[D40]]] : memref<16x64xf16>
// CHECK:          %[[D42:.+]] = vector.broadcast %[[D41]] : f16 to vector<1xf16>
// CHECK:          %[[D43:.+]] = vector.insert_strided_slice %[[D42]], %[[D38]] {offsets = [0, 0, 2, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D44:.+]] = affine.apply #[[MAP7]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D45:.+]] = arith.addi %[[D44]], %[[D23]] : index
// CHECK:          %[[D46:.+]] = memref.load %[[D0]][%[[D29]], %[[D45]]] : memref<16x64xf16>
// CHECK:          %[[D47:.+]] = vector.broadcast %[[D46]] : f16 to vector<1xf16>
// CHECK:          %[[D48:.+]] = vector.insert_strided_slice %[[D47]], %[[D43]] {offsets = [0, 0, 2, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D49:.+]] = affine.apply #[[MAP8]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D50:.+]] = arith.addi %[[D49]], %[[C0]] : index
// CHECK:          %[[D51:.+]] = memref.load %[[D0]][%[[D50]], %[[D30]]] : memref<16x64xf16>
// CHECK:          %[[D52:.+]] = vector.broadcast %[[D51]] : f16 to vector<1xf16>
// CHECK:          %[[D53:.+]] = vector.insert_strided_slice %[[D52]], %[[D48]] {offsets = [0, 0, 1, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D54:.+]] = memref.load %[[D0]][%[[D50]], %[[D35]]] : memref<16x64xf16>
// CHECK:          %[[D55:.+]] = vector.broadcast %[[D54]] : f16 to vector<1xf16>
// CHECK:          %[[D56:.+]] = vector.insert_strided_slice %[[D55]], %[[D53]] {offsets = [0, 0, 1, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D57:.+]] = memref.load %[[D0]][%[[D50]], %[[D40]]] : memref<16x64xf16>
// CHECK:          %[[D58:.+]] = vector.broadcast %[[D57]] : f16 to vector<1xf16>
// CHECK:          %[[D59:.+]] = vector.insert_strided_slice %[[D58]], %[[D56]] {offsets = [0, 0, 3, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D60:.+]] = memref.load %[[D0]][%[[D50]], %[[D45]]] : memref<16x64xf16>
// CHECK:          %[[D61:.+]] = vector.broadcast %[[D60]] : f16 to vector<1xf16>
// CHECK:          %[[D62:.+]] = vector.insert_strided_slice %[[D61]], %[[D59]] {offsets = [0, 0, 3, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D63:.+]] = affine.apply #[[MAP9]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D64:.+]] = arith.addi %[[D63]], %[[C0]] : index
// CHECK:          %[[D65:.+]] = memref.load %[[D1]][%[[D64]], %[[D30]]] : memref<8x64xf16>
// CHECK:          %[[D66:.+]] = vector.broadcast %[[D65]] : f16 to vector<1xf16>
// CHECK:          %[[D67:.+]] = vector.insert_strided_slice %[[D66]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D68:.+]] = memref.load %[[D1]][%[[D64]], %[[D35]]] : memref<8x64xf16>
// CHECK:          %[[D69:.+]] = vector.broadcast %[[D68]] : f16 to vector<1xf16>
// CHECK:          %[[D70:.+]] = vector.insert_strided_slice %[[D69]], %[[D67]] {offsets = [0, 0, 0, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D71:.+]] = memref.load %[[D1]][%[[D64]], %[[D40]]] : memref<8x64xf16>
// CHECK:          %[[D72:.+]] = vector.broadcast %[[D71]] : f16 to vector<1xf16>
// CHECK:          %[[D73:.+]] = vector.insert_strided_slice %[[D72]], %[[D70]] {offsets = [0, 0, 1, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D74:.+]] = memref.load %[[D1]][%[[D64]], %[[D45]]] : memref<8x64xf16>
// CHECK:          %[[D75:.+]] = vector.broadcast %[[D74]] : f16 to vector<1xf16>
// CHECK:          %[[D76:.+]] = vector.insert_strided_slice %[[D75]], %[[D73]] {offsets = [0, 0, 1, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D77:.+]] = vector.extract %[[ARG2]][0, 0] : vector<1x1x2x2xf16>
// CHECK:          %[[D78:.+]] = vector.extract %[[D62]][0, 0] : vector<1x1x4x2xf16>
// CHECK:          %[[D79:.+]] = vector.extract %[[D76]][0, 0] : vector<1x1x2x2xf16>
// CHECK:          %[[D80:.+]] = nvgpu.mma.sync(%[[D78]], %[[D79]], %[[D77]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:          %[[D81:.+]] = vector.insert %[[D80]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:          scf.yield %[[CST_0]], %[[D81]] : vector<16x8xf16>, vector<1x1x2x2xf16>
// CHECK:        }
// CHECK-DAG:    %[[D6:.+]] = affine.apply #[[MAP2]](%[[C0]])[%[[D4]]]
// CHECK-DAG:    %[[D7:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D8:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D9:.+]] = gpu.thread_id  z
// CHECK:        %[[D10:.+]] = vector.extract %[[D5]]#[[D1:.+]][0, 0, 0, 0] : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D11:.+]] = affine.apply #[[MAP3]](%[[D7]], %[[D8]], %[[D9]])
// CHECK-DAG:    %[[D12:.+]] = affine.apply #[[MAP4]](%[[D7]], %[[D8]], %[[D9]])
// CHECK:        %[[D13:.+]] = arith.addi %[[D11]], %[[D6]] : index
// CHECK:        %[[D14:.+]] = arith.addi %[[D12]], %[[C0]] : index
// CHECK:        memref.store %[[D10]], %[[D3]][%[[D13]], %[[D14]]] : memref<16x8xf16>
// CHECK:        %[[D15:.+]] = vector.extract %[[D5]]#[[D1]][0, 0, 0, 1] : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D16:.+]] = affine.apply #[[MAP5]](%[[D7]], %[[D8]], %[[D9]])
// CHECK:        %[[D17:.+]] = arith.addi %[[D16]], %[[C0]] : index
// CHECK:        memref.store %[[D15]], %[[D3]][%[[D13]], %[[D17]]] : memref<16x8xf16>
// CHECK:        %[[D18:.+]] = vector.extract %[[D5]]#[[D1]][0, 0, 1, 0] : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D19:.+]] = affine.apply #[[MAP8]](%[[D7]], %[[D8]], %[[D9]])
// CHECK:        %[[D20:.+]] = arith.addi %[[D19]], %[[D6]] : index
// CHECK:        memref.store %[[D18]], %[[D3]][%[[D20]], %[[D14]]] : memref<16x8xf16>
// CHECK:        %[[D21:.+]] = vector.extract %[[D5]]#[[D1]][0, 0, 1, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D21]], %[[D3]][%[[D20]], %[[D17]]] : memref<16x8xf16>
// CHECK:        return
// CHECK:      }

// -----

#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module {
  func.func @matmul_dispatch_0_matmul_16x8x16() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %cst_1 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x16xf16>
    memref.assume_alignment %1, 64 : memref<8x16xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %3, 64 : memref<16x8xf16>
    %5 = vector.transfer_read %0[%c0, %c0], %cst_1 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %6 = vector.transfer_read %1[%c0, %c0], %cst_1 {in_bounds = [true, true]} : memref<8x16xf16>, vector<8x16xf16>
    %7 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %6, %cst : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    %subview = memref.subview %3[%c0, 0] [16, 8] [1, 1] : memref<16x8xf16> to memref<16x8xf16, strided<[8, 1], offset: ?>>
    %8 = vector.transfer_read %2[%c0, %c0], %cst_1 {in_bounds = [true, true]} : memref<16x8xf16>, vector<16x8xf16>
    %9 = arith.subf %7, %8 : vector<16x8xf16>
    %10 = math.exp %9 : vector<16x8xf16>
    vector.transfer_write %10, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16, strided<[8, 1], offset: ?>>
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_dispatch_0_matmul_16x8x16() {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x16xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<8x16xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<16x8xf16>
// CHECK-DAG:    %[[D4:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D6:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D7:.+]] = affine.apply #[[MAP]](%[[D4]], %[[D5]], %[[D6]])
// CHECK-DAG:    %[[D8:.+]] = affine.apply #[[MAP1]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D9:.+]] = arith.addi %[[D7]], %[[C0]] : index
// CHECK:        %[[D10:.+]] = arith.addi %[[D8]], %[[C0]] : index
// CHECK:        %[[D11:.+]] = memref.load %[[D0]][%[[D9]], %[[D10]]] : memref<16x16xf16>
// CHECK:        %[[D12:.+]] = vector.broadcast %[[D11]] : f16 to vector<1xf16>
// CHECK:        %[[D13:.+]] = vector.insert_strided_slice %[[D12]], %[[CST_0]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D14:.+]] = affine.apply #[[MAP2]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D15:.+]] = arith.addi %[[D14]], %[[C0]] : index
// CHECK:        %[[D16:.+]] = memref.load %[[D0]][%[[D9]], %[[D15]]] : memref<16x16xf16>
// CHECK:        %[[D17:.+]] = vector.broadcast %[[D16]] : f16 to vector<1xf16>
// CHECK:        %[[D18:.+]] = vector.insert_strided_slice %[[D17]], %[[D13]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D19:.+]] = affine.apply #[[MAP3]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D20:.+]] = arith.addi %[[D19]], %[[C0]] : index
// CHECK:        %[[D21:.+]] = memref.load %[[D0]][%[[D9]], %[[D20]]] : memref<16x16xf16>
// CHECK:        %[[D22:.+]] = vector.broadcast %[[D21]] : f16 to vector<1xf16>
// CHECK:        %[[D23:.+]] = vector.insert_strided_slice %[[D22]], %[[D18]] {offsets = [0, 0, 2, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D24:.+]] = affine.apply #[[MAP4]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D25:.+]] = arith.addi %[[D24]], %[[C0]] : index
// CHECK:        %[[D26:.+]] = memref.load %[[D0]][%[[D9]], %[[D25]]] : memref<16x16xf16>
// CHECK:        %[[D27:.+]] = vector.broadcast %[[D26]] : f16 to vector<1xf16>
// CHECK:        %[[D28:.+]] = vector.insert_strided_slice %[[D27]], %[[D23]] {offsets = [0, 0, 2, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D29:.+]] = affine.apply #[[MAP5]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D30:.+]] = arith.addi %[[D29]], %[[C0]] : index
// CHECK:        %[[D31:.+]] = memref.load %[[D0]][%[[D30]], %[[D10]]] : memref<16x16xf16>
// CHECK:        %[[D32:.+]] = vector.broadcast %[[D31]] : f16 to vector<1xf16>
// CHECK:        %[[D33:.+]] = vector.insert_strided_slice %[[D32]], %[[D28]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D34:.+]] = memref.load %[[D0]][%[[D30]], %[[D15]]] : memref<16x16xf16>
// CHECK:        %[[D35:.+]] = vector.broadcast %[[D34]] : f16 to vector<1xf16>
// CHECK:        %[[D36:.+]] = vector.insert_strided_slice %[[D35]], %[[D33]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D37:.+]] = memref.load %[[D0]][%[[D30]], %[[D20]]] : memref<16x16xf16>
// CHECK:        %[[D38:.+]] = vector.broadcast %[[D37]] : f16 to vector<1xf16>
// CHECK:        %[[D39:.+]] = vector.insert_strided_slice %[[D38]], %[[D36]] {offsets = [0, 0, 3, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D40:.+]] = memref.load %[[D0]][%[[D30]], %[[D25]]] : memref<16x16xf16>
// CHECK:        %[[D41:.+]] = vector.broadcast %[[D40]] : f16 to vector<1xf16>
// CHECK:        %[[D42:.+]] = vector.insert_strided_slice %[[D41]], %[[D39]] {offsets = [0, 0, 3, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D43:.+]] = affine.apply #[[MAP6]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D44:.+]] = arith.addi %[[D43]], %[[C0]] : index
// CHECK:        %[[D45:.+]] = memref.load %[[D1]][%[[D44]], %[[D10]]] : memref<8x16xf16>
// CHECK:        %[[D46:.+]] = vector.broadcast %[[D45]] : f16 to vector<1xf16>
// CHECK:        %[[D47:.+]] = vector.insert_strided_slice %[[D46]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D48:.+]] = memref.load %[[D1]][%[[D44]], %[[D15]]] : memref<8x16xf16>
// CHECK:        %[[D49:.+]] = vector.broadcast %[[D48]] : f16 to vector<1xf16>
// CHECK:        %[[D50:.+]] = vector.insert_strided_slice %[[D49]], %[[D47]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D51:.+]] = memref.load %[[D1]][%[[D44]], %[[D20]]] : memref<8x16xf16>
// CHECK:        %[[D52:.+]] = vector.broadcast %[[D51]] : f16 to vector<1xf16>
// CHECK:        %[[D53:.+]] = vector.insert_strided_slice %[[D52]], %[[D50]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D54:.+]] = memref.load %[[D1]][%[[D44]], %[[D25]]] : memref<8x16xf16>
// CHECK:        %[[D55:.+]] = vector.broadcast %[[D54]] : f16 to vector<1xf16>
// CHECK:        %[[D56:.+]] = vector.insert_strided_slice %[[D55]], %[[D53]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D57:.+]] = vector.extract %[[CST]][0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D58:.+]] = vector.extract %[[D42]][0, 0] : vector<1x1x4x2xf16>
// CHECK:        %[[D59:.+]] = vector.extract %[[D56]][0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D60:.+]] = nvgpu.mma.sync(%[[D58]], %[[D59]], %[[D57]]) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>,
// CHECK-SAME:     vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D61:.+]] = vector.insert %[[D60]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[D3]][%[[C0]], 0] [16, 8] [1, 1] : memref<16x8xf16> to
// CHECK-SAME:     memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D62:.+]] = memref.load %[[D2]][%[[D9]], %[[D10]]] : memref<16x8xf16>
// CHECK:        %[[D63:.+]] = vector.broadcast %[[D62]] : f16 to vector<1xf16>
// CHECK:        %[[D64:.+]] = vector.insert_strided_slice %[[D63]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D65:.+]] = memref.load %[[D2]][%[[D9]], %[[D15]]] : memref<16x8xf16>
// CHECK:        %[[D66:.+]] = vector.broadcast %[[D65]] : f16 to vector<1xf16>
// CHECK:        %[[D67:.+]] = vector.insert_strided_slice %[[D66]], %[[D64]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D68:.+]] = memref.load %[[D2]][%[[D30]], %[[D10]]] : memref<16x8xf16>
// CHECK:        %[[D69:.+]] = vector.broadcast %[[D68]] : f16 to vector<1xf16>
// CHECK:        %[[D70:.+]] = vector.insert_strided_slice %[[D69]], %[[D67]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D71:.+]] = memref.load %[[D2]][%[[D30]], %[[D15]]] : memref<16x8xf16>
// CHECK:        %[[D72:.+]] = vector.broadcast %[[D71]] : f16 to vector<1xf16>
// CHECK:        %[[D73:.+]] = vector.insert_strided_slice %[[D72]], %[[D70]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D74:.+]] = arith.subf %[[D61]], %[[D73]] : vector<1x1x2x2xf16>
// CHECK:        %[[D75:.+]] = math.exp %[[D74]] : vector<1x1x2x2xf16>
// CHECK:        %[[D76:.+]] = vector.extract %[[D75]][0, 0, 0, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D76]], %[[SUBVIEW]][%[[D9]], %[[D10]]] : memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D77:.+]] = vector.extract %[[D75]][0, 0, 0, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D77]], %[[SUBVIEW]][%[[D9]], %[[D15]]] : memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D78:.+]] = vector.extract %[[D75]][0, 0, 1, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D78]], %[[SUBVIEW]][%[[D30]], %[[D10]]] : memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D79:.+]] = vector.extract %[[D75]][0, 0, 1, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D79]], %[[SUBVIEW]][%[[D30]], %[[D15]]] : memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        return
// CHECK:      }

// -----

builtin.module {
  func.func @matmul_dispatch_0_matmul_16x8x16_shared() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x16xf16, #gpu.address_space<workgroup>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x16xf16, #gpu.address_space<workgroup>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = vector.transfer_read %0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
    %4 = vector.transfer_read %1[%c0, %c0], %cst_0 {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : memref<8x16xf16, #gpu.address_space<workgroup>>, vector<8x16xf16>
    %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %cst : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    vector.transfer_write %5, %2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
  }
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> ((d0 + d1 * 4) mod 8)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
    // CHECK: func.func @matmul_dispatch_0_matmul_16x8x16_shared() {
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[TX:.+]] = gpu.thread_id  x
// CHECK-DAG:   %[[TY:.+]] = gpu.thread_id  y
// CHECK-DAG:   %[[VECOFF0:.+]] = affine.apply #[[MAP0]](%[[C0]], %[[C0]], %[[C0]])
// CHECK-DAG:   %[[VECOFF1:.+]] = affine.apply #[[MAP1]](%[[C0]], %[[C0]], %[[C0]])
// CHECK-DAG:   %[[LANEID:.+]] = affine.apply #[[MAP2]](%[[TX]], %[[TY]])
// CHECK-DAG:   %[[LANEOFF0:.+]] = affine.apply #[[MAP0]](%[[C0]], %[[LANEID]], %[[C0]])
// CHECK-DAG:   %[[LANEOFF1:.+]] = affine.apply #[[MAP1]](%[[C0]], %[[LANEID]], %[[C0]])
// CHECK-DAG:   %[[OFF0:.+]] = arith.addi %[[VECOFF0]], %[[C0]] : index
// CHECK-DAG:   %[[OFF1:.+]] = arith.addi %[[LANEOFF0]], %[[OFF0]] : index
// CHECK-DAG:   %[[OFF2:.+]] = arith.addi %[[VECOFF1]], %[[C0]] : index
// CHECK-DAG:   %[[OFF3:.+]] = arith.addi %[[LANEOFF1]], %[[OFF2]] : index
//     CHECK:   %[[LD0:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF1]], %[[OFF3]]] {numTiles = 1 : i32, transpose = false} : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V0:.+]] = vector.insert_strided_slice %[[LD0]], %{{.*}} {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x4x2xf16>
//     CHECK:   %[[VECOFF2:.+]] = affine.apply #[[MAP3]](%[[C0]], %[[C0]], %[[C0]])
//     CHECK:   %[[OFF4:.+]] = arith.addi %[[VECOFF2]], %[[C0]] : index
//     CHECK:   %[[OFF5:.+]] = arith.addi %[[LANEOFF1]], %[[OFF4]] : index
//     CHECK:   %[[LD1:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF1]], %[[OFF5]]] {numTiles = 1 : i32, transpose = false} : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V1:.+]] = vector.insert_strided_slice %[[LD1]], %[[V0]] {offsets = [0, 0, 2, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x4x2xf16>
//     CHECK:   %[[VECOFF3:.+]] = affine.apply #[[MAP4]](%[[C0]], %[[C0]], %[[C0]])
//     CHECK:   %[[OFF6:.+]] = arith.addi %[[VECOFF3]], %[[C0]] : index
//     CHECK:   %[[OFF7:.+]] = arith.addi %[[LANEOFF0]], %[[OFF6]] : index
//     CHECK:   %[[LD2:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF7]], %[[OFF3]]] {numTiles = 1 : i32, transpose = false} : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V2:.+]] = vector.insert_strided_slice %[[LD2]], %[[V1]] {offsets = [0, 0, 1, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x4x2xf16>
//     CHECK:   %[[LD3:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF7]], %[[OFF5]]] {numTiles = 1 : i32, transpose = false} : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V3:.+]] = vector.insert_strided_slice %[[LD3]], %[[V2]] {offsets = [0, 0, 3, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:   %[[VECOFF2:.+]] = affine.apply #[[MAP5]](%[[C0]], %[[C0]], %[[C0]])
// CHECK-DAG:   %[[LANEOFF2:.+]] = affine.apply #[[MAP5]](%[[C0]], %[[LANEID]], %[[C0]])
// CHECK-DAG:   %[[OFF8:.+]] = arith.addi %[[VECOFF2]], %[[C0]] : index
// CHECK-DAG:   %[[OFF9:.+]] = arith.addi %[[LANEOFF1]], %[[OFF8]] : index
// CHECK-DAG:   %[[OFF10:.+]] = arith.addi %[[LANEOFF2]], %[[OFF2]] : index
//     CHECK:   %[[LD3:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF10]], %[[OFF9]]] {numTiles = 1 : i32, transpose = true} : memref<8x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V4:.+]] = vector.insert_strided_slice %[[LD3]], %{{.*}} {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x2x2xf16>
//     CHECK:   %[[OFF11:.+]] = arith.addi %[[LANEOFF2]], %[[OFF4]] : index
//     CHECK:   %[[LD4:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF11]], %[[OFF9]]] {numTiles = 1 : i32, transpose = true} : memref<8x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V5:.+]] = vector.insert_strided_slice %[[LD4]], %[[V4]] {offsets = [0, 0, 1, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x2x2xf16>
//     CHECK:   %[[A:.+]] = vector.extract %[[V3]][0, 0] : vector<1x1x4x2xf16>
//     CHECK:   %[[B:.+]] = vector.extract %[[V5]][0, 0] : vector<1x1x2x2xf16>
//     CHECK:   nvgpu.mma.sync(%[[A]], %[[B]], %{{.*}}) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>

// -----

#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module {
  func.func @matmul_dispatch_0_matmul_16x16x16_f16() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x16xf16>
    %c0_0 = arith.constant 0 : index
    %cst_1 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %1, 64 : memref<16x16xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16xf16>
    memref.assume_alignment %2, 64 : memref<16xf16>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16xf16>
    memref.assume_alignment %3, 64 : memref<16xf16>
    %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %4, 64 : memref<16x8xf16>
    %5 = hal.interface.binding.subspan set(0) binding(5) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %5, 64 : memref<16x8xf16>
    %6 = hal.interface.binding.subspan set(0) binding(6) type(storage_buffer) alignment(64) offset(%c0_0) : memref<16x8xf16>
    memref.assume_alignment %6, 64 : memref<16x8xf16>
    %c1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %workgroup_count_z = hal.interface.workgroup.count[2] : index
    %c1_4 = arith.constant 1 : index
    %7 = affine.apply #map(%workgroup_id_x)
    %8 = vector.transfer_read %0[%7, %c0_0], %cst_1 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %9 = vector.transfer_read %1[%c0_0, %c0_0], %cst_1 {in_bounds = [true, true], permutation_map = #map1} : memref<16x16xf16>, vector<16x16xf16>
    %10 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %9, %cst : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
    %11 = vector.transfer_read %2[%7], %cst_1 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    %12 = vector.multi_reduction <maxf>, %10, %11 [1] : vector<16x16xf16> to vector<16xf16>
    %13 = vector.transfer_read %3[%7], %cst_1 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    %14 = arith.subf %11, %12 : vector<16xf16>
    %15 = math.exp %14 : vector<16xf16>
    %16 = arith.mulf %15, %13 : vector<16xf16>
    %17 = vector.broadcast %12 : vector<16xf16> to vector<16x16xf16>
    %18 = vector.transpose %17, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %19 = arith.subf %10, %18 : vector<16x16xf16>
    %20 = math.exp %19 : vector<16x16xf16>
    %21 = vector.multi_reduction <add>, %20, %16 [1] : vector<16x16xf16> to vector<16xf16>
    %22 = vector.broadcast %21 : vector<16xf16> to vector<16x16xf16>
    %23 = vector.transpose %22, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %24 = arith.divf %20, %23 : vector<16x16xf16>
    %subview = memref.subview %6[%7, 0] [16, 8] [1, 1] : memref<16x8xf16> to memref<16x8xf16, strided<[8, 1], offset: ?>>
    %25 = vector.broadcast %16 : vector<16xf16> to vector<8x16xf16>
    %26 = vector.broadcast %21 : vector<16xf16> to vector<8x16xf16>
    %27 = vector.transfer_read %5[%7, %c0_0], %cst_1 {in_bounds = [true, true]} : memref<16x8xf16>, vector<16x8xf16>
    %28 = arith.divf %25, %26 : vector<8x16xf16>
    %29 = vector.transpose %28, [1, 0] : vector<8x16xf16> to vector<16x8xf16>
    %30 = arith.mulf %29, %27 : vector<16x8xf16>
    %31 = vector.transfer_read %4[%c0_0, %c0_0], %cst_1 {in_bounds = [true, true], permutation_map = #map1} : memref<16x8xf16>, vector<8x16xf16>
    %32 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %31, %30 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    vector.transfer_write %32, %subview[%c0_0, %c0_0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16, strided<[8, 1], offset: ?>>
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %reordered_func = transform.iree.reorder_transpose %top_level_func : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %reordered_func { cse } : (!transform.any_op) -> ()
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:      func.func @matmul_dispatch_0_matmul_16x16x16_f16() {
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf16>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant 0.000000e+00 : f16
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<16x16xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16xf16>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16xf16>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<16xf16>
// CHECK:        %[[D4:.+]] = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D4]], 64 : memref<16x8xf16>
// CHECK:        %[[D5:.+]] = hal.interface.binding.subspan set(0) binding(5) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D5]], 64 : memref<16x8xf16>
// CHECK:        %[[D6:.+]] = hal.interface.binding.subspan set(0) binding(6) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D6]], 64 : memref<16x8xf16>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG:    %[[D7:.+]] = affine.apply #[[MAP]](%[[WORKGROUP_ID_X]])
// CHECK:        %[[D8:.+]] = vector.transfer_read %[[D0]][%[[D7]], %[[C0]]], %[[CST_0]] {in_bounds = [true, true]} :
// CHECK-SAME:     memref<16x16xf16>, vector<16x16xf16>
// CHECK:        %[[D9:.+]] = vector.transfer_read %[[D1]][%[[C0]], %[[C0]]], %[[CST_0]] {in_bounds = [true, true],
// CHECK-SAME:     permutation_map = #[[MAP1]]} : memref<16x16xf16>, vector<16x16xf16>
// CHECK:        %[[D10:.+]] = vector.contract {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "reduction"], kind = #[[VECTOR:.+]].kind<add>} %[[D8]], %[[D9]], %[[CST]] :
// CHECK-SAME:     vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
// CHECK:        %[[D11:.+]] = vector.transfer_read %[[D2]][%[[D7]]], %[[CST_0]] {in_bounds = [true]} : memref<16xf16>,
// CHECK-SAME:     vector<16xf16>
// CHECK:        %[[D12:.+]] = vector.multi_reduction <maxf>, %[[D10]], %[[D11]] [1] : vector<16x16xf16> to
// CHECK-SAME:     vector<16xf16>
// CHECK:        %[[D13:.+]] = vector.transfer_read %[[D3]][%[[D7]]], %[[CST_0]] {in_bounds = [true]} : memref<16xf16>,
// CHECK-SAME:     vector<16xf16>
// CHECK:        %[[D14:.+]] = arith.subf %[[D11]], %[[D12]] : vector<16xf16>
// CHECK:        %[[D15:.+]] = math.exp %[[D14]] : vector<16xf16>
// CHECK:        %[[D16:.+]] = arith.mulf %[[D15]], %[[D13]] : vector<16xf16>
// CHECK:        %[[D17:.+]] = vector.broadcast %[[D12]] : vector<16xf16> to vector<16x16xf16>
// CHECK:        %[[D18:.+]] = vector.transpose %[[D17]], [1, 0] : vector<16x16xf16> to vector<16x16xf16>
// CHECK:        %[[D19:.+]] = arith.subf %[[D10]], %[[D18]] : vector<16x16xf16>
// CHECK:        %[[D20:.+]] = math.exp %[[D19]] : vector<16x16xf16>
// CHECK:        %[[D21:.+]] = vector.multi_reduction <add>, %[[D20]], %[[D16]] [1] : vector<16x16xf16> to
// CHECK-SAME:     vector<16xf16>
// CHECK:        %[[D22:.+]] = vector.broadcast %[[D21]] : vector<16xf16> to vector<16x16xf16>
// CHECK:        %[[D23:.+]] = vector.transpose %[[D22]], [1, 0] : vector<16x16xf16> to vector<16x16xf16>
// CHECK:        %[[D24:.+]] = arith.divf %[[D20]], %[[D23]] : vector<16x16xf16>
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[D6]][%[[D7]], 0] [16, 8] [1, 1] : memref<16x8xf16> to
// CHECK-SAME:     memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D25:.+]] = vector.broadcast %[[D16]] : vector<16xf16> to vector<8x16xf16>
// CHECK:        %[[D26:.+]] = vector.broadcast %[[D21]] : vector<16xf16> to vector<8x16xf16>
// CHECK:        %[[D27:.+]] = vector.transfer_read %[[D5]][%[[D7]], %[[C0]]], %[[CST_0]] {in_bounds = [true, true]} :
// CHECK-SAME:     memref<16x8xf16>, vector<16x8xf16>
// CHECK:        %[[D28:.+]] = vector.transpose %[[D25]], [1, 0] : vector<8x16xf16> to vector<16x8xf16>
// CHECK:        %[[D29:.+]] = vector.transpose %[[D26]], [1, 0] : vector<8x16xf16> to vector<16x8xf16>
// CHECK:        %[[D30:.+]] = arith.divf %[[D28]], %[[D29]] : vector<16x8xf16>
// CHECK:        %[[D31:.+]] = arith.mulf %[[D30]], %[[D27]] : vector<16x8xf16>
// CHECK:        %[[D32:.+]] = vector.transfer_read %[[D4]][%[[C0]], %[[C0]]], %[[CST_0]] {in_bounds = [true, true],
// CHECK-SAME:     permutation_map = #[[MAP1]]} : memref<16x8xf16>, vector<8x16xf16>
// CHECK:        %[[D33:.+]] = vector.contract {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "reduction"], kind = #[[VECTOR]].kind<add>} %[[D24]], %[[D32]], %[[D31]] :
// CHECK-SAME:     vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
// CHECK:        vector.transfer_write %[[D33]], %[[SUBVIEW]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} :
// CHECK-SAME:     vector<16x8xf16>, memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        return
// CHECK:      }

// -----

#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module {
  func.func @double_matmul_dispatch_0_matmul_16x16x16() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x16xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %c0_1 = arith.constant 0 : index
    %cst_2 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0_1) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0_1) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %1, 64 : memref<16x16xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0_1) flags(ReadOnly) : memref<8x16xf16>
    memref.assume_alignment %2, 64 : memref<8x16xf16>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0_1) : memref<16x8xf16>
    memref.assume_alignment %3, 64 : memref<16x8xf16>
    %c1 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    %c1_4 = arith.constant 1 : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %workgroup_count_z = hal.interface.workgroup.count[2] : index
    %c1_5 = arith.constant 1 : index
    %4 = affine.apply #map(%workgroup_id_x)
    %5 = vector.transfer_read %0[%4, %c0_1], %cst_2 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %6 = vector.transfer_read %1[%c0_1, %c0_1], %cst_2 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %7 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %6, %cst : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
    %subview = memref.subview %3[%4, 0] [16, 8] [1, 1] : memref<16x8xf16> to memref<16x8xf16, strided<[8, 1], offset: ?>>
    %8 = vector.transfer_read %2[%c0_1, %c0_1], %cst_2 {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : memref<8x16xf16>, vector<8x16xf16>
    %9 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %8, %cst_0 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    vector.transfer_write %9, %subview[%c0_1, %c0_1] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16, strided<[8, 1], offset: ?>>
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP7:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK-DAG:  #[[MAP8:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 8)>
// CHECK:      func.func @double_matmul_dispatch_0_matmul_16x16x16() {
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x2x2x2xf16>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<16x16xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x16xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<8x16xf16>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<16x8xf16>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG:    %[[D4:.+]] = affine.apply #[[MAP]](%[[WORKGROUP_ID_X]])
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D6:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D7:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D8:.+]] = affine.apply #[[MAP1]](%[[D5]], %[[D6]], %[[D7]])
// CHECK-DAG:    %[[D9:.+]] = affine.apply #[[MAP2]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D10:.+]] = arith.addi %[[D8]], %[[D4]] : index
// CHECK:        %[[D11:.+]] = arith.addi %[[D9]], %[[C0]] : index
// CHECK:        %[[D12:.+]] = memref.load %[[D0]][%[[D10]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D13:.+]] = vector.broadcast %[[D12]] : f16 to vector<1xf16>
// CHECK:        %[[D14:.+]] = vector.insert_strided_slice %[[D13]], %[[CST_1]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D15:.+]] = affine.apply #[[MAP3]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D16:.+]] = arith.addi %[[D15]], %[[C0]] : index
// CHECK:        %[[D17:.+]] = memref.load %[[D0]][%[[D10]], %[[D16]]] : memref<16x16xf16>
// CHECK:        %[[D18:.+]] = vector.broadcast %[[D17]] : f16 to vector<1xf16>
// CHECK:        %[[D19:.+]] = vector.insert_strided_slice %[[D18]], %[[D14]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D20:.+]] = affine.apply #[[MAP4]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D21:.+]] = arith.addi %[[D20]], %[[C0]] : index
// CHECK:        %[[D22:.+]] = memref.load %[[D0]][%[[D10]], %[[D21]]] : memref<16x16xf16>
// CHECK:        %[[D23:.+]] = vector.broadcast %[[D22]] : f16 to vector<1xf16>
// CHECK:        %[[D24:.+]] = vector.insert_strided_slice %[[D23]], %[[D19]] {offsets = [0, 0, 2, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D25:.+]] = affine.apply #[[MAP5]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D26:.+]] = arith.addi %[[D25]], %[[C0]] : index
// CHECK:        %[[D27:.+]] = memref.load %[[D0]][%[[D10]], %[[D26]]] : memref<16x16xf16>
// CHECK:        %[[D28:.+]] = vector.broadcast %[[D27]] : f16 to vector<1xf16>
// CHECK:        %[[D29:.+]] = vector.insert_strided_slice %[[D28]], %[[D24]] {offsets = [0, 0, 2, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D30:.+]] = affine.apply #[[MAP6]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D31:.+]] = arith.addi %[[D30]], %[[D4]] : index
// CHECK:        %[[D32:.+]] = memref.load %[[D0]][%[[D31]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D33:.+]] = vector.broadcast %[[D32]] : f16 to vector<1xf16>
// CHECK:        %[[D34:.+]] = vector.insert_strided_slice %[[D33]], %[[D29]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D35:.+]] = memref.load %[[D0]][%[[D31]], %[[D16]]] : memref<16x16xf16>
// CHECK:        %[[D36:.+]] = vector.broadcast %[[D35]] : f16 to vector<1xf16>
// CHECK:        %[[D37:.+]] = vector.insert_strided_slice %[[D36]], %[[D34]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D38:.+]] = memref.load %[[D0]][%[[D31]], %[[D21]]] : memref<16x16xf16>
// CHECK:        %[[D39:.+]] = vector.broadcast %[[D38]] : f16 to vector<1xf16>
// CHECK:        %[[D40:.+]] = vector.insert_strided_slice %[[D39]], %[[D37]] {offsets = [0, 0, 3, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D41:.+]] = memref.load %[[D0]][%[[D31]], %[[D26]]] : memref<16x16xf16>
// CHECK:        %[[D42:.+]] = vector.broadcast %[[D41]] : f16 to vector<1xf16>
// CHECK:        %[[D43:.+]] = vector.insert_strided_slice %[[D42]], %[[D40]] {offsets = [0, 0, 3, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant dense<0.000000e+00> : vector<2x1x2x2xf16>
// CHECK-DAG:    %[[D44:.+]] = affine.apply #[[MAP7]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D45:.+]] = arith.addi %[[D44]], %[[C0]] : index
// CHECK:        %[[D46:.+]] = memref.load %[[D1]][%[[D45]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D47:.+]] = vector.broadcast %[[D46]] : f16 to vector<1xf16>
// CHECK:        %[[D48:.+]] = vector.insert_strided_slice %[[D47]], %[[CST_2]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D49:.+]] = memref.load %[[D1]][%[[D45]], %[[D16]]] : memref<16x16xf16>
// CHECK:        %[[D50:.+]] = vector.broadcast %[[D49]] : f16 to vector<1xf16>
// CHECK:        %[[D51:.+]] = vector.insert_strided_slice %[[D50]], %[[D48]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D52:.+]] = memref.load %[[D1]][%[[D45]], %[[D21]]] : memref<16x16xf16>
// CHECK:        %[[D53:.+]] = vector.broadcast %[[D52]] : f16 to vector<1xf16>
// CHECK:        %[[D54:.+]] = vector.insert_strided_slice %[[D53]], %[[D51]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D55:.+]] = memref.load %[[D1]][%[[D45]], %[[D26]]] : memref<16x16xf16>
// CHECK:        %[[D56:.+]] = vector.broadcast %[[D55]] : f16 to vector<1xf16>
// CHECK:        %[[D57:.+]] = vector.insert_strided_slice %[[D56]], %[[D54]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK-DAG:    %[[D58:.+]] = affine.apply #[[MAP8]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D59:.+]] = arith.addi %[[D58]], %[[C0]] : index
// CHECK:        %[[D60:.+]] = memref.load %[[D1]][%[[D59]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D61:.+]] = vector.broadcast %[[D60]] : f16 to vector<1xf16>
// CHECK:        %[[D62:.+]] = vector.insert_strided_slice %[[D61]], %[[D57]] {offsets = [1, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D63:.+]] = memref.load %[[D1]][%[[D59]], %[[D16]]] : memref<16x16xf16>
// CHECK:        %[[D64:.+]] = vector.broadcast %[[D63]] : f16 to vector<1xf16>
// CHECK:        %[[D65:.+]] = vector.insert_strided_slice %[[D64]], %[[D62]] {offsets = [1, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D66:.+]] = memref.load %[[D1]][%[[D59]], %[[D21]]] : memref<16x16xf16>
// CHECK:        %[[D67:.+]] = vector.broadcast %[[D66]] : f16 to vector<1xf16>
// CHECK:        %[[D68:.+]] = vector.insert_strided_slice %[[D67]], %[[D65]] {offsets = [1, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D69:.+]] = memref.load %[[D1]][%[[D59]], %[[D26]]] : memref<16x16xf16>
// CHECK:        %[[D70:.+]] = vector.broadcast %[[D69]] : f16 to vector<1xf16>
// CHECK:        %[[D71:.+]] = vector.insert_strided_slice %[[D70]], %[[D68]] {offsets = [1, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D72:.+]] = vector.extract %[[CST]][0, 0] : vector<1x2x2x2xf16>
// CHECK:        %[[D73:.+]] = vector.extract %[[D43]][0, 0] : vector<1x1x4x2xf16>
// CHECK:        %[[D74:.+]] = vector.extract %[[D71]][0, 0] : vector<2x1x2x2xf16>
// CHECK:        %[[D75:.+]] = nvgpu.mma.sync(%[[D73]], %[[D74]], %[[D72]]) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>,
// CHECK-SAME:     vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D76:.+]] = vector.insert %[[D75]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x2x2x2xf16>
// CHECK:        %[[D77:.+]] = vector.extract %[[CST]][0, 1] : vector<1x2x2x2xf16>
// CHECK:        %[[D78:.+]] = vector.extract %[[D71]][1, 0] : vector<2x1x2x2xf16>
// CHECK:        %[[D79:.+]] = nvgpu.mma.sync(%[[D73]], %[[D78]], %[[D77]]) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>,
// CHECK-SAME:     vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D80:.+]] = vector.insert %[[D79]], %[[D76]] [0, 1] : vector<2x2xf16> into vector<1x2x2x2xf16>
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[D3]][%[[D4]], 0] [16, 8] [1, 1] : memref<16x8xf16> to
// CHECK-SAME:     memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D81:.+]] = memref.load %[[D2]][%[[D11]], %[[D45]]] : memref<8x16xf16>
// CHECK:        %[[D82:.+]] = vector.broadcast %[[D81]] : f16 to vector<1xf16>
// CHECK:        %[[D83:.+]] = vector.insert_strided_slice %[[D82]], %[[CST_0]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D84:.+]] = memref.load %[[D2]][%[[D16]], %[[D45]]] : memref<8x16xf16>
// CHECK:        %[[D85:.+]] = vector.broadcast %[[D84]] : f16 to vector<1xf16>
// CHECK:        %[[D86:.+]] = vector.insert_strided_slice %[[D85]], %[[D83]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D87:.+]] = memref.load %[[D2]][%[[D21]], %[[D45]]] : memref<8x16xf16>
// CHECK:        %[[D88:.+]] = vector.broadcast %[[D87]] : f16 to vector<1xf16>
// CHECK:        %[[D89:.+]] = vector.insert_strided_slice %[[D88]], %[[D86]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D90:.+]] = memref.load %[[D2]][%[[D26]], %[[D45]]] : memref<8x16xf16>
// CHECK:        %[[D91:.+]] = vector.broadcast %[[D90]] : f16 to vector<1xf16>
// CHECK:        %[[D92:.+]] = vector.insert_strided_slice %[[D91]], %[[D89]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D93:.+]] = vector.extract %[[D80]][0, 0] : vector<1x2x2x2xf16>
// CHECK:        %[[D94:.+]] = vector.insert_strided_slice %[[D93]], %[[CST_1]] {offsets = [0, 0, 0, 0], strides = [1,
// CHECK-SAME:     1]} : vector<2x2xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D95:.+]] = vector.extract %[[D80]][0, 1] : vector<1x2x2x2xf16>
// CHECK:        %[[D96:.+]] = vector.insert_strided_slice %[[D95]], %[[D94]] {offsets = [0, 0, 2, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<2x2xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D97:.+]] = vector.extract %[[CST_0]][0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D98:.+]] = vector.extract %[[D96]][0, 0] : vector<1x1x4x2xf16>
// CHECK:        %[[D99:.+]] = vector.extract %[[D92]][0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D100:.+]] = nvgpu.mma.sync(%[[D98]], %[[D99]], %[[D97]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:     (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D101:.+]] = vector.insert %[[D100]], %[[CST_0]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D102:.+]] = vector.extract %[[D101]][0, 0, 0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D103:.+]] = arith.addi %[[D8]], %[[C0]] : index
// CHECK:        memref.store %[[D102]], %[[SUBVIEW]][%[[D103]], %[[D11]]] : memref<16x8xf16, strided<[8, 1], offset:
// CHECK-SAME:     ?>>
// CHECK:        %[[D104:.+]] = vector.extract %[[D101]][0, 0, 0, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D104]], %[[SUBVIEW]][%[[D103]], %[[D16]]] : memref<16x8xf16, strided<[8, 1], offset:
// CHECK-SAME:     ?>>
// CHECK:        %[[D105:.+]] = vector.extract %[[D101]][0, 0, 1, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D106:.+]] = arith.addi %[[D30]], %[[C0]] : index
// CHECK:        memref.store %[[D105]], %[[SUBVIEW]][%[[D106]], %[[D11]]] : memref<16x8xf16, strided<[8, 1], offset:
// CHECK-SAME:     ?>>
// CHECK:        %[[D107:.+]] = vector.extract %[[D101]][0, 0, 1, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D107]], %[[SUBVIEW]][%[[D106]], %[[D16]]] : memref<16x8xf16, strided<[8, 1], offset:
// CHECK-SAME:     ?>>
// CHECK:        return
// CHECK:      }

// -----

#map = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<()[s0, s1, s2] -> (s2 * 32 + ((s0 + s1 * 4) floordiv 32) * 32 - ((s2 + (s0 + s1 * 4) floordiv 32) floordiv 4) * 128)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module {
  func.func @attention_dispatch_0_attention_20x1024x64xf16() {
    %cst = arith.constant dense<0.000000e+00> : vector<32x64xf32>
    %cst_0 = arith.constant dense<-1.000000e+30> : vector<32xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<32xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<32x128xf32>
    %cst_3 = arith.constant dense<1.000000e+00> : vector<32xf32>
    %cst_4 = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<20x1024x64xf16>
    memref.assume_alignment %0, 64 : memref<20x1024x64xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<20x1024x64xf16>
    memref.assume_alignment %1, 64 : memref<20x1024x64xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<20x1024x64xf16>
    memref.assume_alignment %2, 64 : memref<20x1024x64xf16>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<20x1024x64xf16>
    memref.assume_alignment %3, 64 : memref<20x1024x64xf16>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %4 = affine.apply #map()[%workgroup_id_y]
    %subview = memref.subview %0[%workgroup_id_x, %4, 0] [1, 128, 64] [1, 1, 1] : memref<20x1024x64xf16> to memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>>
    %subview_6 = memref.subview %3[%workgroup_id_x, %4, 0] [1, 128, 64] [1, 1, 1] : memref<20x1024x64xf16> to memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>>) outs(%alloc : memref<1x128x64xf16, #gpu.address_space<workgroup>>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    }
    gpu.barrier
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview_6 : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>>) outs(%alloc_7 : memref<1x128x64xf16, #gpu.address_space<workgroup>>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    }
    gpu.barrier
    %5 = gpu.thread_id  x
    %6 = gpu.thread_id  y
    %7 = gpu.thread_id  z
    %8 = affine.apply #map2()[%5, %6, %7]
    gpu.barrier
    gpu.barrier
    gpu.barrier
    %9 = vector.transfer_read %alloc[%c0, %8, %c0], %cst_4 {in_bounds = [true, true]} : memref<1x128x64xf16, #gpu.address_space<workgroup>>, vector<32x64xf16>
    %11:3 = scf.for %arg0 = %c0 to %c1024 step %c128 iter_args(%arg1 = %cst_0, %arg2 = %cst_1, %arg3 = %cst) -> (vector<32xf32>, vector<32xf32>, vector<32x64xf32>) {
      %subview_8 = memref.subview %1[%workgroup_id_x, %arg0, 0] [1, 128, 64] [1, 1, 1] : memref<20x1024x64xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %subview_9 = memref.subview %2[%workgroup_id_x, %arg0, 0] [1, 128, 64] [1, 1, 1] : memref<20x1024x64xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%subview_8 : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_10 : memref<128x64xf16, #gpu.address_space<workgroup>>) {
      ^bb0(%in: f16, %out: f16):
        linalg.yield %in : f16
      }
      gpu.barrier
      %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%subview_9 : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_11 : memref<128x64xf16, #gpu.address_space<workgroup>>) {
      ^bb0(%in: f16, %out: f16):
        linalg.yield %in : f16
      }
      gpu.barrier
      %13 = vector.transfer_read %alloc_10[%c0, %c0], %cst_4 {in_bounds = [true, true]} : memref<128x64xf16, #gpu.address_space<workgroup>>, vector<128x64xf16>
      %15 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %9, %13, %cst_2 : vector<32x64xf16>, vector<128x64xf16> into vector<32x128xf32>
      %16 = vector.multi_reduction <maxf>, %15, %arg1 [1] : vector<32x128xf32> to vector<32xf32>
      %17 = vector.broadcast %16 : vector<32xf32> to vector<128x32xf32>
      %18 = vector.transpose %17, [1, 0] : vector<128x32xf32> to vector<32x128xf32>
      %19 = arith.subf %15, %18 : vector<32x128xf32>
      %20 = math.exp2 %19 : vector<32x128xf32>
      %21 = arith.subf %arg1, %16 : vector<32xf32>
      %22 = math.exp2 %21 : vector<32xf32>
      %23 = arith.mulf %22, %arg2 : vector<32xf32>
      %24 = vector.multi_reduction <add>, %20, %23 [1] : vector<32x128xf32> to vector<32xf32>
      %25 = arith.divf %cst_3, %24 : vector<32xf32>
      %26 = vector.broadcast %25 : vector<32xf32> to vector<128x32xf32>
      %27 = vector.transpose %26, [1, 0] : vector<128x32xf32> to vector<32x128xf32>
      %28 = arith.mulf %20, %27 : vector<32x128xf32>
      %29 = arith.truncf %28 : vector<32x128xf32> to vector<32x128xf16>
      %30 = vector.broadcast %23 : vector<32xf32> to vector<64x32xf32>
      %31 = vector.broadcast %25 : vector<32xf32> to vector<64x32xf32>
      %32 = arith.mulf %30, %31 : vector<64x32xf32>
      %33 = vector.transpose %32, [1, 0] : vector<64x32xf32> to vector<32x64xf32>
      %34 = arith.mulf %33, %arg3 : vector<32x64xf32>
      %35 = vector.transfer_read %alloc_11[%c0, %c0], %cst_4 {in_bounds = [true, true]} : memref<128x64xf16, #gpu.address_space<workgroup>>, vector<128x64xf16>
      %38 = vector.transpose %35, [1, 0] : vector<128x64xf16> to vector<64x128xf16>
      %39 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %29, %38, %34 : vector<32x128xf16>, vector<64x128xf16> into vector<32x64xf32>
      gpu.barrier
      memref.dealloc %alloc_10 : memref<128x64xf16, #gpu.address_space<workgroup>>
      memref.dealloc %alloc_11 : memref<128x64xf16, #gpu.address_space<workgroup>>
      scf.yield %16, %24, %39 : vector<32xf32>, vector<32xf32>, vector<32x64xf32>
    }
    %12 = arith.truncf %11#2 : vector<32x64xf32> to vector<32x64xf16>
    vector.transfer_write %12, %alloc_7[%c0, %8, %c0] {in_bounds = [true, true]} : vector<32x64xf16>, memref<1x128x64xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    memref.dealloc %alloc : memref<1x128x64xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_7 : memref<1x128x64xf16, #gpu.address_space<workgroup>>) outs(%subview_6 : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    }
    gpu.barrier
    memref.dealloc %alloc_7 : memref<1x128x64xf16, #gpu.address_space<workgroup>>
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %reordered_func = transform.iree.reorder_transpose %top_level_func : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %reordered_func { cse } : (!transform.any_op) -> ()
    transform.iree.apply_patterns %reordered_func {  prepare_vector_to_mma, cse } : (!transform.any_op) -> ()
    %transformed_func = transform.iree.layout_analysis_and_distribution %reordered_func : (!transform.any_op) -> (!transform.any_op)
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> (s2 * 32 + ((s0 + s1 * 4) floordiv 32) * 32 - ((s2 + (s0 + s1 * 4) floordiv 32) floordiv 4) * 128)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1) -> ((d0 + d1 * 4) mod 8)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP7:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP8:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 16)>
// CHECK-DAG:  #[[MAP9:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 24)>
// CHECK-DAG:  #[[MAP10:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 32)>
// CHECK-DAG:  #[[MAP11:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 40)>
// CHECK-DAG:  #[[MAP12:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 48)>
// CHECK-DAG:  #[[MAP13:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 56)>
// CHECK-DAG:  #[[MAP14:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 16)>
// CHECK-DAG:  #[[MAP15:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 24)>
// CHECK-DAG:  #[[MAP16:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[MAP17:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK-DAG:  #[[MAP18:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 8)>
// CHECK-DAG:  #[[MAP19:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 16)>
// CHECK-DAG:  #[[MAP20:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 24)>
// CHECK-DAG:  #[[MAP21:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 32)>
// CHECK-DAG:  #[[MAP22:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 40)>
// CHECK-DAG:  #[[MAP23:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 48)>
// CHECK-DAG:  #[[MAP24:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 56)>
// CHECK-DAG:  #[[MAP25:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 64)>
// CHECK-DAG:  #[[MAP26:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 72)>
// CHECK-DAG:  #[[MAP27:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 80)>
// CHECK-DAG:  #[[MAP28:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 88)>
// CHECK-DAG:  #[[MAP29:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 96)>
// CHECK-DAG:  #[[MAP30:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 104)>
// CHECK-DAG:  #[[MAP31:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 112)>
// CHECK-DAG:  #[[MAP32:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 120)>
// CHECK-DAG:  #[[MAP33:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 64)>
// CHECK-DAG:  #[[MAP34:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 72)>
// CHECK-DAG:  #[[MAP35:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 80)>
// CHECK-DAG:  #[[MAP36:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 88)>
// CHECK-DAG:  #[[MAP37:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 96)>
// CHECK-DAG:  #[[MAP38:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 104)>
// CHECK-DAG:  #[[MAP39:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 112)>
// CHECK-DAG:  #[[MAP40:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 120)>
// CHECK-DAG:  #[[MAP41:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP42:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP43:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 17)>
// CHECK-DAG:  #[[MAP44:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 25)>
// CHECK-DAG:  #[[MAP45:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 33)>
// CHECK-DAG:  #[[MAP46:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 41)>
// CHECK-DAG:  #[[MAP47:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 49)>
// CHECK-DAG:  #[[MAP48:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 57)>
// CHECK:      func.func @attention_dispatch_0_attention_20x1024x64xf16() {
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<2x8x2x2xf32>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<-1.000000e+30> : vector<2x16x2x2xf32>
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<2x16x2x2xf32>
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant dense<1.000000e+00> : vector<2x16x2x2xf32>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<20x1024x64xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<20x1024x64xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<20x1024x64xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<20x1024x64xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<20x1024x64xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<20x1024x64xf16>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<20x1024x64xf16>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<20x1024x64xf16>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:        %[[WORKGROUP_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-DAG:    %[[D4:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_Y]]]
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[D0]][%[[WORKGROUP_ID_X]], %[[D4]], 0] [1, 128, 64] [1, 1, 1] :
// CHECK-SAME:     memref<20x1024x64xf16> to memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>>
// CHECK:        %[[SUBVIEW_3:.+]] = memref.subview %[[D3]][%[[WORKGROUP_ID_X]], %[[D4]], 0] [1, 128, 64] [1, 1, 1] :
// CHECK-SAME:     memref<20x1024x64xf16> to memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>>
// CHECK:        %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU:.+]].address_space<workgroup>>
// CHECK:        gpu.barrier
// CHECK:        linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel",
// CHECK-SAME:     "parallel"]} ins(%[[SUBVIEW]] : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>>)
// CHECK-SAME:     outs(%[[ALLOC]] : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:        ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:          linalg.yield %[[IN]] : f16
// CHECK:        }
// CHECK:        gpu.barrier
// CHECK:        %[[ALLOC_4:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        gpu.barrier
// CHECK:        linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel",
// CHECK-SAME:     "parallel"]} ins(%[[SUBVIEW_3]] : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>>)
// CHECK-SAME:     outs(%[[ALLOC_4]] : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:        ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:          linalg.yield %[[IN]] : f16
// CHECK:        }
// CHECK:        gpu.barrier
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D6:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D7:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[D8:.+]] = affine.apply #[[MAP2]]()[%[[D5]], %[[D6]], %[[D7]]]
// CHECK:        gpu.barrier
// CHECK:        gpu.barrier
// CHECK:        gpu.barrier
// CHECK-DAG:    %[[CST_5:.+]] = arith.constant dense<0.000000e+00> : vector<2x4x4x2xf16>
// CHECK-DAG:    %[[D9:.+]] = affine.apply #[[MAP3]](%[[C0]], %[[C0]], %[[C0]])
// CHECK-DAG:    %[[D10:.+]] = affine.apply #[[MAP4]](%[[C0]], %[[C0]], %[[C0]])
// CHECK-DAG:    %[[D11:.+]] = affine.apply #[[MAP5]](%[[D5]], %[[D6]])
// CHECK-DAG:    %[[D12:.+]] = affine.apply #[[MAP3]](%[[C0]], %[[D11]], %[[C0]])
// CHECK-DAG:    %[[D13:.+]] = affine.apply #[[MAP4]](%[[C0]], %[[D11]], %[[C0]])
// CHECK:        %[[D14:.+]] = arith.addi %[[D9]], %[[D8]] : index
// CHECK:        %[[D15:.+]] = arith.addi %[[D12]], %[[D14]] : index
// CHECK:        %[[D16:.+]] = arith.addi %[[D10]], %[[C0]] : index
// CHECK:        %[[D17:.+]] = arith.addi %[[D13]], %[[D16]] : index
// CHECK:        %[[D18:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D15]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D19:.+]] = vector.insert_strided_slice %[[D18]], %[[CST_5]] {offsets = [0, 0, 0, 0], strides = [1,
// CHECK-SAME:     1]} : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK-DAG:    %[[D20:.+]] = affine.apply #[[MAP6]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:        %[[D21:.+]] = arith.addi %[[D20]], %[[C0]] : index
// CHECK:        %[[D22:.+]] = arith.addi %[[D13]], %[[D21]] : index
// CHECK:        %[[D23:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D15]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D24:.+]] = vector.insert_strided_slice %[[D23]], %[[D19]] {offsets = [0, 0, 2, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK-DAG:    %[[D25:.+]] = affine.apply #[[MAP7]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:        %[[D26:.+]] = arith.addi %[[D25]], %[[D8]] : index
// CHECK:        %[[D27:.+]] = arith.addi %[[D12]], %[[D26]] : index
// CHECK:        %[[D28:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D27]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D29:.+]] = vector.insert_strided_slice %[[D28]], %[[D24]] {offsets = [0, 0, 1, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D30:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D27]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D31:.+]] = vector.insert_strided_slice %[[D30]], %[[D29]] {offsets = [0, 0, 3, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK-DAG:    %[[D32:.+]] = affine.apply #[[MAP8]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:        %[[D33:.+]] = arith.addi %[[D32]], %[[C0]] : index
// CHECK:        %[[D34:.+]] = arith.addi %[[D13]], %[[D33]] : index
// CHECK:        %[[D35:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D15]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D36:.+]] = vector.insert_strided_slice %[[D35]], %[[D31]] {offsets = [0, 1, 0, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK-DAG:    %[[D37:.+]] = affine.apply #[[MAP9]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:        %[[D38:.+]] = arith.addi %[[D37]], %[[C0]] : index
// CHECK:        %[[D39:.+]] = arith.addi %[[D13]], %[[D38]] : index
// CHECK:        %[[D40:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D15]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D41:.+]] = vector.insert_strided_slice %[[D40]], %[[D36]] {offsets = [0, 1, 2, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D42:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D27]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D43:.+]] = vector.insert_strided_slice %[[D42]], %[[D41]] {offsets = [0, 1, 1, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D44:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D27]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D45:.+]] = vector.insert_strided_slice %[[D44]], %[[D43]] {offsets = [0, 1, 3, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK-DAG:    %[[D46:.+]] = affine.apply #[[MAP10]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:        %[[D47:.+]] = arith.addi %[[D46]], %[[C0]] : index
// CHECK:        %[[D48:.+]] = arith.addi %[[D13]], %[[D47]] : index
// CHECK:        %[[D49:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D15]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D50:.+]] = vector.insert_strided_slice %[[D49]], %[[D45]] {offsets = [0, 2, 0, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK-DAG:    %[[D51:.+]] = affine.apply #[[MAP11]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:        %[[D52:.+]] = arith.addi %[[D51]], %[[C0]] : index
// CHECK:        %[[D53:.+]] = arith.addi %[[D13]], %[[D52]] : index
// CHECK:        %[[D54:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D15]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D55:.+]] = vector.insert_strided_slice %[[D54]], %[[D50]] {offsets = [0, 2, 2, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D56:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D27]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D57:.+]] = vector.insert_strided_slice %[[D56]], %[[D55]] {offsets = [0, 2, 1, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D58:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D27]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D59:.+]] = vector.insert_strided_slice %[[D58]], %[[D57]] {offsets = [0, 2, 3, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK-DAG:    %[[D60:.+]] = affine.apply #[[MAP12]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:        %[[D61:.+]] = arith.addi %[[D60]], %[[C0]] : index
// CHECK:        %[[D62:.+]] = arith.addi %[[D13]], %[[D61]] : index
// CHECK:        %[[D63:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D15]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D64:.+]] = vector.insert_strided_slice %[[D63]], %[[D59]] {offsets = [0, 3, 0, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK-DAG:    %[[D65:.+]] = affine.apply #[[MAP13]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:        %[[D66:.+]] = arith.addi %[[D65]], %[[C0]] : index
// CHECK:        %[[D67:.+]] = arith.addi %[[D13]], %[[D66]] : index
// CHECK:        %[[D68:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D15]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D69:.+]] = vector.insert_strided_slice %[[D68]], %[[D64]] {offsets = [0, 3, 2, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D70:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D27]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D71:.+]] = vector.insert_strided_slice %[[D70]], %[[D69]] {offsets = [0, 3, 1, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D72:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D27]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D73:.+]] = vector.insert_strided_slice %[[D72]], %[[D71]] {offsets = [0, 3, 3, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK-DAG:    %[[D74:.+]] = affine.apply #[[MAP14]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:        %[[D75:.+]] = arith.addi %[[D74]], %[[D8]] : index
// CHECK:        %[[D76:.+]] = arith.addi %[[D12]], %[[D75]] : index
// CHECK:        %[[D77:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D76]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D78:.+]] = vector.insert_strided_slice %[[D77]], %[[D73]] {offsets = [1, 0, 0, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D79:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D76]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D80:.+]] = vector.insert_strided_slice %[[D79]], %[[D78]] {offsets = [1, 0, 2, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK-DAG:    %[[D81:.+]] = affine.apply #[[MAP15]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:        %[[D82:.+]] = arith.addi %[[D81]], %[[D8]] : index
// CHECK:        %[[D83:.+]] = arith.addi %[[D12]], %[[D82]] : index
// CHECK:        %[[D84:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D83]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D85:.+]] = vector.insert_strided_slice %[[D84]], %[[D80]] {offsets = [1, 0, 1, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D86:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D83]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D87:.+]] = vector.insert_strided_slice %[[D86]], %[[D85]] {offsets = [1, 0, 3, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D88:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D76]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D89:.+]] = vector.insert_strided_slice %[[D88]], %[[D87]] {offsets = [1, 1, 0, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D90:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D76]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D91:.+]] = vector.insert_strided_slice %[[D90]], %[[D89]] {offsets = [1, 1, 2, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D92:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D83]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D93:.+]] = vector.insert_strided_slice %[[D92]], %[[D91]] {offsets = [1, 1, 1, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D94:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D83]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D95:.+]] = vector.insert_strided_slice %[[D94]], %[[D93]] {offsets = [1, 1, 3, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D96:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D76]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D97:.+]] = vector.insert_strided_slice %[[D96]], %[[D95]] {offsets = [1, 2, 0, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D98:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D76]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D99:.+]] = vector.insert_strided_slice %[[D98]], %[[D97]] {offsets = [1, 2, 2, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D100:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D83]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D101:.+]] = vector.insert_strided_slice %[[D100]], %[[D99]] {offsets = [1, 2, 1, 0], strides = [1,
// CHECK-SAME:     1]} : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D102:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D83]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D103:.+]] = vector.insert_strided_slice %[[D102]], %[[D101]] {offsets = [1, 2, 3, 0], strides = [1,
// CHECK-SAME:     1]} : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D104:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D76]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D105:.+]] = vector.insert_strided_slice %[[D104]], %[[D103]] {offsets = [1, 3, 0, 0], strides = [1,
// CHECK-SAME:     1]} : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D106:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D76]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D107:.+]] = vector.insert_strided_slice %[[D106]], %[[D105]] {offsets = [1, 3, 2, 0], strides = [1,
// CHECK-SAME:     1]} : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D108:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D83]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D109:.+]] = vector.insert_strided_slice %[[D108]], %[[D107]] {offsets = [1, 3, 1, 0], strides = [1,
// CHECK-SAME:     1]} : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK:        %[[D110:.+]] = nvgpu.ldmatrix %[[ALLOC]][%[[C0]], %[[D83]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:     false} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:        %[[D111:.+]] = vector.insert_strided_slice %[[D110]], %[[D109]] {offsets = [1, 3, 3, 0], strides = [1,
// CHECK-SAME:     1]} : vector<1x2xf16> into vector<2x4x4x2xf16>
// CHECK-DAG:    %[[CST_6:.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:    %[[CST_7:.+]] = arith.constant dense<0.000000e+00> : vector<32x64xf32>
// CHECK:        %[[D112:.+]]:6 = scf.for %[[ARG0:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C128]]
// CHECK-SAME:     iter_args(%[[ARG1:[a-zA-Z0-9_]+]] = %[[CST_6]], %[[ARG2:[a-zA-Z0-9_]+]] = %[[CST_6]],
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]] = %[[CST_7]], %[[ARG4:[a-zA-Z0-9_]+]] = %[[CST_0]], %[[ARG5:[a-zA-Z0-9_]+]] =
// CHECK-SAME:     %[[CST_1]], %[[ARG6:[a-zA-Z0-9_]+]] = %[[CST]]) -> (vector<32xf32>, vector<32xf32>,
// CHECK-SAME:     vector<32x64xf32>, vector<2x16x2x2xf32>, vector<2x16x2x2xf32>, vector<2x8x2x2xf32>) {
// CHECK:          %[[SUBVIEW_8:.+]] = memref.subview %[[D1]][%[[WORKGROUP_ID_X]], %[[ARG0]], 0] [1, 128, 64] [1, 1, 1]
// CHECK-SAME:       : memref<20x1024x64xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
// CHECK:          %[[SUBVIEW_9:.+]] = memref.subview %[[D2]][%[[WORKGROUP_ID_X]], %[[ARG0]], 0] [1, 128, 64] [1, 1, 1]
// CHECK-SAME:       : memref<20x1024x64xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
// CHECK:          %[[ALLOC_10:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x64xf16,
// CHECK-SAME:       #[[GPU]].address_space<workgroup>>
// CHECK:          gpu.barrier
// CHECK:          linalg.generic {indexing_maps = [#[[MAP16]], #[[MAP16]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:       ins(%[[SUBVIEW_8]] : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%[[ALLOC_10]] :
// CHECK-SAME:       memref<128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:          ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:            linalg.yield %[[IN]] : f16
// CHECK:          }
// CHECK:          gpu.barrier
// CHECK:          %[[ALLOC_11:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x64xf16,
// CHECK-SAME:       #[[GPU]].address_space<workgroup>>
// CHECK:          gpu.barrier
// CHECK:          linalg.generic {indexing_maps = [#[[MAP16]], #[[MAP16]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:       ins(%[[SUBVIEW_9]] : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%[[ALLOC_11]] :
// CHECK-SAME:       memref<128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:          ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:            linalg.yield %[[IN]] : f16
// CHECK:          }
// CHECK:          gpu.barrier
// CHECK-DAG:      %[[CST_12:.+]] = arith.constant dense<0.000000e+00> : vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D218:.+]] = affine.apply #[[MAP17]](%[[C0]], %[[C0]], %[[C0]])
// CHECK-DAG:      %[[D219:.+]] = affine.apply #[[MAP17]](%[[C0]], %[[D11]], %[[C0]])
// CHECK:          %[[D220:.+]] = arith.addi %[[D218]], %[[C0]] : index
// CHECK:          %[[D221:.+]] = arith.addi %[[D219]], %[[D220]] : index
// CHECK:          %[[D222:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D221]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D223:.+]] = vector.insert_strided_slice %[[D222]], %[[CST_12]] {offsets = [0, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D224:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D221]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D225:.+]] = vector.insert_strided_slice %[[D224]], %[[D223]] {offsets = [0, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D226:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D221]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D227:.+]] = vector.insert_strided_slice %[[D226]], %[[D225]] {offsets = [0, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D228:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D221]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D229:.+]] = vector.insert_strided_slice %[[D228]], %[[D227]] {offsets = [0, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D230:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D221]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D231:.+]] = vector.insert_strided_slice %[[D230]], %[[D229]] {offsets = [0, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D232:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D221]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D233:.+]] = vector.insert_strided_slice %[[D232]], %[[D231]] {offsets = [0, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D234:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D221]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D235:.+]] = vector.insert_strided_slice %[[D234]], %[[D233]] {offsets = [0, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D236:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D221]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D237:.+]] = vector.insert_strided_slice %[[D236]], %[[D235]] {offsets = [0, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D238:.+]] = affine.apply #[[MAP18]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D239:.+]] = arith.addi %[[D238]], %[[C0]] : index
// CHECK:          %[[D240:.+]] = arith.addi %[[D219]], %[[D239]] : index
// CHECK:          %[[D241:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D240]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D242:.+]] = vector.insert_strided_slice %[[D241]], %[[D237]] {offsets = [1, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D243:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D240]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D244:.+]] = vector.insert_strided_slice %[[D243]], %[[D242]] {offsets = [1, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D245:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D240]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D246:.+]] = vector.insert_strided_slice %[[D245]], %[[D244]] {offsets = [1, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D247:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D240]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D248:.+]] = vector.insert_strided_slice %[[D247]], %[[D246]] {offsets = [1, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D249:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D240]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D250:.+]] = vector.insert_strided_slice %[[D249]], %[[D248]] {offsets = [1, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D251:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D240]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D252:.+]] = vector.insert_strided_slice %[[D251]], %[[D250]] {offsets = [1, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D253:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D240]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D254:.+]] = vector.insert_strided_slice %[[D253]], %[[D252]] {offsets = [1, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D255:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D240]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D256:.+]] = vector.insert_strided_slice %[[D255]], %[[D254]] {offsets = [1, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D257:.+]] = affine.apply #[[MAP19]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D258:.+]] = arith.addi %[[D257]], %[[C0]] : index
// CHECK:          %[[D259:.+]] = arith.addi %[[D219]], %[[D258]] : index
// CHECK:          %[[D260:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D259]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D261:.+]] = vector.insert_strided_slice %[[D260]], %[[D256]] {offsets = [2, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D262:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D259]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D263:.+]] = vector.insert_strided_slice %[[D262]], %[[D261]] {offsets = [2, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D264:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D259]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D265:.+]] = vector.insert_strided_slice %[[D264]], %[[D263]] {offsets = [2, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D266:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D259]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D267:.+]] = vector.insert_strided_slice %[[D266]], %[[D265]] {offsets = [2, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D268:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D259]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D269:.+]] = vector.insert_strided_slice %[[D268]], %[[D267]] {offsets = [2, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D270:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D259]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D271:.+]] = vector.insert_strided_slice %[[D270]], %[[D269]] {offsets = [2, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D272:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D259]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D273:.+]] = vector.insert_strided_slice %[[D272]], %[[D271]] {offsets = [2, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D274:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D259]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D275:.+]] = vector.insert_strided_slice %[[D274]], %[[D273]] {offsets = [2, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D276:.+]] = affine.apply #[[MAP20]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D277:.+]] = arith.addi %[[D276]], %[[C0]] : index
// CHECK:          %[[D278:.+]] = arith.addi %[[D219]], %[[D277]] : index
// CHECK:          %[[D279:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D278]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D280:.+]] = vector.insert_strided_slice %[[D279]], %[[D275]] {offsets = [3, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D281:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D278]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D282:.+]] = vector.insert_strided_slice %[[D281]], %[[D280]] {offsets = [3, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D283:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D278]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D284:.+]] = vector.insert_strided_slice %[[D283]], %[[D282]] {offsets = [3, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D285:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D278]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D286:.+]] = vector.insert_strided_slice %[[D285]], %[[D284]] {offsets = [3, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D287:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D278]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D288:.+]] = vector.insert_strided_slice %[[D287]], %[[D286]] {offsets = [3, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D289:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D278]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D290:.+]] = vector.insert_strided_slice %[[D289]], %[[D288]] {offsets = [3, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D291:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D278]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D292:.+]] = vector.insert_strided_slice %[[D291]], %[[D290]] {offsets = [3, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D293:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D278]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D294:.+]] = vector.insert_strided_slice %[[D293]], %[[D292]] {offsets = [3, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D295:.+]] = affine.apply #[[MAP21]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D296:.+]] = arith.addi %[[D295]], %[[C0]] : index
// CHECK:          %[[D297:.+]] = arith.addi %[[D219]], %[[D296]] : index
// CHECK:          %[[D298:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D297]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D299:.+]] = vector.insert_strided_slice %[[D298]], %[[D294]] {offsets = [4, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D300:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D297]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D301:.+]] = vector.insert_strided_slice %[[D300]], %[[D299]] {offsets = [4, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D302:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D297]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D303:.+]] = vector.insert_strided_slice %[[D302]], %[[D301]] {offsets = [4, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D304:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D297]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D305:.+]] = vector.insert_strided_slice %[[D304]], %[[D303]] {offsets = [4, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D306:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D297]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D307:.+]] = vector.insert_strided_slice %[[D306]], %[[D305]] {offsets = [4, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D308:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D297]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D309:.+]] = vector.insert_strided_slice %[[D308]], %[[D307]] {offsets = [4, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D310:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D297]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D311:.+]] = vector.insert_strided_slice %[[D310]], %[[D309]] {offsets = [4, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D312:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D297]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D313:.+]] = vector.insert_strided_slice %[[D312]], %[[D311]] {offsets = [4, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D314:.+]] = affine.apply #[[MAP22]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D315:.+]] = arith.addi %[[D314]], %[[C0]] : index
// CHECK:          %[[D316:.+]] = arith.addi %[[D219]], %[[D315]] : index
// CHECK:          %[[D317:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D316]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D318:.+]] = vector.insert_strided_slice %[[D317]], %[[D313]] {offsets = [5, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D319:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D316]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D320:.+]] = vector.insert_strided_slice %[[D319]], %[[D318]] {offsets = [5, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D321:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D316]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D322:.+]] = vector.insert_strided_slice %[[D321]], %[[D320]] {offsets = [5, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D323:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D316]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D324:.+]] = vector.insert_strided_slice %[[D323]], %[[D322]] {offsets = [5, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D325:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D316]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D326:.+]] = vector.insert_strided_slice %[[D325]], %[[D324]] {offsets = [5, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D327:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D316]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D328:.+]] = vector.insert_strided_slice %[[D327]], %[[D326]] {offsets = [5, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D329:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D316]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D330:.+]] = vector.insert_strided_slice %[[D329]], %[[D328]] {offsets = [5, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D331:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D316]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D332:.+]] = vector.insert_strided_slice %[[D331]], %[[D330]] {offsets = [5, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D333:.+]] = affine.apply #[[MAP23]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D334:.+]] = arith.addi %[[D333]], %[[C0]] : index
// CHECK:          %[[D335:.+]] = arith.addi %[[D219]], %[[D334]] : index
// CHECK:          %[[D336:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D335]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D337:.+]] = vector.insert_strided_slice %[[D336]], %[[D332]] {offsets = [6, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D338:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D335]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D339:.+]] = vector.insert_strided_slice %[[D338]], %[[D337]] {offsets = [6, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D340:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D335]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D341:.+]] = vector.insert_strided_slice %[[D340]], %[[D339]] {offsets = [6, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D342:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D335]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D343:.+]] = vector.insert_strided_slice %[[D342]], %[[D341]] {offsets = [6, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D344:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D335]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D345:.+]] = vector.insert_strided_slice %[[D344]], %[[D343]] {offsets = [6, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D346:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D335]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D347:.+]] = vector.insert_strided_slice %[[D346]], %[[D345]] {offsets = [6, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D348:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D335]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D349:.+]] = vector.insert_strided_slice %[[D348]], %[[D347]] {offsets = [6, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D350:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D335]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D351:.+]] = vector.insert_strided_slice %[[D350]], %[[D349]] {offsets = [6, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D352:.+]] = affine.apply #[[MAP24]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D353:.+]] = arith.addi %[[D352]], %[[C0]] : index
// CHECK:          %[[D354:.+]] = arith.addi %[[D219]], %[[D353]] : index
// CHECK:          %[[D355:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D354]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D356:.+]] = vector.insert_strided_slice %[[D355]], %[[D351]] {offsets = [7, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D357:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D354]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D358:.+]] = vector.insert_strided_slice %[[D357]], %[[D356]] {offsets = [7, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D359:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D354]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D360:.+]] = vector.insert_strided_slice %[[D359]], %[[D358]] {offsets = [7, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D361:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D354]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D362:.+]] = vector.insert_strided_slice %[[D361]], %[[D360]] {offsets = [7, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D363:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D354]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D364:.+]] = vector.insert_strided_slice %[[D363]], %[[D362]] {offsets = [7, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D365:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D354]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D366:.+]] = vector.insert_strided_slice %[[D365]], %[[D364]] {offsets = [7, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D367:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D354]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D368:.+]] = vector.insert_strided_slice %[[D367]], %[[D366]] {offsets = [7, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D369:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D354]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D370:.+]] = vector.insert_strided_slice %[[D369]], %[[D368]] {offsets = [7, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D371:.+]] = affine.apply #[[MAP25]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D372:.+]] = arith.addi %[[D371]], %[[C0]] : index
// CHECK:          %[[D373:.+]] = arith.addi %[[D219]], %[[D372]] : index
// CHECK:          %[[D374:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D373]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D375:.+]] = vector.insert_strided_slice %[[D374]], %[[D370]] {offsets = [8, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D376:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D373]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D377:.+]] = vector.insert_strided_slice %[[D376]], %[[D375]] {offsets = [8, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D378:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D373]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D379:.+]] = vector.insert_strided_slice %[[D378]], %[[D377]] {offsets = [8, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D380:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D373]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D381:.+]] = vector.insert_strided_slice %[[D380]], %[[D379]] {offsets = [8, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D382:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D373]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D383:.+]] = vector.insert_strided_slice %[[D382]], %[[D381]] {offsets = [8, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D384:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D373]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D385:.+]] = vector.insert_strided_slice %[[D384]], %[[D383]] {offsets = [8, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D386:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D373]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D387:.+]] = vector.insert_strided_slice %[[D386]], %[[D385]] {offsets = [8, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D388:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D373]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D389:.+]] = vector.insert_strided_slice %[[D388]], %[[D387]] {offsets = [8, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D390:.+]] = affine.apply #[[MAP26]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D391:.+]] = arith.addi %[[D390]], %[[C0]] : index
// CHECK:          %[[D392:.+]] = arith.addi %[[D219]], %[[D391]] : index
// CHECK:          %[[D393:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D392]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D394:.+]] = vector.insert_strided_slice %[[D393]], %[[D389]] {offsets = [9, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D395:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D392]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D396:.+]] = vector.insert_strided_slice %[[D395]], %[[D394]] {offsets = [9, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D397:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D392]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D398:.+]] = vector.insert_strided_slice %[[D397]], %[[D396]] {offsets = [9, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D399:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D392]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D400:.+]] = vector.insert_strided_slice %[[D399]], %[[D398]] {offsets = [9, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D401:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D392]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D402:.+]] = vector.insert_strided_slice %[[D401]], %[[D400]] {offsets = [9, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D403:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D392]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D404:.+]] = vector.insert_strided_slice %[[D403]], %[[D402]] {offsets = [9, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D405:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D392]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D406:.+]] = vector.insert_strided_slice %[[D405]], %[[D404]] {offsets = [9, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D407:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D392]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D408:.+]] = vector.insert_strided_slice %[[D407]], %[[D406]] {offsets = [9, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D409:.+]] = affine.apply #[[MAP27]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D410:.+]] = arith.addi %[[D409]], %[[C0]] : index
// CHECK:          %[[D411:.+]] = arith.addi %[[D219]], %[[D410]] : index
// CHECK:          %[[D412:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D411]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D413:.+]] = vector.insert_strided_slice %[[D412]], %[[D408]] {offsets = [10, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D414:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D411]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D415:.+]] = vector.insert_strided_slice %[[D414]], %[[D413]] {offsets = [10, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D416:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D411]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D417:.+]] = vector.insert_strided_slice %[[D416]], %[[D415]] {offsets = [10, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D418:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D411]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D419:.+]] = vector.insert_strided_slice %[[D418]], %[[D417]] {offsets = [10, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D420:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D411]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D421:.+]] = vector.insert_strided_slice %[[D420]], %[[D419]] {offsets = [10, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D422:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D411]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D423:.+]] = vector.insert_strided_slice %[[D422]], %[[D421]] {offsets = [10, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D424:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D411]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D425:.+]] = vector.insert_strided_slice %[[D424]], %[[D423]] {offsets = [10, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D426:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D411]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D427:.+]] = vector.insert_strided_slice %[[D426]], %[[D425]] {offsets = [10, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D428:.+]] = affine.apply #[[MAP28]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D429:.+]] = arith.addi %[[D428]], %[[C0]] : index
// CHECK:          %[[D430:.+]] = arith.addi %[[D219]], %[[D429]] : index
// CHECK:          %[[D431:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D430]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D432:.+]] = vector.insert_strided_slice %[[D431]], %[[D427]] {offsets = [11, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D433:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D430]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D434:.+]] = vector.insert_strided_slice %[[D433]], %[[D432]] {offsets = [11, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D435:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D430]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D436:.+]] = vector.insert_strided_slice %[[D435]], %[[D434]] {offsets = [11, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D437:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D430]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D438:.+]] = vector.insert_strided_slice %[[D437]], %[[D436]] {offsets = [11, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D439:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D430]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D440:.+]] = vector.insert_strided_slice %[[D439]], %[[D438]] {offsets = [11, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D441:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D430]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D442:.+]] = vector.insert_strided_slice %[[D441]], %[[D440]] {offsets = [11, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D443:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D430]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D444:.+]] = vector.insert_strided_slice %[[D443]], %[[D442]] {offsets = [11, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D445:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D430]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D446:.+]] = vector.insert_strided_slice %[[D445]], %[[D444]] {offsets = [11, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D447:.+]] = affine.apply #[[MAP29]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D448:.+]] = arith.addi %[[D447]], %[[C0]] : index
// CHECK:          %[[D449:.+]] = arith.addi %[[D219]], %[[D448]] : index
// CHECK:          %[[D450:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D449]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D451:.+]] = vector.insert_strided_slice %[[D450]], %[[D446]] {offsets = [12, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D452:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D449]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D453:.+]] = vector.insert_strided_slice %[[D452]], %[[D451]] {offsets = [12, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D454:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D449]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D455:.+]] = vector.insert_strided_slice %[[D454]], %[[D453]] {offsets = [12, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D456:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D449]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D457:.+]] = vector.insert_strided_slice %[[D456]], %[[D455]] {offsets = [12, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D458:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D449]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D459:.+]] = vector.insert_strided_slice %[[D458]], %[[D457]] {offsets = [12, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D460:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D449]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D461:.+]] = vector.insert_strided_slice %[[D460]], %[[D459]] {offsets = [12, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D462:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D449]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D463:.+]] = vector.insert_strided_slice %[[D462]], %[[D461]] {offsets = [12, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D464:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D449]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D465:.+]] = vector.insert_strided_slice %[[D464]], %[[D463]] {offsets = [12, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D466:.+]] = affine.apply #[[MAP30]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D467:.+]] = arith.addi %[[D466]], %[[C0]] : index
// CHECK:          %[[D468:.+]] = arith.addi %[[D219]], %[[D467]] : index
// CHECK:          %[[D469:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D468]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D470:.+]] = vector.insert_strided_slice %[[D469]], %[[D465]] {offsets = [13, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D471:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D468]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D472:.+]] = vector.insert_strided_slice %[[D471]], %[[D470]] {offsets = [13, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D473:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D468]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D474:.+]] = vector.insert_strided_slice %[[D473]], %[[D472]] {offsets = [13, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D475:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D468]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D476:.+]] = vector.insert_strided_slice %[[D475]], %[[D474]] {offsets = [13, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D477:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D468]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D478:.+]] = vector.insert_strided_slice %[[D477]], %[[D476]] {offsets = [13, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D479:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D468]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D480:.+]] = vector.insert_strided_slice %[[D479]], %[[D478]] {offsets = [13, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D481:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D468]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D482:.+]] = vector.insert_strided_slice %[[D481]], %[[D480]] {offsets = [13, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D483:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D468]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D484:.+]] = vector.insert_strided_slice %[[D483]], %[[D482]] {offsets = [13, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D485:.+]] = affine.apply #[[MAP31]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D486:.+]] = arith.addi %[[D485]], %[[C0]] : index
// CHECK:          %[[D487:.+]] = arith.addi %[[D219]], %[[D486]] : index
// CHECK:          %[[D488:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D487]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D489:.+]] = vector.insert_strided_slice %[[D488]], %[[D484]] {offsets = [14, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D490:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D487]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D491:.+]] = vector.insert_strided_slice %[[D490]], %[[D489]] {offsets = [14, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D492:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D487]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D493:.+]] = vector.insert_strided_slice %[[D492]], %[[D491]] {offsets = [14, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D494:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D487]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D495:.+]] = vector.insert_strided_slice %[[D494]], %[[D493]] {offsets = [14, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D496:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D487]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D497:.+]] = vector.insert_strided_slice %[[D496]], %[[D495]] {offsets = [14, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D498:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D487]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D499:.+]] = vector.insert_strided_slice %[[D498]], %[[D497]] {offsets = [14, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D500:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D487]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D501:.+]] = vector.insert_strided_slice %[[D500]], %[[D499]] {offsets = [14, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D502:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D487]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D503:.+]] = vector.insert_strided_slice %[[D502]], %[[D501]] {offsets = [14, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK-DAG:      %[[D504:.+]] = affine.apply #[[MAP32]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D505:.+]] = arith.addi %[[D504]], %[[C0]] : index
// CHECK:          %[[D506:.+]] = arith.addi %[[D219]], %[[D505]] : index
// CHECK:          %[[D507:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D506]], %[[D17]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D508:.+]] = vector.insert_strided_slice %[[D507]], %[[D503]] {offsets = [15, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D509:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D506]], %[[D22]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D510:.+]] = vector.insert_strided_slice %[[D509]], %[[D508]] {offsets = [15, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D511:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D506]], %[[D34]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D512:.+]] = vector.insert_strided_slice %[[D511]], %[[D510]] {offsets = [15, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D513:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D506]], %[[D39]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D514:.+]] = vector.insert_strided_slice %[[D513]], %[[D512]] {offsets = [15, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D515:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D506]], %[[D48]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D516:.+]] = vector.insert_strided_slice %[[D515]], %[[D514]] {offsets = [15, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D517:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D506]], %[[D53]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D518:.+]] = vector.insert_strided_slice %[[D517]], %[[D516]] {offsets = [15, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D519:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D506]], %[[D62]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D520:.+]] = vector.insert_strided_slice %[[D519]], %[[D518]] {offsets = [15, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D521:.+]] = nvgpu.ldmatrix %[[ALLOC_10]][%[[D506]], %[[D67]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       false} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D522:.+]] = vector.insert_strided_slice %[[D521]], %[[D520]] {offsets = [15, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<16x4x2x2xf16>
// CHECK:          %[[D523:.+]] = vector.extract %[[CST_1]][0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D524:.+]] = vector.extract %[[D111]][0, 0] : vector<2x4x4x2xf16>
// CHECK:          %[[D525:.+]] = vector.extract %[[D522]][0, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D526:.+]] = nvgpu.mma.sync(%[[D524]], %[[D525]], %[[D523]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D527:.+]] = vector.extract %[[D111]][0, 1] : vector<2x4x4x2xf16>
// CHECK:          %[[D528:.+]] = vector.extract %[[D522]][0, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D529:.+]] = nvgpu.mma.sync(%[[D527]], %[[D528]], %[[D526]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D530:.+]] = vector.extract %[[D111]][0, 2] : vector<2x4x4x2xf16>
// CHECK:          %[[D531:.+]] = vector.extract %[[D522]][0, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D532:.+]] = nvgpu.mma.sync(%[[D530]], %[[D531]], %[[D529]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D533:.+]] = vector.extract %[[D111]][0, 3] : vector<2x4x4x2xf16>
// CHECK:          %[[D534:.+]] = vector.extract %[[D522]][0, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D535:.+]] = nvgpu.mma.sync(%[[D533]], %[[D534]], %[[D532]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D536:.+]] = vector.insert %[[D535]], %[[CST_1]] [0, 0] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D537:.+]] = vector.extract %[[CST_1]][0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D538:.+]] = vector.extract %[[D522]][1, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D539:.+]] = nvgpu.mma.sync(%[[D524]], %[[D538]], %[[D537]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D540:.+]] = vector.extract %[[D522]][1, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D541:.+]] = nvgpu.mma.sync(%[[D527]], %[[D540]], %[[D539]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D542:.+]] = vector.extract %[[D522]][1, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D543:.+]] = nvgpu.mma.sync(%[[D530]], %[[D542]], %[[D541]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D544:.+]] = vector.extract %[[D522]][1, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D545:.+]] = nvgpu.mma.sync(%[[D533]], %[[D544]], %[[D543]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D546:.+]] = vector.insert %[[D545]], %[[D536]] [0, 1] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D547:.+]] = vector.extract %[[CST_1]][0, 2] : vector<2x16x2x2xf32>
// CHECK:          %[[D548:.+]] = vector.extract %[[D522]][2, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D549:.+]] = nvgpu.mma.sync(%[[D524]], %[[D548]], %[[D547]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D550:.+]] = vector.extract %[[D522]][2, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D551:.+]] = nvgpu.mma.sync(%[[D527]], %[[D550]], %[[D549]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D552:.+]] = vector.extract %[[D522]][2, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D553:.+]] = nvgpu.mma.sync(%[[D530]], %[[D552]], %[[D551]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D554:.+]] = vector.extract %[[D522]][2, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D555:.+]] = nvgpu.mma.sync(%[[D533]], %[[D554]], %[[D553]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D556:.+]] = vector.insert %[[D555]], %[[D546]] [0, 2] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D557:.+]] = vector.extract %[[CST_1]][0, 3] : vector<2x16x2x2xf32>
// CHECK:          %[[D558:.+]] = vector.extract %[[D522]][3, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D559:.+]] = nvgpu.mma.sync(%[[D524]], %[[D558]], %[[D557]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D560:.+]] = vector.extract %[[D522]][3, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D561:.+]] = nvgpu.mma.sync(%[[D527]], %[[D560]], %[[D559]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D562:.+]] = vector.extract %[[D522]][3, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D563:.+]] = nvgpu.mma.sync(%[[D530]], %[[D562]], %[[D561]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D564:.+]] = vector.extract %[[D522]][3, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D565:.+]] = nvgpu.mma.sync(%[[D533]], %[[D564]], %[[D563]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D566:.+]] = vector.insert %[[D565]], %[[D556]] [0, 3] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D567:.+]] = vector.extract %[[CST_1]][0, 4] : vector<2x16x2x2xf32>
// CHECK:          %[[D568:.+]] = vector.extract %[[D522]][4, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D569:.+]] = nvgpu.mma.sync(%[[D524]], %[[D568]], %[[D567]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D570:.+]] = vector.extract %[[D522]][4, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D571:.+]] = nvgpu.mma.sync(%[[D527]], %[[D570]], %[[D569]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D572:.+]] = vector.extract %[[D522]][4, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D573:.+]] = nvgpu.mma.sync(%[[D530]], %[[D572]], %[[D571]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D574:.+]] = vector.extract %[[D522]][4, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D575:.+]] = nvgpu.mma.sync(%[[D533]], %[[D574]], %[[D573]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D576:.+]] = vector.insert %[[D575]], %[[D566]] [0, 4] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D577:.+]] = vector.extract %[[CST_1]][0, 5] : vector<2x16x2x2xf32>
// CHECK:          %[[D578:.+]] = vector.extract %[[D522]][5, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D579:.+]] = nvgpu.mma.sync(%[[D524]], %[[D578]], %[[D577]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D580:.+]] = vector.extract %[[D522]][5, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D581:.+]] = nvgpu.mma.sync(%[[D527]], %[[D580]], %[[D579]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D582:.+]] = vector.extract %[[D522]][5, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D583:.+]] = nvgpu.mma.sync(%[[D530]], %[[D582]], %[[D581]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D584:.+]] = vector.extract %[[D522]][5, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D585:.+]] = nvgpu.mma.sync(%[[D533]], %[[D584]], %[[D583]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D586:.+]] = vector.insert %[[D585]], %[[D576]] [0, 5] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D587:.+]] = vector.extract %[[CST_1]][0, 6] : vector<2x16x2x2xf32>
// CHECK:          %[[D588:.+]] = vector.extract %[[D522]][6, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D589:.+]] = nvgpu.mma.sync(%[[D524]], %[[D588]], %[[D587]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D590:.+]] = vector.extract %[[D522]][6, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D591:.+]] = nvgpu.mma.sync(%[[D527]], %[[D590]], %[[D589]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D592:.+]] = vector.extract %[[D522]][6, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D593:.+]] = nvgpu.mma.sync(%[[D530]], %[[D592]], %[[D591]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D594:.+]] = vector.extract %[[D522]][6, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D595:.+]] = nvgpu.mma.sync(%[[D533]], %[[D594]], %[[D593]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D596:.+]] = vector.insert %[[D595]], %[[D586]] [0, 6] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D597:.+]] = vector.extract %[[CST_1]][0, 7] : vector<2x16x2x2xf32>
// CHECK:          %[[D598:.+]] = vector.extract %[[D522]][7, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D599:.+]] = nvgpu.mma.sync(%[[D524]], %[[D598]], %[[D597]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D600:.+]] = vector.extract %[[D522]][7, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D601:.+]] = nvgpu.mma.sync(%[[D527]], %[[D600]], %[[D599]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D602:.+]] = vector.extract %[[D522]][7, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D603:.+]] = nvgpu.mma.sync(%[[D530]], %[[D602]], %[[D601]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D604:.+]] = vector.extract %[[D522]][7, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D605:.+]] = nvgpu.mma.sync(%[[D533]], %[[D604]], %[[D603]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D606:.+]] = vector.insert %[[D605]], %[[D596]] [0, 7] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D607:.+]] = vector.extract %[[CST_1]][0, 8] : vector<2x16x2x2xf32>
// CHECK:          %[[D608:.+]] = vector.extract %[[D522]][8, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D609:.+]] = nvgpu.mma.sync(%[[D524]], %[[D608]], %[[D607]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D610:.+]] = vector.extract %[[D522]][8, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D611:.+]] = nvgpu.mma.sync(%[[D527]], %[[D610]], %[[D609]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D612:.+]] = vector.extract %[[D522]][8, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D613:.+]] = nvgpu.mma.sync(%[[D530]], %[[D612]], %[[D611]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D614:.+]] = vector.extract %[[D522]][8, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D615:.+]] = nvgpu.mma.sync(%[[D533]], %[[D614]], %[[D613]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D616:.+]] = vector.insert %[[D615]], %[[D606]] [0, 8] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D617:.+]] = vector.extract %[[CST_1]][0, 9] : vector<2x16x2x2xf32>
// CHECK:          %[[D618:.+]] = vector.extract %[[D522]][9, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D619:.+]] = nvgpu.mma.sync(%[[D524]], %[[D618]], %[[D617]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D620:.+]] = vector.extract %[[D522]][9, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D621:.+]] = nvgpu.mma.sync(%[[D527]], %[[D620]], %[[D619]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D622:.+]] = vector.extract %[[D522]][9, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D623:.+]] = nvgpu.mma.sync(%[[D530]], %[[D622]], %[[D621]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D624:.+]] = vector.extract %[[D522]][9, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D625:.+]] = nvgpu.mma.sync(%[[D533]], %[[D624]], %[[D623]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D626:.+]] = vector.insert %[[D625]], %[[D616]] [0, 9] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D627:.+]] = vector.extract %[[CST_1]][0, 10] : vector<2x16x2x2xf32>
// CHECK:          %[[D628:.+]] = vector.extract %[[D522]][10, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D629:.+]] = nvgpu.mma.sync(%[[D524]], %[[D628]], %[[D627]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D630:.+]] = vector.extract %[[D522]][10, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D631:.+]] = nvgpu.mma.sync(%[[D527]], %[[D630]], %[[D629]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D632:.+]] = vector.extract %[[D522]][10, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D633:.+]] = nvgpu.mma.sync(%[[D530]], %[[D632]], %[[D631]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D634:.+]] = vector.extract %[[D522]][10, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D635:.+]] = nvgpu.mma.sync(%[[D533]], %[[D634]], %[[D633]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D636:.+]] = vector.insert %[[D635]], %[[D626]] [0, 10] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D637:.+]] = vector.extract %[[CST_1]][0, 11] : vector<2x16x2x2xf32>
// CHECK:          %[[D638:.+]] = vector.extract %[[D522]][11, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D639:.+]] = nvgpu.mma.sync(%[[D524]], %[[D638]], %[[D637]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D640:.+]] = vector.extract %[[D522]][11, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D641:.+]] = nvgpu.mma.sync(%[[D527]], %[[D640]], %[[D639]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D642:.+]] = vector.extract %[[D522]][11, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D643:.+]] = nvgpu.mma.sync(%[[D530]], %[[D642]], %[[D641]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D644:.+]] = vector.extract %[[D522]][11, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D645:.+]] = nvgpu.mma.sync(%[[D533]], %[[D644]], %[[D643]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D646:.+]] = vector.insert %[[D645]], %[[D636]] [0, 11] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D647:.+]] = vector.extract %[[CST_1]][0, 12] : vector<2x16x2x2xf32>
// CHECK:          %[[D648:.+]] = vector.extract %[[D522]][12, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D649:.+]] = nvgpu.mma.sync(%[[D524]], %[[D648]], %[[D647]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D650:.+]] = vector.extract %[[D522]][12, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D651:.+]] = nvgpu.mma.sync(%[[D527]], %[[D650]], %[[D649]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D652:.+]] = vector.extract %[[D522]][12, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D653:.+]] = nvgpu.mma.sync(%[[D530]], %[[D652]], %[[D651]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D654:.+]] = vector.extract %[[D522]][12, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D655:.+]] = nvgpu.mma.sync(%[[D533]], %[[D654]], %[[D653]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D656:.+]] = vector.insert %[[D655]], %[[D646]] [0, 12] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D657:.+]] = vector.extract %[[CST_1]][0, 13] : vector<2x16x2x2xf32>
// CHECK:          %[[D658:.+]] = vector.extract %[[D522]][13, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D659:.+]] = nvgpu.mma.sync(%[[D524]], %[[D658]], %[[D657]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D660:.+]] = vector.extract %[[D522]][13, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D661:.+]] = nvgpu.mma.sync(%[[D527]], %[[D660]], %[[D659]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D662:.+]] = vector.extract %[[D522]][13, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D663:.+]] = nvgpu.mma.sync(%[[D530]], %[[D662]], %[[D661]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D664:.+]] = vector.extract %[[D522]][13, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D665:.+]] = nvgpu.mma.sync(%[[D533]], %[[D664]], %[[D663]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D666:.+]] = vector.insert %[[D665]], %[[D656]] [0, 13] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D667:.+]] = vector.extract %[[CST_1]][0, 14] : vector<2x16x2x2xf32>
// CHECK:          %[[D668:.+]] = vector.extract %[[D522]][14, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D669:.+]] = nvgpu.mma.sync(%[[D524]], %[[D668]], %[[D667]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D670:.+]] = vector.extract %[[D522]][14, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D671:.+]] = nvgpu.mma.sync(%[[D527]], %[[D670]], %[[D669]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D672:.+]] = vector.extract %[[D522]][14, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D673:.+]] = nvgpu.mma.sync(%[[D530]], %[[D672]], %[[D671]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D674:.+]] = vector.extract %[[D522]][14, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D675:.+]] = nvgpu.mma.sync(%[[D533]], %[[D674]], %[[D673]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D676:.+]] = vector.insert %[[D675]], %[[D666]] [0, 14] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D677:.+]] = vector.extract %[[CST_1]][0, 15] : vector<2x16x2x2xf32>
// CHECK:          %[[D678:.+]] = vector.extract %[[D522]][15, 0] : vector<16x4x2x2xf16>
// CHECK:          %[[D679:.+]] = nvgpu.mma.sync(%[[D524]], %[[D678]], %[[D677]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D680:.+]] = vector.extract %[[D522]][15, 1] : vector<16x4x2x2xf16>
// CHECK:          %[[D681:.+]] = nvgpu.mma.sync(%[[D527]], %[[D680]], %[[D679]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D682:.+]] = vector.extract %[[D522]][15, 2] : vector<16x4x2x2xf16>
// CHECK:          %[[D683:.+]] = nvgpu.mma.sync(%[[D530]], %[[D682]], %[[D681]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D684:.+]] = vector.extract %[[D522]][15, 3] : vector<16x4x2x2xf16>
// CHECK:          %[[D685:.+]] = nvgpu.mma.sync(%[[D533]], %[[D684]], %[[D683]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D686:.+]] = vector.insert %[[D685]], %[[D676]] [0, 15] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D687:.+]] = vector.extract %[[CST_1]][1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D688:.+]] = vector.extract %[[D111]][1, 0] : vector<2x4x4x2xf16>
// CHECK:          %[[D689:.+]] = nvgpu.mma.sync(%[[D688]], %[[D525]], %[[D687]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D690:.+]] = vector.extract %[[D111]][1, 1] : vector<2x4x4x2xf16>
// CHECK:          %[[D691:.+]] = nvgpu.mma.sync(%[[D690]], %[[D528]], %[[D689]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D692:.+]] = vector.extract %[[D111]][1, 2] : vector<2x4x4x2xf16>
// CHECK:          %[[D693:.+]] = nvgpu.mma.sync(%[[D692]], %[[D531]], %[[D691]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D694:.+]] = vector.extract %[[D111]][1, 3] : vector<2x4x4x2xf16>
// CHECK:          %[[D695:.+]] = nvgpu.mma.sync(%[[D694]], %[[D534]], %[[D693]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D696:.+]] = vector.insert %[[D695]], %[[D686]] [1, 0] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D697:.+]] = vector.extract %[[CST_1]][1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D698:.+]] = nvgpu.mma.sync(%[[D688]], %[[D538]], %[[D697]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D699:.+]] = nvgpu.mma.sync(%[[D690]], %[[D540]], %[[D698]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D700:.+]] = nvgpu.mma.sync(%[[D692]], %[[D542]], %[[D699]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D701:.+]] = nvgpu.mma.sync(%[[D694]], %[[D544]], %[[D700]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D702:.+]] = vector.insert %[[D701]], %[[D696]] [1, 1] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D703:.+]] = vector.extract %[[CST_1]][1, 2] : vector<2x16x2x2xf32>
// CHECK:          %[[D704:.+]] = nvgpu.mma.sync(%[[D688]], %[[D548]], %[[D703]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D705:.+]] = nvgpu.mma.sync(%[[D690]], %[[D550]], %[[D704]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D706:.+]] = nvgpu.mma.sync(%[[D692]], %[[D552]], %[[D705]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D707:.+]] = nvgpu.mma.sync(%[[D694]], %[[D554]], %[[D706]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D708:.+]] = vector.insert %[[D707]], %[[D702]] [1, 2] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D709:.+]] = vector.extract %[[CST_1]][1, 3] : vector<2x16x2x2xf32>
// CHECK:          %[[D710:.+]] = nvgpu.mma.sync(%[[D688]], %[[D558]], %[[D709]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D711:.+]] = nvgpu.mma.sync(%[[D690]], %[[D560]], %[[D710]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D712:.+]] = nvgpu.mma.sync(%[[D692]], %[[D562]], %[[D711]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D713:.+]] = nvgpu.mma.sync(%[[D694]], %[[D564]], %[[D712]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D714:.+]] = vector.insert %[[D713]], %[[D708]] [1, 3] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D715:.+]] = vector.extract %[[CST_1]][1, 4] : vector<2x16x2x2xf32>
// CHECK:          %[[D716:.+]] = nvgpu.mma.sync(%[[D688]], %[[D568]], %[[D715]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D717:.+]] = nvgpu.mma.sync(%[[D690]], %[[D570]], %[[D716]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D718:.+]] = nvgpu.mma.sync(%[[D692]], %[[D572]], %[[D717]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D719:.+]] = nvgpu.mma.sync(%[[D694]], %[[D574]], %[[D718]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D720:.+]] = vector.insert %[[D719]], %[[D714]] [1, 4] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D721:.+]] = vector.extract %[[CST_1]][1, 5] : vector<2x16x2x2xf32>
// CHECK:          %[[D722:.+]] = nvgpu.mma.sync(%[[D688]], %[[D578]], %[[D721]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D723:.+]] = nvgpu.mma.sync(%[[D690]], %[[D580]], %[[D722]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D724:.+]] = nvgpu.mma.sync(%[[D692]], %[[D582]], %[[D723]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D725:.+]] = nvgpu.mma.sync(%[[D694]], %[[D584]], %[[D724]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D726:.+]] = vector.insert %[[D725]], %[[D720]] [1, 5] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D727:.+]] = vector.extract %[[CST_1]][1, 6] : vector<2x16x2x2xf32>
// CHECK:          %[[D728:.+]] = nvgpu.mma.sync(%[[D688]], %[[D588]], %[[D727]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D729:.+]] = nvgpu.mma.sync(%[[D690]], %[[D590]], %[[D728]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D730:.+]] = nvgpu.mma.sync(%[[D692]], %[[D592]], %[[D729]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D731:.+]] = nvgpu.mma.sync(%[[D694]], %[[D594]], %[[D730]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D732:.+]] = vector.insert %[[D731]], %[[D726]] [1, 6] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D733:.+]] = vector.extract %[[CST_1]][1, 7] : vector<2x16x2x2xf32>
// CHECK:          %[[D734:.+]] = nvgpu.mma.sync(%[[D688]], %[[D598]], %[[D733]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D735:.+]] = nvgpu.mma.sync(%[[D690]], %[[D600]], %[[D734]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D736:.+]] = nvgpu.mma.sync(%[[D692]], %[[D602]], %[[D735]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D737:.+]] = nvgpu.mma.sync(%[[D694]], %[[D604]], %[[D736]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D738:.+]] = vector.insert %[[D737]], %[[D732]] [1, 7] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D739:.+]] = vector.extract %[[CST_1]][1, 8] : vector<2x16x2x2xf32>
// CHECK:          %[[D740:.+]] = nvgpu.mma.sync(%[[D688]], %[[D608]], %[[D739]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D741:.+]] = nvgpu.mma.sync(%[[D690]], %[[D610]], %[[D740]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D742:.+]] = nvgpu.mma.sync(%[[D692]], %[[D612]], %[[D741]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D743:.+]] = nvgpu.mma.sync(%[[D694]], %[[D614]], %[[D742]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D744:.+]] = vector.insert %[[D743]], %[[D738]] [1, 8] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D745:.+]] = vector.extract %[[CST_1]][1, 9] : vector<2x16x2x2xf32>
// CHECK:          %[[D746:.+]] = nvgpu.mma.sync(%[[D688]], %[[D618]], %[[D745]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D747:.+]] = nvgpu.mma.sync(%[[D690]], %[[D620]], %[[D746]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D748:.+]] = nvgpu.mma.sync(%[[D692]], %[[D622]], %[[D747]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D749:.+]] = nvgpu.mma.sync(%[[D694]], %[[D624]], %[[D748]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D750:.+]] = vector.insert %[[D749]], %[[D744]] [1, 9] : vector<2x2xf32> into vector<2x16x2x2xf32>
// CHECK:          %[[D751:.+]] = vector.extract %[[CST_1]][1, 10] : vector<2x16x2x2xf32>
// CHECK:          %[[D752:.+]] = nvgpu.mma.sync(%[[D688]], %[[D628]], %[[D751]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D753:.+]] = nvgpu.mma.sync(%[[D690]], %[[D630]], %[[D752]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D754:.+]] = nvgpu.mma.sync(%[[D692]], %[[D632]], %[[D753]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D755:.+]] = nvgpu.mma.sync(%[[D694]], %[[D634]], %[[D754]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D756:.+]] = vector.insert %[[D755]], %[[D750]] [1, 10] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D757:.+]] = vector.extract %[[CST_1]][1, 11] : vector<2x16x2x2xf32>
// CHECK:          %[[D758:.+]] = nvgpu.mma.sync(%[[D688]], %[[D638]], %[[D757]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D759:.+]] = nvgpu.mma.sync(%[[D690]], %[[D640]], %[[D758]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D760:.+]] = nvgpu.mma.sync(%[[D692]], %[[D642]], %[[D759]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D761:.+]] = nvgpu.mma.sync(%[[D694]], %[[D644]], %[[D760]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D762:.+]] = vector.insert %[[D761]], %[[D756]] [1, 11] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D763:.+]] = vector.extract %[[CST_1]][1, 12] : vector<2x16x2x2xf32>
// CHECK:          %[[D764:.+]] = nvgpu.mma.sync(%[[D688]], %[[D648]], %[[D763]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D765:.+]] = nvgpu.mma.sync(%[[D690]], %[[D650]], %[[D764]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D766:.+]] = nvgpu.mma.sync(%[[D692]], %[[D652]], %[[D765]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D767:.+]] = nvgpu.mma.sync(%[[D694]], %[[D654]], %[[D766]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D768:.+]] = vector.insert %[[D767]], %[[D762]] [1, 12] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D769:.+]] = vector.extract %[[CST_1]][1, 13] : vector<2x16x2x2xf32>
// CHECK:          %[[D770:.+]] = nvgpu.mma.sync(%[[D688]], %[[D658]], %[[D769]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D771:.+]] = nvgpu.mma.sync(%[[D690]], %[[D660]], %[[D770]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D772:.+]] = nvgpu.mma.sync(%[[D692]], %[[D662]], %[[D771]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D773:.+]] = nvgpu.mma.sync(%[[D694]], %[[D664]], %[[D772]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D774:.+]] = vector.insert %[[D773]], %[[D768]] [1, 13] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D775:.+]] = vector.extract %[[CST_1]][1, 14] : vector<2x16x2x2xf32>
// CHECK:          %[[D776:.+]] = nvgpu.mma.sync(%[[D688]], %[[D668]], %[[D775]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D777:.+]] = nvgpu.mma.sync(%[[D690]], %[[D670]], %[[D776]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D778:.+]] = nvgpu.mma.sync(%[[D692]], %[[D672]], %[[D777]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D779:.+]] = nvgpu.mma.sync(%[[D694]], %[[D674]], %[[D778]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D780:.+]] = vector.insert %[[D779]], %[[D774]] [1, 14] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D781:.+]] = vector.extract %[[CST_1]][1, 15] : vector<2x16x2x2xf32>
// CHECK:          %[[D782:.+]] = nvgpu.mma.sync(%[[D688]], %[[D678]], %[[D781]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D783:.+]] = nvgpu.mma.sync(%[[D690]], %[[D680]], %[[D782]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D784:.+]] = nvgpu.mma.sync(%[[D692]], %[[D682]], %[[D783]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D785:.+]] = nvgpu.mma.sync(%[[D694]], %[[D684]], %[[D784]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D786:.+]] = vector.insert %[[D785]], %[[D780]] [1, 15] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x16x2x2xf32>
// CHECK:          %[[D787:.+]] = vector.extract %[[ARG4]][0, 0, 0, 0] : vector<2x16x2x2xf32>
// CHECK-DAG:      %[[CST_13:.+]] = arith.constant dense<0.000000e+00> : vector<1xf32>
// CHECK:          %[[D788:.+]] = vector.extract %[[D786]][0, 0, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D789:.+]] = vector.insert %[[D788]], %[[CST_13]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D790:.+]] = vector.extract %[[D786]][0, 0, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D791:.+]] = vector.insert %[[D790]], %[[D789]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D792:.+]] = arith.maxf %[[D789]], %[[D791]] : vector<1xf32>
// CHECK:          %[[D793:.+]] = vector.extract %[[D786]][0, 1, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D794:.+]] = vector.insert %[[D793]], %[[D791]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D795:.+]] = arith.maxf %[[D792]], %[[D794]] : vector<1xf32>
// CHECK:          %[[D796:.+]] = vector.extract %[[D786]][0, 1, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D797:.+]] = vector.insert %[[D796]], %[[D794]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D798:.+]] = arith.maxf %[[D795]], %[[D797]] : vector<1xf32>
// CHECK:          %[[D799:.+]] = vector.extract %[[D786]][0, 2, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D800:.+]] = vector.insert %[[D799]], %[[D797]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D801:.+]] = arith.maxf %[[D798]], %[[D800]] : vector<1xf32>
// CHECK:          %[[D802:.+]] = vector.extract %[[D786]][0, 2, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D803:.+]] = vector.insert %[[D802]], %[[D800]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D804:.+]] = arith.maxf %[[D801]], %[[D803]] : vector<1xf32>
// CHECK:          %[[D805:.+]] = vector.extract %[[D786]][0, 3, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D806:.+]] = vector.insert %[[D805]], %[[D803]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D807:.+]] = arith.maxf %[[D804]], %[[D806]] : vector<1xf32>
// CHECK:          %[[D808:.+]] = vector.extract %[[D786]][0, 3, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D809:.+]] = vector.insert %[[D808]], %[[D806]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D810:.+]] = arith.maxf %[[D807]], %[[D809]] : vector<1xf32>
// CHECK:          %[[D811:.+]] = vector.extract %[[D786]][0, 4, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D812:.+]] = vector.insert %[[D811]], %[[D809]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D813:.+]] = arith.maxf %[[D810]], %[[D812]] : vector<1xf32>
// CHECK:          %[[D814:.+]] = vector.extract %[[D786]][0, 4, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D815:.+]] = vector.insert %[[D814]], %[[D812]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D816:.+]] = arith.maxf %[[D813]], %[[D815]] : vector<1xf32>
// CHECK:          %[[D817:.+]] = vector.extract %[[D786]][0, 5, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D818:.+]] = vector.insert %[[D817]], %[[D815]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D819:.+]] = arith.maxf %[[D816]], %[[D818]] : vector<1xf32>
// CHECK:          %[[D820:.+]] = vector.extract %[[D786]][0, 5, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D821:.+]] = vector.insert %[[D820]], %[[D818]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D822:.+]] = arith.maxf %[[D819]], %[[D821]] : vector<1xf32>
// CHECK:          %[[D823:.+]] = vector.extract %[[D786]][0, 6, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D824:.+]] = vector.insert %[[D823]], %[[D821]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D825:.+]] = arith.maxf %[[D822]], %[[D824]] : vector<1xf32>
// CHECK:          %[[D826:.+]] = vector.extract %[[D786]][0, 6, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D827:.+]] = vector.insert %[[D826]], %[[D824]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D828:.+]] = arith.maxf %[[D825]], %[[D827]] : vector<1xf32>
// CHECK:          %[[D829:.+]] = vector.extract %[[D786]][0, 7, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D830:.+]] = vector.insert %[[D829]], %[[D827]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D831:.+]] = arith.maxf %[[D828]], %[[D830]] : vector<1xf32>
// CHECK:          %[[D832:.+]] = vector.extract %[[D786]][0, 7, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D833:.+]] = vector.insert %[[D832]], %[[D830]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D834:.+]] = arith.maxf %[[D831]], %[[D833]] : vector<1xf32>
// CHECK:          %[[D835:.+]] = vector.extract %[[D786]][0, 8, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D836:.+]] = vector.insert %[[D835]], %[[D833]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D837:.+]] = arith.maxf %[[D834]], %[[D836]] : vector<1xf32>
// CHECK:          %[[D838:.+]] = vector.extract %[[D786]][0, 8, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D839:.+]] = vector.insert %[[D838]], %[[D836]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D840:.+]] = arith.maxf %[[D837]], %[[D839]] : vector<1xf32>
// CHECK:          %[[D841:.+]] = vector.extract %[[D786]][0, 9, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D842:.+]] = vector.insert %[[D841]], %[[D839]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D843:.+]] = arith.maxf %[[D840]], %[[D842]] : vector<1xf32>
// CHECK:          %[[D844:.+]] = vector.extract %[[D786]][0, 9, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D845:.+]] = vector.insert %[[D844]], %[[D842]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D846:.+]] = arith.maxf %[[D843]], %[[D845]] : vector<1xf32>
// CHECK:          %[[D847:.+]] = vector.extract %[[D786]][0, 10, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D848:.+]] = vector.insert %[[D847]], %[[D845]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D849:.+]] = arith.maxf %[[D846]], %[[D848]] : vector<1xf32>
// CHECK:          %[[D850:.+]] = vector.extract %[[D786]][0, 10, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D851:.+]] = vector.insert %[[D850]], %[[D848]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D852:.+]] = arith.maxf %[[D849]], %[[D851]] : vector<1xf32>
// CHECK:          %[[D853:.+]] = vector.extract %[[D786]][0, 11, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D854:.+]] = vector.insert %[[D853]], %[[D851]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D855:.+]] = arith.maxf %[[D852]], %[[D854]] : vector<1xf32>
// CHECK:          %[[D856:.+]] = vector.extract %[[D786]][0, 11, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D857:.+]] = vector.insert %[[D856]], %[[D854]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D858:.+]] = arith.maxf %[[D855]], %[[D857]] : vector<1xf32>
// CHECK:          %[[D859:.+]] = vector.extract %[[D786]][0, 12, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D860:.+]] = vector.insert %[[D859]], %[[D857]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D861:.+]] = arith.maxf %[[D858]], %[[D860]] : vector<1xf32>
// CHECK:          %[[D862:.+]] = vector.extract %[[D786]][0, 12, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D863:.+]] = vector.insert %[[D862]], %[[D860]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D864:.+]] = arith.maxf %[[D861]], %[[D863]] : vector<1xf32>
// CHECK:          %[[D865:.+]] = vector.extract %[[D786]][0, 13, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D866:.+]] = vector.insert %[[D865]], %[[D863]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D867:.+]] = arith.maxf %[[D864]], %[[D866]] : vector<1xf32>
// CHECK:          %[[D868:.+]] = vector.extract %[[D786]][0, 13, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D869:.+]] = vector.insert %[[D868]], %[[D866]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D870:.+]] = arith.maxf %[[D867]], %[[D869]] : vector<1xf32>
// CHECK:          %[[D871:.+]] = vector.extract %[[D786]][0, 14, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D872:.+]] = vector.insert %[[D871]], %[[D869]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D873:.+]] = arith.maxf %[[D870]], %[[D872]] : vector<1xf32>
// CHECK:          %[[D874:.+]] = vector.extract %[[D786]][0, 14, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D875:.+]] = vector.insert %[[D874]], %[[D872]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D876:.+]] = arith.maxf %[[D873]], %[[D875]] : vector<1xf32>
// CHECK:          %[[D877:.+]] = vector.extract %[[D786]][0, 15, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D878:.+]] = vector.insert %[[D877]], %[[D875]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D879:.+]] = arith.maxf %[[D876]], %[[D878]] : vector<1xf32>
// CHECK:          %[[D880:.+]] = vector.extract %[[D786]][0, 15, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D881:.+]] = vector.insert %[[D880]], %[[D878]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D882:.+]] = arith.maxf %[[D879]], %[[D881]] : vector<1xf32>
// CHECK:          %[[D883:.+]] = vector.bitcast %[[D882]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D884:.+]] = vector.extract %[[D883]][0] : vector<1xi32>
// CHECK-DAG:      %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-DAG:      %[[C32_I32:.+]] = arith.constant 32 : i32
// CHECK:          %[[SHUFFLERESULT:.+]], %[[VALID:.+]] = gpu.shuffle  xor %[[D884]], %[[C1_I32]], %[[C32_I32]] : i32
// CHECK:          %[[D885:.+]] = vector.broadcast %[[SHUFFLERESULT]] : i32 to vector<1xi32>
// CHECK:          %[[D886:.+]] = vector.bitcast %[[D885]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D887:.+]] = arith.maxf %[[D886]], %[[D882]] : vector<1xf32>
// CHECK:          %[[D888:.+]] = vector.bitcast %[[D887]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D889:.+]] = vector.extract %[[D888]][0] : vector<1xi32>
// CHECK-DAG:      %[[C2_I32:.+]] = arith.constant 2 : i32
// CHECK:          %[[SHUFFLERESULT_14:.+]], %[[VALID_15:.+]] = gpu.shuffle  xor %[[D889]], %[[C2_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D890:.+]] = vector.broadcast %[[SHUFFLERESULT_14]] : i32 to vector<1xi32>
// CHECK:          %[[D891:.+]] = vector.bitcast %[[D890]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D892:.+]] = arith.maxf %[[D891]], %[[D887]] : vector<1xf32>
// CHECK:          %[[D893:.+]] = vector.extract %[[D892]][0] : vector<1xf32>
// CHECK:          %[[D894:.+]] = arith.maxf %[[D893]], %[[D787]] : f32
// CHECK:          %[[D895:.+]] = vector.insert %[[D894]], %[[CST_1]] [0, 0, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D896:.+]] = vector.insert %[[D894]], %[[D895]] [0, 0, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D897:.+]] = vector.insert %[[D894]], %[[D896]] [0, 1, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D898:.+]] = vector.insert %[[D894]], %[[D897]] [0, 1, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D899:.+]] = vector.insert %[[D894]], %[[D898]] [0, 2, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D900:.+]] = vector.insert %[[D894]], %[[D899]] [0, 2, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D901:.+]] = vector.insert %[[D894]], %[[D900]] [0, 3, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D902:.+]] = vector.insert %[[D894]], %[[D901]] [0, 3, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D903:.+]] = vector.insert %[[D894]], %[[D902]] [0, 4, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D904:.+]] = vector.insert %[[D894]], %[[D903]] [0, 4, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D905:.+]] = vector.insert %[[D894]], %[[D904]] [0, 5, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D906:.+]] = vector.insert %[[D894]], %[[D905]] [0, 5, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D907:.+]] = vector.insert %[[D894]], %[[D906]] [0, 6, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D908:.+]] = vector.insert %[[D894]], %[[D907]] [0, 6, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D909:.+]] = vector.insert %[[D894]], %[[D908]] [0, 7, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D910:.+]] = vector.insert %[[D894]], %[[D909]] [0, 7, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D911:.+]] = vector.insert %[[D894]], %[[D910]] [0, 8, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D912:.+]] = vector.insert %[[D894]], %[[D911]] [0, 8, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D913:.+]] = vector.insert %[[D894]], %[[D912]] [0, 9, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D914:.+]] = vector.insert %[[D894]], %[[D913]] [0, 9, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D915:.+]] = vector.insert %[[D894]], %[[D914]] [0, 10, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D916:.+]] = vector.insert %[[D894]], %[[D915]] [0, 10, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D917:.+]] = vector.insert %[[D894]], %[[D916]] [0, 11, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D918:.+]] = vector.insert %[[D894]], %[[D917]] [0, 11, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D919:.+]] = vector.insert %[[D894]], %[[D918]] [0, 12, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D920:.+]] = vector.insert %[[D894]], %[[D919]] [0, 12, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D921:.+]] = vector.insert %[[D894]], %[[D920]] [0, 13, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D922:.+]] = vector.insert %[[D894]], %[[D921]] [0, 13, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D923:.+]] = vector.insert %[[D894]], %[[D922]] [0, 14, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D924:.+]] = vector.insert %[[D894]], %[[D923]] [0, 14, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D925:.+]] = vector.insert %[[D894]], %[[D924]] [0, 15, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D926:.+]] = vector.insert %[[D894]], %[[D925]] [0, 15, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D927:.+]] = vector.extract %[[ARG4]][0, 0, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D928:.+]] = vector.extract %[[D786]][0, 0, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D929:.+]] = vector.insert %[[D928]], %[[CST_13]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D930:.+]] = vector.extract %[[D786]][0, 0, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D931:.+]] = vector.insert %[[D930]], %[[D929]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D932:.+]] = arith.maxf %[[D929]], %[[D931]] : vector<1xf32>
// CHECK:          %[[D933:.+]] = vector.extract %[[D786]][0, 1, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D934:.+]] = vector.insert %[[D933]], %[[D931]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D935:.+]] = arith.maxf %[[D932]], %[[D934]] : vector<1xf32>
// CHECK:          %[[D936:.+]] = vector.extract %[[D786]][0, 1, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D937:.+]] = vector.insert %[[D936]], %[[D934]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D938:.+]] = arith.maxf %[[D935]], %[[D937]] : vector<1xf32>
// CHECK:          %[[D939:.+]] = vector.extract %[[D786]][0, 2, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D940:.+]] = vector.insert %[[D939]], %[[D937]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D941:.+]] = arith.maxf %[[D938]], %[[D940]] : vector<1xf32>
// CHECK:          %[[D942:.+]] = vector.extract %[[D786]][0, 2, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D943:.+]] = vector.insert %[[D942]], %[[D940]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D944:.+]] = arith.maxf %[[D941]], %[[D943]] : vector<1xf32>
// CHECK:          %[[D945:.+]] = vector.extract %[[D786]][0, 3, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D946:.+]] = vector.insert %[[D945]], %[[D943]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D947:.+]] = arith.maxf %[[D944]], %[[D946]] : vector<1xf32>
// CHECK:          %[[D948:.+]] = vector.extract %[[D786]][0, 3, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D949:.+]] = vector.insert %[[D948]], %[[D946]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D950:.+]] = arith.maxf %[[D947]], %[[D949]] : vector<1xf32>
// CHECK:          %[[D951:.+]] = vector.extract %[[D786]][0, 4, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D952:.+]] = vector.insert %[[D951]], %[[D949]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D953:.+]] = arith.maxf %[[D950]], %[[D952]] : vector<1xf32>
// CHECK:          %[[D954:.+]] = vector.extract %[[D786]][0, 4, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D955:.+]] = vector.insert %[[D954]], %[[D952]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D956:.+]] = arith.maxf %[[D953]], %[[D955]] : vector<1xf32>
// CHECK:          %[[D957:.+]] = vector.extract %[[D786]][0, 5, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D958:.+]] = vector.insert %[[D957]], %[[D955]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D959:.+]] = arith.maxf %[[D956]], %[[D958]] : vector<1xf32>
// CHECK:          %[[D960:.+]] = vector.extract %[[D786]][0, 5, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D961:.+]] = vector.insert %[[D960]], %[[D958]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D962:.+]] = arith.maxf %[[D959]], %[[D961]] : vector<1xf32>
// CHECK:          %[[D963:.+]] = vector.extract %[[D786]][0, 6, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D964:.+]] = vector.insert %[[D963]], %[[D961]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D965:.+]] = arith.maxf %[[D962]], %[[D964]] : vector<1xf32>
// CHECK:          %[[D966:.+]] = vector.extract %[[D786]][0, 6, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D967:.+]] = vector.insert %[[D966]], %[[D964]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D968:.+]] = arith.maxf %[[D965]], %[[D967]] : vector<1xf32>
// CHECK:          %[[D969:.+]] = vector.extract %[[D786]][0, 7, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D970:.+]] = vector.insert %[[D969]], %[[D967]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D971:.+]] = arith.maxf %[[D968]], %[[D970]] : vector<1xf32>
// CHECK:          %[[D972:.+]] = vector.extract %[[D786]][0, 7, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D973:.+]] = vector.insert %[[D972]], %[[D970]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D974:.+]] = arith.maxf %[[D971]], %[[D973]] : vector<1xf32>
// CHECK:          %[[D975:.+]] = vector.extract %[[D786]][0, 8, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D976:.+]] = vector.insert %[[D975]], %[[D973]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D977:.+]] = arith.maxf %[[D974]], %[[D976]] : vector<1xf32>
// CHECK:          %[[D978:.+]] = vector.extract %[[D786]][0, 8, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D979:.+]] = vector.insert %[[D978]], %[[D976]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D980:.+]] = arith.maxf %[[D977]], %[[D979]] : vector<1xf32>
// CHECK:          %[[D981:.+]] = vector.extract %[[D786]][0, 9, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D982:.+]] = vector.insert %[[D981]], %[[D979]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D983:.+]] = arith.maxf %[[D980]], %[[D982]] : vector<1xf32>
// CHECK:          %[[D984:.+]] = vector.extract %[[D786]][0, 9, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D985:.+]] = vector.insert %[[D984]], %[[D982]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D986:.+]] = arith.maxf %[[D983]], %[[D985]] : vector<1xf32>
// CHECK:          %[[D987:.+]] = vector.extract %[[D786]][0, 10, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D988:.+]] = vector.insert %[[D987]], %[[D985]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D989:.+]] = arith.maxf %[[D986]], %[[D988]] : vector<1xf32>
// CHECK:          %[[D990:.+]] = vector.extract %[[D786]][0, 10, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D991:.+]] = vector.insert %[[D990]], %[[D988]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D992:.+]] = arith.maxf %[[D989]], %[[D991]] : vector<1xf32>
// CHECK:          %[[D993:.+]] = vector.extract %[[D786]][0, 11, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D994:.+]] = vector.insert %[[D993]], %[[D991]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D995:.+]] = arith.maxf %[[D992]], %[[D994]] : vector<1xf32>
// CHECK:          %[[D996:.+]] = vector.extract %[[D786]][0, 11, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D997:.+]] = vector.insert %[[D996]], %[[D994]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D998:.+]] = arith.maxf %[[D995]], %[[D997]] : vector<1xf32>
// CHECK:          %[[D999:.+]] = vector.extract %[[D786]][0, 12, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1000:.+]] = vector.insert %[[D999]], %[[D997]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1001:.+]] = arith.maxf %[[D998]], %[[D1000]] : vector<1xf32>
// CHECK:          %[[D1002:.+]] = vector.extract %[[D786]][0, 12, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1003:.+]] = vector.insert %[[D1002]], %[[D1000]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1004:.+]] = arith.maxf %[[D1001]], %[[D1003]] : vector<1xf32>
// CHECK:          %[[D1005:.+]] = vector.extract %[[D786]][0, 13, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1006:.+]] = vector.insert %[[D1005]], %[[D1003]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1007:.+]] = arith.maxf %[[D1004]], %[[D1006]] : vector<1xf32>
// CHECK:          %[[D1008:.+]] = vector.extract %[[D786]][0, 13, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1009:.+]] = vector.insert %[[D1008]], %[[D1006]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1010:.+]] = arith.maxf %[[D1007]], %[[D1009]] : vector<1xf32>
// CHECK:          %[[D1011:.+]] = vector.extract %[[D786]][0, 14, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1012:.+]] = vector.insert %[[D1011]], %[[D1009]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1013:.+]] = arith.maxf %[[D1010]], %[[D1012]] : vector<1xf32>
// CHECK:          %[[D1014:.+]] = vector.extract %[[D786]][0, 14, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1015:.+]] = vector.insert %[[D1014]], %[[D1012]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1016:.+]] = arith.maxf %[[D1013]], %[[D1015]] : vector<1xf32>
// CHECK:          %[[D1017:.+]] = vector.extract %[[D786]][0, 15, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1018:.+]] = vector.insert %[[D1017]], %[[D1015]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1019:.+]] = arith.maxf %[[D1016]], %[[D1018]] : vector<1xf32>
// CHECK:          %[[D1020:.+]] = vector.extract %[[D786]][0, 15, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1021:.+]] = vector.insert %[[D1020]], %[[D1018]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1022:.+]] = arith.maxf %[[D1019]], %[[D1021]] : vector<1xf32>
// CHECK:          %[[D1023:.+]] = vector.bitcast %[[D1022]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1024:.+]] = vector.extract %[[D1023]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_16:.+]], %[[VALID_17:.+]] = gpu.shuffle  xor %[[D1024]], %[[C1_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1025:.+]] = vector.broadcast %[[SHUFFLERESULT_16]] : i32 to vector<1xi32>
// CHECK:          %[[D1026:.+]] = vector.bitcast %[[D1025]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1027:.+]] = arith.maxf %[[D1026]], %[[D1022]] : vector<1xf32>
// CHECK:          %[[D1028:.+]] = vector.bitcast %[[D1027]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1029:.+]] = vector.extract %[[D1028]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_18:.+]], %[[VALID_19:.+]] = gpu.shuffle  xor %[[D1029]], %[[C2_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1030:.+]] = vector.broadcast %[[SHUFFLERESULT_18]] : i32 to vector<1xi32>
// CHECK:          %[[D1031:.+]] = vector.bitcast %[[D1030]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1032:.+]] = arith.maxf %[[D1031]], %[[D1027]] : vector<1xf32>
// CHECK:          %[[D1033:.+]] = vector.extract %[[D1032]][0] : vector<1xf32>
// CHECK:          %[[D1034:.+]] = arith.maxf %[[D1033]], %[[D927]] : f32
// CHECK:          %[[D1035:.+]] = vector.insert %[[D1034]], %[[D926]] [0, 0, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1036:.+]] = vector.insert %[[D1034]], %[[D1035]] [0, 0, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1037:.+]] = vector.insert %[[D1034]], %[[D1036]] [0, 1, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1038:.+]] = vector.insert %[[D1034]], %[[D1037]] [0, 1, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1039:.+]] = vector.insert %[[D1034]], %[[D1038]] [0, 2, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1040:.+]] = vector.insert %[[D1034]], %[[D1039]] [0, 2, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1041:.+]] = vector.insert %[[D1034]], %[[D1040]] [0, 3, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1042:.+]] = vector.insert %[[D1034]], %[[D1041]] [0, 3, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1043:.+]] = vector.insert %[[D1034]], %[[D1042]] [0, 4, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1044:.+]] = vector.insert %[[D1034]], %[[D1043]] [0, 4, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1045:.+]] = vector.insert %[[D1034]], %[[D1044]] [0, 5, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1046:.+]] = vector.insert %[[D1034]], %[[D1045]] [0, 5, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1047:.+]] = vector.insert %[[D1034]], %[[D1046]] [0, 6, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1048:.+]] = vector.insert %[[D1034]], %[[D1047]] [0, 6, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1049:.+]] = vector.insert %[[D1034]], %[[D1048]] [0, 7, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1050:.+]] = vector.insert %[[D1034]], %[[D1049]] [0, 7, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1051:.+]] = vector.insert %[[D1034]], %[[D1050]] [0, 8, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1052:.+]] = vector.insert %[[D1034]], %[[D1051]] [0, 8, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1053:.+]] = vector.insert %[[D1034]], %[[D1052]] [0, 9, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1054:.+]] = vector.insert %[[D1034]], %[[D1053]] [0, 9, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1055:.+]] = vector.insert %[[D1034]], %[[D1054]] [0, 10, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1056:.+]] = vector.insert %[[D1034]], %[[D1055]] [0, 10, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1057:.+]] = vector.insert %[[D1034]], %[[D1056]] [0, 11, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1058:.+]] = vector.insert %[[D1034]], %[[D1057]] [0, 11, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1059:.+]] = vector.insert %[[D1034]], %[[D1058]] [0, 12, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1060:.+]] = vector.insert %[[D1034]], %[[D1059]] [0, 12, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1061:.+]] = vector.insert %[[D1034]], %[[D1060]] [0, 13, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1062:.+]] = vector.insert %[[D1034]], %[[D1061]] [0, 13, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1063:.+]] = vector.insert %[[D1034]], %[[D1062]] [0, 14, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1064:.+]] = vector.insert %[[D1034]], %[[D1063]] [0, 14, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1065:.+]] = vector.insert %[[D1034]], %[[D1064]] [0, 15, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1066:.+]] = vector.insert %[[D1034]], %[[D1065]] [0, 15, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1067:.+]] = vector.extract %[[ARG4]][1, 0, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1068:.+]] = vector.extract %[[D786]][1, 0, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1069:.+]] = vector.insert %[[D1068]], %[[CST_13]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1070:.+]] = vector.extract %[[D786]][1, 0, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1071:.+]] = vector.insert %[[D1070]], %[[D1069]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1072:.+]] = arith.maxf %[[D1069]], %[[D1071]] : vector<1xf32>
// CHECK:          %[[D1073:.+]] = vector.extract %[[D786]][1, 1, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1074:.+]] = vector.insert %[[D1073]], %[[D1071]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1075:.+]] = arith.maxf %[[D1072]], %[[D1074]] : vector<1xf32>
// CHECK:          %[[D1076:.+]] = vector.extract %[[D786]][1, 1, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1077:.+]] = vector.insert %[[D1076]], %[[D1074]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1078:.+]] = arith.maxf %[[D1075]], %[[D1077]] : vector<1xf32>
// CHECK:          %[[D1079:.+]] = vector.extract %[[D786]][1, 2, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1080:.+]] = vector.insert %[[D1079]], %[[D1077]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1081:.+]] = arith.maxf %[[D1078]], %[[D1080]] : vector<1xf32>
// CHECK:          %[[D1082:.+]] = vector.extract %[[D786]][1, 2, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1083:.+]] = vector.insert %[[D1082]], %[[D1080]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1084:.+]] = arith.maxf %[[D1081]], %[[D1083]] : vector<1xf32>
// CHECK:          %[[D1085:.+]] = vector.extract %[[D786]][1, 3, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1086:.+]] = vector.insert %[[D1085]], %[[D1083]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1087:.+]] = arith.maxf %[[D1084]], %[[D1086]] : vector<1xf32>
// CHECK:          %[[D1088:.+]] = vector.extract %[[D786]][1, 3, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1089:.+]] = vector.insert %[[D1088]], %[[D1086]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1090:.+]] = arith.maxf %[[D1087]], %[[D1089]] : vector<1xf32>
// CHECK:          %[[D1091:.+]] = vector.extract %[[D786]][1, 4, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1092:.+]] = vector.insert %[[D1091]], %[[D1089]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1093:.+]] = arith.maxf %[[D1090]], %[[D1092]] : vector<1xf32>
// CHECK:          %[[D1094:.+]] = vector.extract %[[D786]][1, 4, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1095:.+]] = vector.insert %[[D1094]], %[[D1092]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1096:.+]] = arith.maxf %[[D1093]], %[[D1095]] : vector<1xf32>
// CHECK:          %[[D1097:.+]] = vector.extract %[[D786]][1, 5, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1098:.+]] = vector.insert %[[D1097]], %[[D1095]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1099:.+]] = arith.maxf %[[D1096]], %[[D1098]] : vector<1xf32>
// CHECK:          %[[D1100:.+]] = vector.extract %[[D786]][1, 5, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1101:.+]] = vector.insert %[[D1100]], %[[D1098]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1102:.+]] = arith.maxf %[[D1099]], %[[D1101]] : vector<1xf32>
// CHECK:          %[[D1103:.+]] = vector.extract %[[D786]][1, 6, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1104:.+]] = vector.insert %[[D1103]], %[[D1101]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1105:.+]] = arith.maxf %[[D1102]], %[[D1104]] : vector<1xf32>
// CHECK:          %[[D1106:.+]] = vector.extract %[[D786]][1, 6, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1107:.+]] = vector.insert %[[D1106]], %[[D1104]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1108:.+]] = arith.maxf %[[D1105]], %[[D1107]] : vector<1xf32>
// CHECK:          %[[D1109:.+]] = vector.extract %[[D786]][1, 7, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1110:.+]] = vector.insert %[[D1109]], %[[D1107]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1111:.+]] = arith.maxf %[[D1108]], %[[D1110]] : vector<1xf32>
// CHECK:          %[[D1112:.+]] = vector.extract %[[D786]][1, 7, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1113:.+]] = vector.insert %[[D1112]], %[[D1110]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1114:.+]] = arith.maxf %[[D1111]], %[[D1113]] : vector<1xf32>
// CHECK:          %[[D1115:.+]] = vector.extract %[[D786]][1, 8, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1116:.+]] = vector.insert %[[D1115]], %[[D1113]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1117:.+]] = arith.maxf %[[D1114]], %[[D1116]] : vector<1xf32>
// CHECK:          %[[D1118:.+]] = vector.extract %[[D786]][1, 8, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1119:.+]] = vector.insert %[[D1118]], %[[D1116]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1120:.+]] = arith.maxf %[[D1117]], %[[D1119]] : vector<1xf32>
// CHECK:          %[[D1121:.+]] = vector.extract %[[D786]][1, 9, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1122:.+]] = vector.insert %[[D1121]], %[[D1119]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1123:.+]] = arith.maxf %[[D1120]], %[[D1122]] : vector<1xf32>
// CHECK:          %[[D1124:.+]] = vector.extract %[[D786]][1, 9, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1125:.+]] = vector.insert %[[D1124]], %[[D1122]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1126:.+]] = arith.maxf %[[D1123]], %[[D1125]] : vector<1xf32>
// CHECK:          %[[D1127:.+]] = vector.extract %[[D786]][1, 10, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1128:.+]] = vector.insert %[[D1127]], %[[D1125]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1129:.+]] = arith.maxf %[[D1126]], %[[D1128]] : vector<1xf32>
// CHECK:          %[[D1130:.+]] = vector.extract %[[D786]][1, 10, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1131:.+]] = vector.insert %[[D1130]], %[[D1128]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1132:.+]] = arith.maxf %[[D1129]], %[[D1131]] : vector<1xf32>
// CHECK:          %[[D1133:.+]] = vector.extract %[[D786]][1, 11, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1134:.+]] = vector.insert %[[D1133]], %[[D1131]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1135:.+]] = arith.maxf %[[D1132]], %[[D1134]] : vector<1xf32>
// CHECK:          %[[D1136:.+]] = vector.extract %[[D786]][1, 11, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1137:.+]] = vector.insert %[[D1136]], %[[D1134]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1138:.+]] = arith.maxf %[[D1135]], %[[D1137]] : vector<1xf32>
// CHECK:          %[[D1139:.+]] = vector.extract %[[D786]][1, 12, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1140:.+]] = vector.insert %[[D1139]], %[[D1137]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1141:.+]] = arith.maxf %[[D1138]], %[[D1140]] : vector<1xf32>
// CHECK:          %[[D1142:.+]] = vector.extract %[[D786]][1, 12, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1143:.+]] = vector.insert %[[D1142]], %[[D1140]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1144:.+]] = arith.maxf %[[D1141]], %[[D1143]] : vector<1xf32>
// CHECK:          %[[D1145:.+]] = vector.extract %[[D786]][1, 13, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1146:.+]] = vector.insert %[[D1145]], %[[D1143]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1147:.+]] = arith.maxf %[[D1144]], %[[D1146]] : vector<1xf32>
// CHECK:          %[[D1148:.+]] = vector.extract %[[D786]][1, 13, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1149:.+]] = vector.insert %[[D1148]], %[[D1146]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1150:.+]] = arith.maxf %[[D1147]], %[[D1149]] : vector<1xf32>
// CHECK:          %[[D1151:.+]] = vector.extract %[[D786]][1, 14, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1152:.+]] = vector.insert %[[D1151]], %[[D1149]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1153:.+]] = arith.maxf %[[D1150]], %[[D1152]] : vector<1xf32>
// CHECK:          %[[D1154:.+]] = vector.extract %[[D786]][1, 14, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1155:.+]] = vector.insert %[[D1154]], %[[D1152]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1156:.+]] = arith.maxf %[[D1153]], %[[D1155]] : vector<1xf32>
// CHECK:          %[[D1157:.+]] = vector.extract %[[D786]][1, 15, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1158:.+]] = vector.insert %[[D1157]], %[[D1155]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1159:.+]] = arith.maxf %[[D1156]], %[[D1158]] : vector<1xf32>
// CHECK:          %[[D1160:.+]] = vector.extract %[[D786]][1, 15, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1161:.+]] = vector.insert %[[D1160]], %[[D1158]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1162:.+]] = arith.maxf %[[D1159]], %[[D1161]] : vector<1xf32>
// CHECK:          %[[D1163:.+]] = vector.bitcast %[[D1162]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1164:.+]] = vector.extract %[[D1163]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_20:.+]], %[[VALID_21:.+]] = gpu.shuffle  xor %[[D1164]], %[[C1_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1165:.+]] = vector.broadcast %[[SHUFFLERESULT_20]] : i32 to vector<1xi32>
// CHECK:          %[[D1166:.+]] = vector.bitcast %[[D1165]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1167:.+]] = arith.maxf %[[D1166]], %[[D1162]] : vector<1xf32>
// CHECK:          %[[D1168:.+]] = vector.bitcast %[[D1167]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1169:.+]] = vector.extract %[[D1168]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_22:.+]], %[[VALID_23:.+]] = gpu.shuffle  xor %[[D1169]], %[[C2_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1170:.+]] = vector.broadcast %[[SHUFFLERESULT_22]] : i32 to vector<1xi32>
// CHECK:          %[[D1171:.+]] = vector.bitcast %[[D1170]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1172:.+]] = arith.maxf %[[D1171]], %[[D1167]] : vector<1xf32>
// CHECK:          %[[D1173:.+]] = vector.extract %[[D1172]][0] : vector<1xf32>
// CHECK:          %[[D1174:.+]] = arith.maxf %[[D1173]], %[[D1067]] : f32
// CHECK:          %[[D1175:.+]] = vector.insert %[[D1174]], %[[D1066]] [1, 0, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1176:.+]] = vector.insert %[[D1174]], %[[D1175]] [1, 0, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1177:.+]] = vector.insert %[[D1174]], %[[D1176]] [1, 1, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1178:.+]] = vector.insert %[[D1174]], %[[D1177]] [1, 1, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1179:.+]] = vector.insert %[[D1174]], %[[D1178]] [1, 2, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1180:.+]] = vector.insert %[[D1174]], %[[D1179]] [1, 2, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1181:.+]] = vector.insert %[[D1174]], %[[D1180]] [1, 3, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1182:.+]] = vector.insert %[[D1174]], %[[D1181]] [1, 3, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1183:.+]] = vector.insert %[[D1174]], %[[D1182]] [1, 4, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1184:.+]] = vector.insert %[[D1174]], %[[D1183]] [1, 4, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1185:.+]] = vector.insert %[[D1174]], %[[D1184]] [1, 5, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1186:.+]] = vector.insert %[[D1174]], %[[D1185]] [1, 5, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1187:.+]] = vector.insert %[[D1174]], %[[D1186]] [1, 6, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1188:.+]] = vector.insert %[[D1174]], %[[D1187]] [1, 6, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1189:.+]] = vector.insert %[[D1174]], %[[D1188]] [1, 7, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1190:.+]] = vector.insert %[[D1174]], %[[D1189]] [1, 7, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1191:.+]] = vector.insert %[[D1174]], %[[D1190]] [1, 8, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1192:.+]] = vector.insert %[[D1174]], %[[D1191]] [1, 8, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1193:.+]] = vector.insert %[[D1174]], %[[D1192]] [1, 9, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1194:.+]] = vector.insert %[[D1174]], %[[D1193]] [1, 9, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1195:.+]] = vector.insert %[[D1174]], %[[D1194]] [1, 10, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1196:.+]] = vector.insert %[[D1174]], %[[D1195]] [1, 10, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1197:.+]] = vector.insert %[[D1174]], %[[D1196]] [1, 11, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1198:.+]] = vector.insert %[[D1174]], %[[D1197]] [1, 11, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1199:.+]] = vector.insert %[[D1174]], %[[D1198]] [1, 12, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1200:.+]] = vector.insert %[[D1174]], %[[D1199]] [1, 12, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1201:.+]] = vector.insert %[[D1174]], %[[D1200]] [1, 13, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1202:.+]] = vector.insert %[[D1174]], %[[D1201]] [1, 13, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1203:.+]] = vector.insert %[[D1174]], %[[D1202]] [1, 14, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1204:.+]] = vector.insert %[[D1174]], %[[D1203]] [1, 14, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1205:.+]] = vector.insert %[[D1174]], %[[D1204]] [1, 15, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1206:.+]] = vector.insert %[[D1174]], %[[D1205]] [1, 15, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1207:.+]] = vector.extract %[[ARG4]][1, 0, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1208:.+]] = vector.extract %[[D786]][1, 0, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1209:.+]] = vector.insert %[[D1208]], %[[CST_13]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1210:.+]] = vector.extract %[[D786]][1, 0, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1211:.+]] = vector.insert %[[D1210]], %[[D1209]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1212:.+]] = arith.maxf %[[D1209]], %[[D1211]] : vector<1xf32>
// CHECK:          %[[D1213:.+]] = vector.extract %[[D786]][1, 1, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1214:.+]] = vector.insert %[[D1213]], %[[D1211]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1215:.+]] = arith.maxf %[[D1212]], %[[D1214]] : vector<1xf32>
// CHECK:          %[[D1216:.+]] = vector.extract %[[D786]][1, 1, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1217:.+]] = vector.insert %[[D1216]], %[[D1214]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1218:.+]] = arith.maxf %[[D1215]], %[[D1217]] : vector<1xf32>
// CHECK:          %[[D1219:.+]] = vector.extract %[[D786]][1, 2, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1220:.+]] = vector.insert %[[D1219]], %[[D1217]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1221:.+]] = arith.maxf %[[D1218]], %[[D1220]] : vector<1xf32>
// CHECK:          %[[D1222:.+]] = vector.extract %[[D786]][1, 2, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1223:.+]] = vector.insert %[[D1222]], %[[D1220]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1224:.+]] = arith.maxf %[[D1221]], %[[D1223]] : vector<1xf32>
// CHECK:          %[[D1225:.+]] = vector.extract %[[D786]][1, 3, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1226:.+]] = vector.insert %[[D1225]], %[[D1223]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1227:.+]] = arith.maxf %[[D1224]], %[[D1226]] : vector<1xf32>
// CHECK:          %[[D1228:.+]] = vector.extract %[[D786]][1, 3, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1229:.+]] = vector.insert %[[D1228]], %[[D1226]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1230:.+]] = arith.maxf %[[D1227]], %[[D1229]] : vector<1xf32>
// CHECK:          %[[D1231:.+]] = vector.extract %[[D786]][1, 4, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1232:.+]] = vector.insert %[[D1231]], %[[D1229]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1233:.+]] = arith.maxf %[[D1230]], %[[D1232]] : vector<1xf32>
// CHECK:          %[[D1234:.+]] = vector.extract %[[D786]][1, 4, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1235:.+]] = vector.insert %[[D1234]], %[[D1232]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1236:.+]] = arith.maxf %[[D1233]], %[[D1235]] : vector<1xf32>
// CHECK:          %[[D1237:.+]] = vector.extract %[[D786]][1, 5, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1238:.+]] = vector.insert %[[D1237]], %[[D1235]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1239:.+]] = arith.maxf %[[D1236]], %[[D1238]] : vector<1xf32>
// CHECK:          %[[D1240:.+]] = vector.extract %[[D786]][1, 5, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1241:.+]] = vector.insert %[[D1240]], %[[D1238]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1242:.+]] = arith.maxf %[[D1239]], %[[D1241]] : vector<1xf32>
// CHECK:          %[[D1243:.+]] = vector.extract %[[D786]][1, 6, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1244:.+]] = vector.insert %[[D1243]], %[[D1241]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1245:.+]] = arith.maxf %[[D1242]], %[[D1244]] : vector<1xf32>
// CHECK:          %[[D1246:.+]] = vector.extract %[[D786]][1, 6, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1247:.+]] = vector.insert %[[D1246]], %[[D1244]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1248:.+]] = arith.maxf %[[D1245]], %[[D1247]] : vector<1xf32>
// CHECK:          %[[D1249:.+]] = vector.extract %[[D786]][1, 7, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1250:.+]] = vector.insert %[[D1249]], %[[D1247]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1251:.+]] = arith.maxf %[[D1248]], %[[D1250]] : vector<1xf32>
// CHECK:          %[[D1252:.+]] = vector.extract %[[D786]][1, 7, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1253:.+]] = vector.insert %[[D1252]], %[[D1250]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1254:.+]] = arith.maxf %[[D1251]], %[[D1253]] : vector<1xf32>
// CHECK:          %[[D1255:.+]] = vector.extract %[[D786]][1, 8, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1256:.+]] = vector.insert %[[D1255]], %[[D1253]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1257:.+]] = arith.maxf %[[D1254]], %[[D1256]] : vector<1xf32>
// CHECK:          %[[D1258:.+]] = vector.extract %[[D786]][1, 8, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1259:.+]] = vector.insert %[[D1258]], %[[D1256]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1260:.+]] = arith.maxf %[[D1257]], %[[D1259]] : vector<1xf32>
// CHECK:          %[[D1261:.+]] = vector.extract %[[D786]][1, 9, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1262:.+]] = vector.insert %[[D1261]], %[[D1259]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1263:.+]] = arith.maxf %[[D1260]], %[[D1262]] : vector<1xf32>
// CHECK:          %[[D1264:.+]] = vector.extract %[[D786]][1, 9, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1265:.+]] = vector.insert %[[D1264]], %[[D1262]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1266:.+]] = arith.maxf %[[D1263]], %[[D1265]] : vector<1xf32>
// CHECK:          %[[D1267:.+]] = vector.extract %[[D786]][1, 10, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1268:.+]] = vector.insert %[[D1267]], %[[D1265]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1269:.+]] = arith.maxf %[[D1266]], %[[D1268]] : vector<1xf32>
// CHECK:          %[[D1270:.+]] = vector.extract %[[D786]][1, 10, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1271:.+]] = vector.insert %[[D1270]], %[[D1268]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1272:.+]] = arith.maxf %[[D1269]], %[[D1271]] : vector<1xf32>
// CHECK:          %[[D1273:.+]] = vector.extract %[[D786]][1, 11, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1274:.+]] = vector.insert %[[D1273]], %[[D1271]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1275:.+]] = arith.maxf %[[D1272]], %[[D1274]] : vector<1xf32>
// CHECK:          %[[D1276:.+]] = vector.extract %[[D786]][1, 11, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1277:.+]] = vector.insert %[[D1276]], %[[D1274]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1278:.+]] = arith.maxf %[[D1275]], %[[D1277]] : vector<1xf32>
// CHECK:          %[[D1279:.+]] = vector.extract %[[D786]][1, 12, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1280:.+]] = vector.insert %[[D1279]], %[[D1277]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1281:.+]] = arith.maxf %[[D1278]], %[[D1280]] : vector<1xf32>
// CHECK:          %[[D1282:.+]] = vector.extract %[[D786]][1, 12, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1283:.+]] = vector.insert %[[D1282]], %[[D1280]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1284:.+]] = arith.maxf %[[D1281]], %[[D1283]] : vector<1xf32>
// CHECK:          %[[D1285:.+]] = vector.extract %[[D786]][1, 13, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1286:.+]] = vector.insert %[[D1285]], %[[D1283]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1287:.+]] = arith.maxf %[[D1284]], %[[D1286]] : vector<1xf32>
// CHECK:          %[[D1288:.+]] = vector.extract %[[D786]][1, 13, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1289:.+]] = vector.insert %[[D1288]], %[[D1286]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1290:.+]] = arith.maxf %[[D1287]], %[[D1289]] : vector<1xf32>
// CHECK:          %[[D1291:.+]] = vector.extract %[[D786]][1, 14, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1292:.+]] = vector.insert %[[D1291]], %[[D1289]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1293:.+]] = arith.maxf %[[D1290]], %[[D1292]] : vector<1xf32>
// CHECK:          %[[D1294:.+]] = vector.extract %[[D786]][1, 14, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1295:.+]] = vector.insert %[[D1294]], %[[D1292]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1296:.+]] = arith.maxf %[[D1293]], %[[D1295]] : vector<1xf32>
// CHECK:          %[[D1297:.+]] = vector.extract %[[D786]][1, 15, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1298:.+]] = vector.insert %[[D1297]], %[[D1295]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1299:.+]] = arith.maxf %[[D1296]], %[[D1298]] : vector<1xf32>
// CHECK:          %[[D1300:.+]] = vector.extract %[[D786]][1, 15, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1301:.+]] = vector.insert %[[D1300]], %[[D1298]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1302:.+]] = arith.maxf %[[D1299]], %[[D1301]] : vector<1xf32>
// CHECK:          %[[D1303:.+]] = vector.bitcast %[[D1302]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1304:.+]] = vector.extract %[[D1303]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_24:.+]], %[[VALID_25:.+]] = gpu.shuffle  xor %[[D1304]], %[[C1_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1305:.+]] = vector.broadcast %[[SHUFFLERESULT_24]] : i32 to vector<1xi32>
// CHECK:          %[[D1306:.+]] = vector.bitcast %[[D1305]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1307:.+]] = arith.maxf %[[D1306]], %[[D1302]] : vector<1xf32>
// CHECK:          %[[D1308:.+]] = vector.bitcast %[[D1307]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1309:.+]] = vector.extract %[[D1308]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_26:.+]], %[[VALID_27:.+]] = gpu.shuffle  xor %[[D1309]], %[[C2_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1310:.+]] = vector.broadcast %[[SHUFFLERESULT_26]] : i32 to vector<1xi32>
// CHECK:          %[[D1311:.+]] = vector.bitcast %[[D1310]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1312:.+]] = arith.maxf %[[D1311]], %[[D1307]] : vector<1xf32>
// CHECK:          %[[D1313:.+]] = vector.extract %[[D1312]][0] : vector<1xf32>
// CHECK:          %[[D1314:.+]] = arith.maxf %[[D1313]], %[[D1207]] : f32
// CHECK:          %[[D1315:.+]] = vector.insert %[[D1314]], %[[D1206]] [1, 0, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1316:.+]] = vector.insert %[[D1314]], %[[D1315]] [1, 0, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1317:.+]] = vector.insert %[[D1314]], %[[D1316]] [1, 1, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1318:.+]] = vector.insert %[[D1314]], %[[D1317]] [1, 1, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1319:.+]] = vector.insert %[[D1314]], %[[D1318]] [1, 2, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1320:.+]] = vector.insert %[[D1314]], %[[D1319]] [1, 2, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1321:.+]] = vector.insert %[[D1314]], %[[D1320]] [1, 3, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1322:.+]] = vector.insert %[[D1314]], %[[D1321]] [1, 3, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1323:.+]] = vector.insert %[[D1314]], %[[D1322]] [1, 4, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1324:.+]] = vector.insert %[[D1314]], %[[D1323]] [1, 4, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1325:.+]] = vector.insert %[[D1314]], %[[D1324]] [1, 5, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1326:.+]] = vector.insert %[[D1314]], %[[D1325]] [1, 5, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1327:.+]] = vector.insert %[[D1314]], %[[D1326]] [1, 6, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1328:.+]] = vector.insert %[[D1314]], %[[D1327]] [1, 6, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1329:.+]] = vector.insert %[[D1314]], %[[D1328]] [1, 7, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1330:.+]] = vector.insert %[[D1314]], %[[D1329]] [1, 7, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1331:.+]] = vector.insert %[[D1314]], %[[D1330]] [1, 8, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1332:.+]] = vector.insert %[[D1314]], %[[D1331]] [1, 8, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1333:.+]] = vector.insert %[[D1314]], %[[D1332]] [1, 9, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1334:.+]] = vector.insert %[[D1314]], %[[D1333]] [1, 9, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1335:.+]] = vector.insert %[[D1314]], %[[D1334]] [1, 10, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1336:.+]] = vector.insert %[[D1314]], %[[D1335]] [1, 10, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1337:.+]] = vector.insert %[[D1314]], %[[D1336]] [1, 11, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1338:.+]] = vector.insert %[[D1314]], %[[D1337]] [1, 11, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1339:.+]] = vector.insert %[[D1314]], %[[D1338]] [1, 12, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1340:.+]] = vector.insert %[[D1314]], %[[D1339]] [1, 12, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1341:.+]] = vector.insert %[[D1314]], %[[D1340]] [1, 13, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1342:.+]] = vector.insert %[[D1314]], %[[D1341]] [1, 13, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1343:.+]] = vector.insert %[[D1314]], %[[D1342]] [1, 14, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1344:.+]] = vector.insert %[[D1314]], %[[D1343]] [1, 14, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1345:.+]] = vector.insert %[[D1314]], %[[D1344]] [1, 15, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1346:.+]] = vector.insert %[[D1314]], %[[D1345]] [1, 15, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1347:.+]] = arith.subf %[[D786]], %[[D1346]] : vector<2x16x2x2xf32>
// CHECK:          %[[D1348:.+]] = math.exp2 %[[D1347]] : vector<2x16x2x2xf32>
// CHECK:          %[[D1349:.+]] = arith.subf %[[ARG4]], %[[D1346]] : vector<2x16x2x2xf32>
// CHECK:          %[[D1350:.+]] = math.exp2 %[[D1349]] : vector<2x16x2x2xf32>
// CHECK:          %[[D1351:.+]] = arith.mulf %[[D1350]], %[[ARG5]] : vector<2x16x2x2xf32>
// CHECK:          %[[D1352:.+]] = vector.extract %[[D1351]][0, 0, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1353:.+]] = vector.extract %[[D1348]][0, 0, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1354:.+]] = vector.insert %[[D1353]], %[[CST_13]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1355:.+]] = vector.extract %[[D1348]][0, 0, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1356:.+]] = vector.insert %[[D1355]], %[[D1354]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1357:.+]] = arith.addf %[[D1354]], %[[D1356]] : vector<1xf32>
// CHECK:          %[[D1358:.+]] = vector.extract %[[D1348]][0, 1, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1359:.+]] = vector.insert %[[D1358]], %[[D1356]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1360:.+]] = arith.addf %[[D1357]], %[[D1359]] : vector<1xf32>
// CHECK:          %[[D1361:.+]] = vector.extract %[[D1348]][0, 1, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1362:.+]] = vector.insert %[[D1361]], %[[D1359]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1363:.+]] = arith.addf %[[D1360]], %[[D1362]] : vector<1xf32>
// CHECK:          %[[D1364:.+]] = vector.extract %[[D1348]][0, 2, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1365:.+]] = vector.insert %[[D1364]], %[[D1362]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1366:.+]] = arith.addf %[[D1363]], %[[D1365]] : vector<1xf32>
// CHECK:          %[[D1367:.+]] = vector.extract %[[D1348]][0, 2, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1368:.+]] = vector.insert %[[D1367]], %[[D1365]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1369:.+]] = arith.addf %[[D1366]], %[[D1368]] : vector<1xf32>
// CHECK:          %[[D1370:.+]] = vector.extract %[[D1348]][0, 3, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1371:.+]] = vector.insert %[[D1370]], %[[D1368]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1372:.+]] = arith.addf %[[D1369]], %[[D1371]] : vector<1xf32>
// CHECK:          %[[D1373:.+]] = vector.extract %[[D1348]][0, 3, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1374:.+]] = vector.insert %[[D1373]], %[[D1371]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1375:.+]] = arith.addf %[[D1372]], %[[D1374]] : vector<1xf32>
// CHECK:          %[[D1376:.+]] = vector.extract %[[D1348]][0, 4, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1377:.+]] = vector.insert %[[D1376]], %[[D1374]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1378:.+]] = arith.addf %[[D1375]], %[[D1377]] : vector<1xf32>
// CHECK:          %[[D1379:.+]] = vector.extract %[[D1348]][0, 4, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1380:.+]] = vector.insert %[[D1379]], %[[D1377]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1381:.+]] = arith.addf %[[D1378]], %[[D1380]] : vector<1xf32>
// CHECK:          %[[D1382:.+]] = vector.extract %[[D1348]][0, 5, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1383:.+]] = vector.insert %[[D1382]], %[[D1380]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1384:.+]] = arith.addf %[[D1381]], %[[D1383]] : vector<1xf32>
// CHECK:          %[[D1385:.+]] = vector.extract %[[D1348]][0, 5, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1386:.+]] = vector.insert %[[D1385]], %[[D1383]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1387:.+]] = arith.addf %[[D1384]], %[[D1386]] : vector<1xf32>
// CHECK:          %[[D1388:.+]] = vector.extract %[[D1348]][0, 6, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1389:.+]] = vector.insert %[[D1388]], %[[D1386]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1390:.+]] = arith.addf %[[D1387]], %[[D1389]] : vector<1xf32>
// CHECK:          %[[D1391:.+]] = vector.extract %[[D1348]][0, 6, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1392:.+]] = vector.insert %[[D1391]], %[[D1389]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1393:.+]] = arith.addf %[[D1390]], %[[D1392]] : vector<1xf32>
// CHECK:          %[[D1394:.+]] = vector.extract %[[D1348]][0, 7, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1395:.+]] = vector.insert %[[D1394]], %[[D1392]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1396:.+]] = arith.addf %[[D1393]], %[[D1395]] : vector<1xf32>
// CHECK:          %[[D1397:.+]] = vector.extract %[[D1348]][0, 7, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1398:.+]] = vector.insert %[[D1397]], %[[D1395]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1399:.+]] = arith.addf %[[D1396]], %[[D1398]] : vector<1xf32>
// CHECK:          %[[D1400:.+]] = vector.extract %[[D1348]][0, 8, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1401:.+]] = vector.insert %[[D1400]], %[[D1398]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1402:.+]] = arith.addf %[[D1399]], %[[D1401]] : vector<1xf32>
// CHECK:          %[[D1403:.+]] = vector.extract %[[D1348]][0, 8, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1404:.+]] = vector.insert %[[D1403]], %[[D1401]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1405:.+]] = arith.addf %[[D1402]], %[[D1404]] : vector<1xf32>
// CHECK:          %[[D1406:.+]] = vector.extract %[[D1348]][0, 9, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1407:.+]] = vector.insert %[[D1406]], %[[D1404]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1408:.+]] = arith.addf %[[D1405]], %[[D1407]] : vector<1xf32>
// CHECK:          %[[D1409:.+]] = vector.extract %[[D1348]][0, 9, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1410:.+]] = vector.insert %[[D1409]], %[[D1407]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1411:.+]] = arith.addf %[[D1408]], %[[D1410]] : vector<1xf32>
// CHECK:          %[[D1412:.+]] = vector.extract %[[D1348]][0, 10, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1413:.+]] = vector.insert %[[D1412]], %[[D1410]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1414:.+]] = arith.addf %[[D1411]], %[[D1413]] : vector<1xf32>
// CHECK:          %[[D1415:.+]] = vector.extract %[[D1348]][0, 10, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1416:.+]] = vector.insert %[[D1415]], %[[D1413]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1417:.+]] = arith.addf %[[D1414]], %[[D1416]] : vector<1xf32>
// CHECK:          %[[D1418:.+]] = vector.extract %[[D1348]][0, 11, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1419:.+]] = vector.insert %[[D1418]], %[[D1416]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1420:.+]] = arith.addf %[[D1417]], %[[D1419]] : vector<1xf32>
// CHECK:          %[[D1421:.+]] = vector.extract %[[D1348]][0, 11, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1422:.+]] = vector.insert %[[D1421]], %[[D1419]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1423:.+]] = arith.addf %[[D1420]], %[[D1422]] : vector<1xf32>
// CHECK:          %[[D1424:.+]] = vector.extract %[[D1348]][0, 12, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1425:.+]] = vector.insert %[[D1424]], %[[D1422]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1426:.+]] = arith.addf %[[D1423]], %[[D1425]] : vector<1xf32>
// CHECK:          %[[D1427:.+]] = vector.extract %[[D1348]][0, 12, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1428:.+]] = vector.insert %[[D1427]], %[[D1425]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1429:.+]] = arith.addf %[[D1426]], %[[D1428]] : vector<1xf32>
// CHECK:          %[[D1430:.+]] = vector.extract %[[D1348]][0, 13, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1431:.+]] = vector.insert %[[D1430]], %[[D1428]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1432:.+]] = arith.addf %[[D1429]], %[[D1431]] : vector<1xf32>
// CHECK:          %[[D1433:.+]] = vector.extract %[[D1348]][0, 13, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1434:.+]] = vector.insert %[[D1433]], %[[D1431]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1435:.+]] = arith.addf %[[D1432]], %[[D1434]] : vector<1xf32>
// CHECK:          %[[D1436:.+]] = vector.extract %[[D1348]][0, 14, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1437:.+]] = vector.insert %[[D1436]], %[[D1434]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1438:.+]] = arith.addf %[[D1435]], %[[D1437]] : vector<1xf32>
// CHECK:          %[[D1439:.+]] = vector.extract %[[D1348]][0, 14, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1440:.+]] = vector.insert %[[D1439]], %[[D1437]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1441:.+]] = arith.addf %[[D1438]], %[[D1440]] : vector<1xf32>
// CHECK:          %[[D1442:.+]] = vector.extract %[[D1348]][0, 15, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1443:.+]] = vector.insert %[[D1442]], %[[D1440]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1444:.+]] = arith.addf %[[D1441]], %[[D1443]] : vector<1xf32>
// CHECK:          %[[D1445:.+]] = vector.extract %[[D1348]][0, 15, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1446:.+]] = vector.insert %[[D1445]], %[[D1443]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1447:.+]] = arith.addf %[[D1444]], %[[D1446]] : vector<1xf32>
// CHECK:          %[[D1448:.+]] = vector.bitcast %[[D1447]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1449:.+]] = vector.extract %[[D1448]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_28:.+]], %[[VALID_29:.+]] = gpu.shuffle  xor %[[D1449]], %[[C1_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1450:.+]] = vector.broadcast %[[SHUFFLERESULT_28]] : i32 to vector<1xi32>
// CHECK:          %[[D1451:.+]] = vector.bitcast %[[D1450]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1452:.+]] = arith.addf %[[D1451]], %[[D1447]] : vector<1xf32>
// CHECK:          %[[D1453:.+]] = vector.bitcast %[[D1452]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1454:.+]] = vector.extract %[[D1453]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_30:.+]], %[[VALID_31:.+]] = gpu.shuffle  xor %[[D1454]], %[[C2_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1455:.+]] = vector.broadcast %[[SHUFFLERESULT_30]] : i32 to vector<1xi32>
// CHECK:          %[[D1456:.+]] = vector.bitcast %[[D1455]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1457:.+]] = arith.addf %[[D1456]], %[[D1452]] : vector<1xf32>
// CHECK:          %[[D1458:.+]] = vector.extract %[[D1457]][0] : vector<1xf32>
// CHECK:          %[[D1459:.+]] = arith.addf %[[D1458]], %[[D1352]] : f32
// CHECK:          %[[D1460:.+]] = vector.insert %[[D1459]], %[[CST_1]] [0, 0, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1461:.+]] = vector.insert %[[D1459]], %[[D1460]] [0, 0, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1462:.+]] = vector.insert %[[D1459]], %[[D1461]] [0, 1, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1463:.+]] = vector.insert %[[D1459]], %[[D1462]] [0, 1, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1464:.+]] = vector.insert %[[D1459]], %[[D1463]] [0, 2, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1465:.+]] = vector.insert %[[D1459]], %[[D1464]] [0, 2, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1466:.+]] = vector.insert %[[D1459]], %[[D1465]] [0, 3, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1467:.+]] = vector.insert %[[D1459]], %[[D1466]] [0, 3, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1468:.+]] = vector.insert %[[D1459]], %[[D1467]] [0, 4, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1469:.+]] = vector.insert %[[D1459]], %[[D1468]] [0, 4, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1470:.+]] = vector.insert %[[D1459]], %[[D1469]] [0, 5, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1471:.+]] = vector.insert %[[D1459]], %[[D1470]] [0, 5, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1472:.+]] = vector.insert %[[D1459]], %[[D1471]] [0, 6, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1473:.+]] = vector.insert %[[D1459]], %[[D1472]] [0, 6, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1474:.+]] = vector.insert %[[D1459]], %[[D1473]] [0, 7, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1475:.+]] = vector.insert %[[D1459]], %[[D1474]] [0, 7, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1476:.+]] = vector.insert %[[D1459]], %[[D1475]] [0, 8, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1477:.+]] = vector.insert %[[D1459]], %[[D1476]] [0, 8, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1478:.+]] = vector.insert %[[D1459]], %[[D1477]] [0, 9, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1479:.+]] = vector.insert %[[D1459]], %[[D1478]] [0, 9, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1480:.+]] = vector.insert %[[D1459]], %[[D1479]] [0, 10, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1481:.+]] = vector.insert %[[D1459]], %[[D1480]] [0, 10, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1482:.+]] = vector.insert %[[D1459]], %[[D1481]] [0, 11, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1483:.+]] = vector.insert %[[D1459]], %[[D1482]] [0, 11, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1484:.+]] = vector.insert %[[D1459]], %[[D1483]] [0, 12, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1485:.+]] = vector.insert %[[D1459]], %[[D1484]] [0, 12, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1486:.+]] = vector.insert %[[D1459]], %[[D1485]] [0, 13, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1487:.+]] = vector.insert %[[D1459]], %[[D1486]] [0, 13, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1488:.+]] = vector.insert %[[D1459]], %[[D1487]] [0, 14, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1489:.+]] = vector.insert %[[D1459]], %[[D1488]] [0, 14, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1490:.+]] = vector.insert %[[D1459]], %[[D1489]] [0, 15, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1491:.+]] = vector.insert %[[D1459]], %[[D1490]] [0, 15, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1492:.+]] = vector.extract %[[D1351]][0, 0, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1493:.+]] = vector.extract %[[D1348]][0, 0, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1494:.+]] = vector.insert %[[D1493]], %[[CST_13]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1495:.+]] = vector.extract %[[D1348]][0, 0, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1496:.+]] = vector.insert %[[D1495]], %[[D1494]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1497:.+]] = arith.addf %[[D1494]], %[[D1496]] : vector<1xf32>
// CHECK:          %[[D1498:.+]] = vector.extract %[[D1348]][0, 1, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1499:.+]] = vector.insert %[[D1498]], %[[D1496]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1500:.+]] = arith.addf %[[D1497]], %[[D1499]] : vector<1xf32>
// CHECK:          %[[D1501:.+]] = vector.extract %[[D1348]][0, 1, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1502:.+]] = vector.insert %[[D1501]], %[[D1499]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1503:.+]] = arith.addf %[[D1500]], %[[D1502]] : vector<1xf32>
// CHECK:          %[[D1504:.+]] = vector.extract %[[D1348]][0, 2, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1505:.+]] = vector.insert %[[D1504]], %[[D1502]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1506:.+]] = arith.addf %[[D1503]], %[[D1505]] : vector<1xf32>
// CHECK:          %[[D1507:.+]] = vector.extract %[[D1348]][0, 2, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1508:.+]] = vector.insert %[[D1507]], %[[D1505]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1509:.+]] = arith.addf %[[D1506]], %[[D1508]] : vector<1xf32>
// CHECK:          %[[D1510:.+]] = vector.extract %[[D1348]][0, 3, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1511:.+]] = vector.insert %[[D1510]], %[[D1508]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1512:.+]] = arith.addf %[[D1509]], %[[D1511]] : vector<1xf32>
// CHECK:          %[[D1513:.+]] = vector.extract %[[D1348]][0, 3, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1514:.+]] = vector.insert %[[D1513]], %[[D1511]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1515:.+]] = arith.addf %[[D1512]], %[[D1514]] : vector<1xf32>
// CHECK:          %[[D1516:.+]] = vector.extract %[[D1348]][0, 4, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1517:.+]] = vector.insert %[[D1516]], %[[D1514]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1518:.+]] = arith.addf %[[D1515]], %[[D1517]] : vector<1xf32>
// CHECK:          %[[D1519:.+]] = vector.extract %[[D1348]][0, 4, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1520:.+]] = vector.insert %[[D1519]], %[[D1517]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1521:.+]] = arith.addf %[[D1518]], %[[D1520]] : vector<1xf32>
// CHECK:          %[[D1522:.+]] = vector.extract %[[D1348]][0, 5, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1523:.+]] = vector.insert %[[D1522]], %[[D1520]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1524:.+]] = arith.addf %[[D1521]], %[[D1523]] : vector<1xf32>
// CHECK:          %[[D1525:.+]] = vector.extract %[[D1348]][0, 5, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1526:.+]] = vector.insert %[[D1525]], %[[D1523]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1527:.+]] = arith.addf %[[D1524]], %[[D1526]] : vector<1xf32>
// CHECK:          %[[D1528:.+]] = vector.extract %[[D1348]][0, 6, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1529:.+]] = vector.insert %[[D1528]], %[[D1526]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1530:.+]] = arith.addf %[[D1527]], %[[D1529]] : vector<1xf32>
// CHECK:          %[[D1531:.+]] = vector.extract %[[D1348]][0, 6, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1532:.+]] = vector.insert %[[D1531]], %[[D1529]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1533:.+]] = arith.addf %[[D1530]], %[[D1532]] : vector<1xf32>
// CHECK:          %[[D1534:.+]] = vector.extract %[[D1348]][0, 7, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1535:.+]] = vector.insert %[[D1534]], %[[D1532]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1536:.+]] = arith.addf %[[D1533]], %[[D1535]] : vector<1xf32>
// CHECK:          %[[D1537:.+]] = vector.extract %[[D1348]][0, 7, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1538:.+]] = vector.insert %[[D1537]], %[[D1535]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1539:.+]] = arith.addf %[[D1536]], %[[D1538]] : vector<1xf32>
// CHECK:          %[[D1540:.+]] = vector.extract %[[D1348]][0, 8, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1541:.+]] = vector.insert %[[D1540]], %[[D1538]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1542:.+]] = arith.addf %[[D1539]], %[[D1541]] : vector<1xf32>
// CHECK:          %[[D1543:.+]] = vector.extract %[[D1348]][0, 8, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1544:.+]] = vector.insert %[[D1543]], %[[D1541]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1545:.+]] = arith.addf %[[D1542]], %[[D1544]] : vector<1xf32>
// CHECK:          %[[D1546:.+]] = vector.extract %[[D1348]][0, 9, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1547:.+]] = vector.insert %[[D1546]], %[[D1544]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1548:.+]] = arith.addf %[[D1545]], %[[D1547]] : vector<1xf32>
// CHECK:          %[[D1549:.+]] = vector.extract %[[D1348]][0, 9, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1550:.+]] = vector.insert %[[D1549]], %[[D1547]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1551:.+]] = arith.addf %[[D1548]], %[[D1550]] : vector<1xf32>
// CHECK:          %[[D1552:.+]] = vector.extract %[[D1348]][0, 10, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1553:.+]] = vector.insert %[[D1552]], %[[D1550]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1554:.+]] = arith.addf %[[D1551]], %[[D1553]] : vector<1xf32>
// CHECK:          %[[D1555:.+]] = vector.extract %[[D1348]][0, 10, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1556:.+]] = vector.insert %[[D1555]], %[[D1553]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1557:.+]] = arith.addf %[[D1554]], %[[D1556]] : vector<1xf32>
// CHECK:          %[[D1558:.+]] = vector.extract %[[D1348]][0, 11, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1559:.+]] = vector.insert %[[D1558]], %[[D1556]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1560:.+]] = arith.addf %[[D1557]], %[[D1559]] : vector<1xf32>
// CHECK:          %[[D1561:.+]] = vector.extract %[[D1348]][0, 11, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1562:.+]] = vector.insert %[[D1561]], %[[D1559]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1563:.+]] = arith.addf %[[D1560]], %[[D1562]] : vector<1xf32>
// CHECK:          %[[D1564:.+]] = vector.extract %[[D1348]][0, 12, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1565:.+]] = vector.insert %[[D1564]], %[[D1562]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1566:.+]] = arith.addf %[[D1563]], %[[D1565]] : vector<1xf32>
// CHECK:          %[[D1567:.+]] = vector.extract %[[D1348]][0, 12, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1568:.+]] = vector.insert %[[D1567]], %[[D1565]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1569:.+]] = arith.addf %[[D1566]], %[[D1568]] : vector<1xf32>
// CHECK:          %[[D1570:.+]] = vector.extract %[[D1348]][0, 13, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1571:.+]] = vector.insert %[[D1570]], %[[D1568]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1572:.+]] = arith.addf %[[D1569]], %[[D1571]] : vector<1xf32>
// CHECK:          %[[D1573:.+]] = vector.extract %[[D1348]][0, 13, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1574:.+]] = vector.insert %[[D1573]], %[[D1571]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1575:.+]] = arith.addf %[[D1572]], %[[D1574]] : vector<1xf32>
// CHECK:          %[[D1576:.+]] = vector.extract %[[D1348]][0, 14, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1577:.+]] = vector.insert %[[D1576]], %[[D1574]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1578:.+]] = arith.addf %[[D1575]], %[[D1577]] : vector<1xf32>
// CHECK:          %[[D1579:.+]] = vector.extract %[[D1348]][0, 14, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1580:.+]] = vector.insert %[[D1579]], %[[D1577]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1581:.+]] = arith.addf %[[D1578]], %[[D1580]] : vector<1xf32>
// CHECK:          %[[D1582:.+]] = vector.extract %[[D1348]][0, 15, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1583:.+]] = vector.insert %[[D1582]], %[[D1580]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1584:.+]] = arith.addf %[[D1581]], %[[D1583]] : vector<1xf32>
// CHECK:          %[[D1585:.+]] = vector.extract %[[D1348]][0, 15, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1586:.+]] = vector.insert %[[D1585]], %[[D1583]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1587:.+]] = arith.addf %[[D1584]], %[[D1586]] : vector<1xf32>
// CHECK:          %[[D1588:.+]] = vector.bitcast %[[D1587]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1589:.+]] = vector.extract %[[D1588]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_32:.+]], %[[VALID_33:.+]] = gpu.shuffle  xor %[[D1589]], %[[C1_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1590:.+]] = vector.broadcast %[[SHUFFLERESULT_32]] : i32 to vector<1xi32>
// CHECK:          %[[D1591:.+]] = vector.bitcast %[[D1590]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1592:.+]] = arith.addf %[[D1591]], %[[D1587]] : vector<1xf32>
// CHECK:          %[[D1593:.+]] = vector.bitcast %[[D1592]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1594:.+]] = vector.extract %[[D1593]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_34:.+]], %[[VALID_35:.+]] = gpu.shuffle  xor %[[D1594]], %[[C2_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1595:.+]] = vector.broadcast %[[SHUFFLERESULT_34]] : i32 to vector<1xi32>
// CHECK:          %[[D1596:.+]] = vector.bitcast %[[D1595]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1597:.+]] = arith.addf %[[D1596]], %[[D1592]] : vector<1xf32>
// CHECK:          %[[D1598:.+]] = vector.extract %[[D1597]][0] : vector<1xf32>
// CHECK:          %[[D1599:.+]] = arith.addf %[[D1598]], %[[D1492]] : f32
// CHECK:          %[[D1600:.+]] = vector.insert %[[D1599]], %[[D1491]] [0, 0, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1601:.+]] = vector.insert %[[D1599]], %[[D1600]] [0, 0, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1602:.+]] = vector.insert %[[D1599]], %[[D1601]] [0, 1, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1603:.+]] = vector.insert %[[D1599]], %[[D1602]] [0, 1, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1604:.+]] = vector.insert %[[D1599]], %[[D1603]] [0, 2, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1605:.+]] = vector.insert %[[D1599]], %[[D1604]] [0, 2, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1606:.+]] = vector.insert %[[D1599]], %[[D1605]] [0, 3, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1607:.+]] = vector.insert %[[D1599]], %[[D1606]] [0, 3, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1608:.+]] = vector.insert %[[D1599]], %[[D1607]] [0, 4, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1609:.+]] = vector.insert %[[D1599]], %[[D1608]] [0, 4, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1610:.+]] = vector.insert %[[D1599]], %[[D1609]] [0, 5, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1611:.+]] = vector.insert %[[D1599]], %[[D1610]] [0, 5, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1612:.+]] = vector.insert %[[D1599]], %[[D1611]] [0, 6, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1613:.+]] = vector.insert %[[D1599]], %[[D1612]] [0, 6, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1614:.+]] = vector.insert %[[D1599]], %[[D1613]] [0, 7, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1615:.+]] = vector.insert %[[D1599]], %[[D1614]] [0, 7, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1616:.+]] = vector.insert %[[D1599]], %[[D1615]] [0, 8, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1617:.+]] = vector.insert %[[D1599]], %[[D1616]] [0, 8, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1618:.+]] = vector.insert %[[D1599]], %[[D1617]] [0, 9, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1619:.+]] = vector.insert %[[D1599]], %[[D1618]] [0, 9, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1620:.+]] = vector.insert %[[D1599]], %[[D1619]] [0, 10, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1621:.+]] = vector.insert %[[D1599]], %[[D1620]] [0, 10, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1622:.+]] = vector.insert %[[D1599]], %[[D1621]] [0, 11, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1623:.+]] = vector.insert %[[D1599]], %[[D1622]] [0, 11, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1624:.+]] = vector.insert %[[D1599]], %[[D1623]] [0, 12, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1625:.+]] = vector.insert %[[D1599]], %[[D1624]] [0, 12, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1626:.+]] = vector.insert %[[D1599]], %[[D1625]] [0, 13, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1627:.+]] = vector.insert %[[D1599]], %[[D1626]] [0, 13, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1628:.+]] = vector.insert %[[D1599]], %[[D1627]] [0, 14, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1629:.+]] = vector.insert %[[D1599]], %[[D1628]] [0, 14, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1630:.+]] = vector.insert %[[D1599]], %[[D1629]] [0, 15, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1631:.+]] = vector.insert %[[D1599]], %[[D1630]] [0, 15, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1632:.+]] = vector.extract %[[D1351]][1, 0, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1633:.+]] = vector.extract %[[D1348]][1, 0, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1634:.+]] = vector.insert %[[D1633]], %[[CST_13]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1635:.+]] = vector.extract %[[D1348]][1, 0, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1636:.+]] = vector.insert %[[D1635]], %[[D1634]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1637:.+]] = arith.addf %[[D1634]], %[[D1636]] : vector<1xf32>
// CHECK:          %[[D1638:.+]] = vector.extract %[[D1348]][1, 1, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1639:.+]] = vector.insert %[[D1638]], %[[D1636]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1640:.+]] = arith.addf %[[D1637]], %[[D1639]] : vector<1xf32>
// CHECK:          %[[D1641:.+]] = vector.extract %[[D1348]][1, 1, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1642:.+]] = vector.insert %[[D1641]], %[[D1639]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1643:.+]] = arith.addf %[[D1640]], %[[D1642]] : vector<1xf32>
// CHECK:          %[[D1644:.+]] = vector.extract %[[D1348]][1, 2, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1645:.+]] = vector.insert %[[D1644]], %[[D1642]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1646:.+]] = arith.addf %[[D1643]], %[[D1645]] : vector<1xf32>
// CHECK:          %[[D1647:.+]] = vector.extract %[[D1348]][1, 2, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1648:.+]] = vector.insert %[[D1647]], %[[D1645]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1649:.+]] = arith.addf %[[D1646]], %[[D1648]] : vector<1xf32>
// CHECK:          %[[D1650:.+]] = vector.extract %[[D1348]][1, 3, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1651:.+]] = vector.insert %[[D1650]], %[[D1648]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1652:.+]] = arith.addf %[[D1649]], %[[D1651]] : vector<1xf32>
// CHECK:          %[[D1653:.+]] = vector.extract %[[D1348]][1, 3, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1654:.+]] = vector.insert %[[D1653]], %[[D1651]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1655:.+]] = arith.addf %[[D1652]], %[[D1654]] : vector<1xf32>
// CHECK:          %[[D1656:.+]] = vector.extract %[[D1348]][1, 4, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1657:.+]] = vector.insert %[[D1656]], %[[D1654]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1658:.+]] = arith.addf %[[D1655]], %[[D1657]] : vector<1xf32>
// CHECK:          %[[D1659:.+]] = vector.extract %[[D1348]][1, 4, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1660:.+]] = vector.insert %[[D1659]], %[[D1657]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1661:.+]] = arith.addf %[[D1658]], %[[D1660]] : vector<1xf32>
// CHECK:          %[[D1662:.+]] = vector.extract %[[D1348]][1, 5, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1663:.+]] = vector.insert %[[D1662]], %[[D1660]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1664:.+]] = arith.addf %[[D1661]], %[[D1663]] : vector<1xf32>
// CHECK:          %[[D1665:.+]] = vector.extract %[[D1348]][1, 5, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1666:.+]] = vector.insert %[[D1665]], %[[D1663]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1667:.+]] = arith.addf %[[D1664]], %[[D1666]] : vector<1xf32>
// CHECK:          %[[D1668:.+]] = vector.extract %[[D1348]][1, 6, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1669:.+]] = vector.insert %[[D1668]], %[[D1666]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1670:.+]] = arith.addf %[[D1667]], %[[D1669]] : vector<1xf32>
// CHECK:          %[[D1671:.+]] = vector.extract %[[D1348]][1, 6, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1672:.+]] = vector.insert %[[D1671]], %[[D1669]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1673:.+]] = arith.addf %[[D1670]], %[[D1672]] : vector<1xf32>
// CHECK:          %[[D1674:.+]] = vector.extract %[[D1348]][1, 7, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1675:.+]] = vector.insert %[[D1674]], %[[D1672]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1676:.+]] = arith.addf %[[D1673]], %[[D1675]] : vector<1xf32>
// CHECK:          %[[D1677:.+]] = vector.extract %[[D1348]][1, 7, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1678:.+]] = vector.insert %[[D1677]], %[[D1675]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1679:.+]] = arith.addf %[[D1676]], %[[D1678]] : vector<1xf32>
// CHECK:          %[[D1680:.+]] = vector.extract %[[D1348]][1, 8, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1681:.+]] = vector.insert %[[D1680]], %[[D1678]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1682:.+]] = arith.addf %[[D1679]], %[[D1681]] : vector<1xf32>
// CHECK:          %[[D1683:.+]] = vector.extract %[[D1348]][1, 8, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1684:.+]] = vector.insert %[[D1683]], %[[D1681]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1685:.+]] = arith.addf %[[D1682]], %[[D1684]] : vector<1xf32>
// CHECK:          %[[D1686:.+]] = vector.extract %[[D1348]][1, 9, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1687:.+]] = vector.insert %[[D1686]], %[[D1684]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1688:.+]] = arith.addf %[[D1685]], %[[D1687]] : vector<1xf32>
// CHECK:          %[[D1689:.+]] = vector.extract %[[D1348]][1, 9, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1690:.+]] = vector.insert %[[D1689]], %[[D1687]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1691:.+]] = arith.addf %[[D1688]], %[[D1690]] : vector<1xf32>
// CHECK:          %[[D1692:.+]] = vector.extract %[[D1348]][1, 10, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1693:.+]] = vector.insert %[[D1692]], %[[D1690]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1694:.+]] = arith.addf %[[D1691]], %[[D1693]] : vector<1xf32>
// CHECK:          %[[D1695:.+]] = vector.extract %[[D1348]][1, 10, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1696:.+]] = vector.insert %[[D1695]], %[[D1693]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1697:.+]] = arith.addf %[[D1694]], %[[D1696]] : vector<1xf32>
// CHECK:          %[[D1698:.+]] = vector.extract %[[D1348]][1, 11, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1699:.+]] = vector.insert %[[D1698]], %[[D1696]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1700:.+]] = arith.addf %[[D1697]], %[[D1699]] : vector<1xf32>
// CHECK:          %[[D1701:.+]] = vector.extract %[[D1348]][1, 11, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1702:.+]] = vector.insert %[[D1701]], %[[D1699]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1703:.+]] = arith.addf %[[D1700]], %[[D1702]] : vector<1xf32>
// CHECK:          %[[D1704:.+]] = vector.extract %[[D1348]][1, 12, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1705:.+]] = vector.insert %[[D1704]], %[[D1702]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1706:.+]] = arith.addf %[[D1703]], %[[D1705]] : vector<1xf32>
// CHECK:          %[[D1707:.+]] = vector.extract %[[D1348]][1, 12, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1708:.+]] = vector.insert %[[D1707]], %[[D1705]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1709:.+]] = arith.addf %[[D1706]], %[[D1708]] : vector<1xf32>
// CHECK:          %[[D1710:.+]] = vector.extract %[[D1348]][1, 13, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1711:.+]] = vector.insert %[[D1710]], %[[D1708]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1712:.+]] = arith.addf %[[D1709]], %[[D1711]] : vector<1xf32>
// CHECK:          %[[D1713:.+]] = vector.extract %[[D1348]][1, 13, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1714:.+]] = vector.insert %[[D1713]], %[[D1711]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1715:.+]] = arith.addf %[[D1712]], %[[D1714]] : vector<1xf32>
// CHECK:          %[[D1716:.+]] = vector.extract %[[D1348]][1, 14, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1717:.+]] = vector.insert %[[D1716]], %[[D1714]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1718:.+]] = arith.addf %[[D1715]], %[[D1717]] : vector<1xf32>
// CHECK:          %[[D1719:.+]] = vector.extract %[[D1348]][1, 14, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1720:.+]] = vector.insert %[[D1719]], %[[D1717]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1721:.+]] = arith.addf %[[D1718]], %[[D1720]] : vector<1xf32>
// CHECK:          %[[D1722:.+]] = vector.extract %[[D1348]][1, 15, 0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1723:.+]] = vector.insert %[[D1722]], %[[D1720]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1724:.+]] = arith.addf %[[D1721]], %[[D1723]] : vector<1xf32>
// CHECK:          %[[D1725:.+]] = vector.extract %[[D1348]][1, 15, 0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1726:.+]] = vector.insert %[[D1725]], %[[D1723]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1727:.+]] = arith.addf %[[D1724]], %[[D1726]] : vector<1xf32>
// CHECK:          %[[D1728:.+]] = vector.bitcast %[[D1727]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1729:.+]] = vector.extract %[[D1728]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_36:.+]], %[[VALID_37:.+]] = gpu.shuffle  xor %[[D1729]], %[[C1_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1730:.+]] = vector.broadcast %[[SHUFFLERESULT_36]] : i32 to vector<1xi32>
// CHECK:          %[[D1731:.+]] = vector.bitcast %[[D1730]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1732:.+]] = arith.addf %[[D1731]], %[[D1727]] : vector<1xf32>
// CHECK:          %[[D1733:.+]] = vector.bitcast %[[D1732]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1734:.+]] = vector.extract %[[D1733]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_38:.+]], %[[VALID_39:.+]] = gpu.shuffle  xor %[[D1734]], %[[C2_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1735:.+]] = vector.broadcast %[[SHUFFLERESULT_38]] : i32 to vector<1xi32>
// CHECK:          %[[D1736:.+]] = vector.bitcast %[[D1735]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1737:.+]] = arith.addf %[[D1736]], %[[D1732]] : vector<1xf32>
// CHECK:          %[[D1738:.+]] = vector.extract %[[D1737]][0] : vector<1xf32>
// CHECK:          %[[D1739:.+]] = arith.addf %[[D1738]], %[[D1632]] : f32
// CHECK:          %[[D1740:.+]] = vector.insert %[[D1739]], %[[D1631]] [1, 0, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1741:.+]] = vector.insert %[[D1739]], %[[D1740]] [1, 0, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1742:.+]] = vector.insert %[[D1739]], %[[D1741]] [1, 1, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1743:.+]] = vector.insert %[[D1739]], %[[D1742]] [1, 1, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1744:.+]] = vector.insert %[[D1739]], %[[D1743]] [1, 2, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1745:.+]] = vector.insert %[[D1739]], %[[D1744]] [1, 2, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1746:.+]] = vector.insert %[[D1739]], %[[D1745]] [1, 3, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1747:.+]] = vector.insert %[[D1739]], %[[D1746]] [1, 3, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1748:.+]] = vector.insert %[[D1739]], %[[D1747]] [1, 4, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1749:.+]] = vector.insert %[[D1739]], %[[D1748]] [1, 4, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1750:.+]] = vector.insert %[[D1739]], %[[D1749]] [1, 5, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1751:.+]] = vector.insert %[[D1739]], %[[D1750]] [1, 5, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1752:.+]] = vector.insert %[[D1739]], %[[D1751]] [1, 6, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1753:.+]] = vector.insert %[[D1739]], %[[D1752]] [1, 6, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1754:.+]] = vector.insert %[[D1739]], %[[D1753]] [1, 7, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1755:.+]] = vector.insert %[[D1739]], %[[D1754]] [1, 7, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1756:.+]] = vector.insert %[[D1739]], %[[D1755]] [1, 8, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1757:.+]] = vector.insert %[[D1739]], %[[D1756]] [1, 8, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1758:.+]] = vector.insert %[[D1739]], %[[D1757]] [1, 9, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1759:.+]] = vector.insert %[[D1739]], %[[D1758]] [1, 9, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1760:.+]] = vector.insert %[[D1739]], %[[D1759]] [1, 10, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1761:.+]] = vector.insert %[[D1739]], %[[D1760]] [1, 10, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1762:.+]] = vector.insert %[[D1739]], %[[D1761]] [1, 11, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1763:.+]] = vector.insert %[[D1739]], %[[D1762]] [1, 11, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1764:.+]] = vector.insert %[[D1739]], %[[D1763]] [1, 12, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1765:.+]] = vector.insert %[[D1739]], %[[D1764]] [1, 12, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1766:.+]] = vector.insert %[[D1739]], %[[D1765]] [1, 13, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1767:.+]] = vector.insert %[[D1739]], %[[D1766]] [1, 13, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1768:.+]] = vector.insert %[[D1739]], %[[D1767]] [1, 14, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1769:.+]] = vector.insert %[[D1739]], %[[D1768]] [1, 14, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1770:.+]] = vector.insert %[[D1739]], %[[D1769]] [1, 15, 0, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1771:.+]] = vector.insert %[[D1739]], %[[D1770]] [1, 15, 0, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1772:.+]] = vector.extract %[[D1351]][1, 0, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1773:.+]] = vector.extract %[[D1348]][1, 0, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1774:.+]] = vector.insert %[[D1773]], %[[CST_13]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1775:.+]] = vector.extract %[[D1348]][1, 0, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1776:.+]] = vector.insert %[[D1775]], %[[D1774]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1777:.+]] = arith.addf %[[D1774]], %[[D1776]] : vector<1xf32>
// CHECK:          %[[D1778:.+]] = vector.extract %[[D1348]][1, 1, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1779:.+]] = vector.insert %[[D1778]], %[[D1776]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1780:.+]] = arith.addf %[[D1777]], %[[D1779]] : vector<1xf32>
// CHECK:          %[[D1781:.+]] = vector.extract %[[D1348]][1, 1, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1782:.+]] = vector.insert %[[D1781]], %[[D1779]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1783:.+]] = arith.addf %[[D1780]], %[[D1782]] : vector<1xf32>
// CHECK:          %[[D1784:.+]] = vector.extract %[[D1348]][1, 2, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1785:.+]] = vector.insert %[[D1784]], %[[D1782]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1786:.+]] = arith.addf %[[D1783]], %[[D1785]] : vector<1xf32>
// CHECK:          %[[D1787:.+]] = vector.extract %[[D1348]][1, 2, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1788:.+]] = vector.insert %[[D1787]], %[[D1785]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1789:.+]] = arith.addf %[[D1786]], %[[D1788]] : vector<1xf32>
// CHECK:          %[[D1790:.+]] = vector.extract %[[D1348]][1, 3, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1791:.+]] = vector.insert %[[D1790]], %[[D1788]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1792:.+]] = arith.addf %[[D1789]], %[[D1791]] : vector<1xf32>
// CHECK:          %[[D1793:.+]] = vector.extract %[[D1348]][1, 3, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1794:.+]] = vector.insert %[[D1793]], %[[D1791]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1795:.+]] = arith.addf %[[D1792]], %[[D1794]] : vector<1xf32>
// CHECK:          %[[D1796:.+]] = vector.extract %[[D1348]][1, 4, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1797:.+]] = vector.insert %[[D1796]], %[[D1794]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1798:.+]] = arith.addf %[[D1795]], %[[D1797]] : vector<1xf32>
// CHECK:          %[[D1799:.+]] = vector.extract %[[D1348]][1, 4, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1800:.+]] = vector.insert %[[D1799]], %[[D1797]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1801:.+]] = arith.addf %[[D1798]], %[[D1800]] : vector<1xf32>
// CHECK:          %[[D1802:.+]] = vector.extract %[[D1348]][1, 5, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1803:.+]] = vector.insert %[[D1802]], %[[D1800]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1804:.+]] = arith.addf %[[D1801]], %[[D1803]] : vector<1xf32>
// CHECK:          %[[D1805:.+]] = vector.extract %[[D1348]][1, 5, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1806:.+]] = vector.insert %[[D1805]], %[[D1803]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1807:.+]] = arith.addf %[[D1804]], %[[D1806]] : vector<1xf32>
// CHECK:          %[[D1808:.+]] = vector.extract %[[D1348]][1, 6, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1809:.+]] = vector.insert %[[D1808]], %[[D1806]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1810:.+]] = arith.addf %[[D1807]], %[[D1809]] : vector<1xf32>
// CHECK:          %[[D1811:.+]] = vector.extract %[[D1348]][1, 6, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1812:.+]] = vector.insert %[[D1811]], %[[D1809]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1813:.+]] = arith.addf %[[D1810]], %[[D1812]] : vector<1xf32>
// CHECK:          %[[D1814:.+]] = vector.extract %[[D1348]][1, 7, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1815:.+]] = vector.insert %[[D1814]], %[[D1812]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1816:.+]] = arith.addf %[[D1813]], %[[D1815]] : vector<1xf32>
// CHECK:          %[[D1817:.+]] = vector.extract %[[D1348]][1, 7, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1818:.+]] = vector.insert %[[D1817]], %[[D1815]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1819:.+]] = arith.addf %[[D1816]], %[[D1818]] : vector<1xf32>
// CHECK:          %[[D1820:.+]] = vector.extract %[[D1348]][1, 8, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1821:.+]] = vector.insert %[[D1820]], %[[D1818]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1822:.+]] = arith.addf %[[D1819]], %[[D1821]] : vector<1xf32>
// CHECK:          %[[D1823:.+]] = vector.extract %[[D1348]][1, 8, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1824:.+]] = vector.insert %[[D1823]], %[[D1821]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1825:.+]] = arith.addf %[[D1822]], %[[D1824]] : vector<1xf32>
// CHECK:          %[[D1826:.+]] = vector.extract %[[D1348]][1, 9, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1827:.+]] = vector.insert %[[D1826]], %[[D1824]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1828:.+]] = arith.addf %[[D1825]], %[[D1827]] : vector<1xf32>
// CHECK:          %[[D1829:.+]] = vector.extract %[[D1348]][1, 9, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1830:.+]] = vector.insert %[[D1829]], %[[D1827]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1831:.+]] = arith.addf %[[D1828]], %[[D1830]] : vector<1xf32>
// CHECK:          %[[D1832:.+]] = vector.extract %[[D1348]][1, 10, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1833:.+]] = vector.insert %[[D1832]], %[[D1830]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1834:.+]] = arith.addf %[[D1831]], %[[D1833]] : vector<1xf32>
// CHECK:          %[[D1835:.+]] = vector.extract %[[D1348]][1, 10, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1836:.+]] = vector.insert %[[D1835]], %[[D1833]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1837:.+]] = arith.addf %[[D1834]], %[[D1836]] : vector<1xf32>
// CHECK:          %[[D1838:.+]] = vector.extract %[[D1348]][1, 11, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1839:.+]] = vector.insert %[[D1838]], %[[D1836]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1840:.+]] = arith.addf %[[D1837]], %[[D1839]] : vector<1xf32>
// CHECK:          %[[D1841:.+]] = vector.extract %[[D1348]][1, 11, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1842:.+]] = vector.insert %[[D1841]], %[[D1839]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1843:.+]] = arith.addf %[[D1840]], %[[D1842]] : vector<1xf32>
// CHECK:          %[[D1844:.+]] = vector.extract %[[D1348]][1, 12, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1845:.+]] = vector.insert %[[D1844]], %[[D1842]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1846:.+]] = arith.addf %[[D1843]], %[[D1845]] : vector<1xf32>
// CHECK:          %[[D1847:.+]] = vector.extract %[[D1348]][1, 12, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1848:.+]] = vector.insert %[[D1847]], %[[D1845]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1849:.+]] = arith.addf %[[D1846]], %[[D1848]] : vector<1xf32>
// CHECK:          %[[D1850:.+]] = vector.extract %[[D1348]][1, 13, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1851:.+]] = vector.insert %[[D1850]], %[[D1848]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1852:.+]] = arith.addf %[[D1849]], %[[D1851]] : vector<1xf32>
// CHECK:          %[[D1853:.+]] = vector.extract %[[D1348]][1, 13, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1854:.+]] = vector.insert %[[D1853]], %[[D1851]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1855:.+]] = arith.addf %[[D1852]], %[[D1854]] : vector<1xf32>
// CHECK:          %[[D1856:.+]] = vector.extract %[[D1348]][1, 14, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1857:.+]] = vector.insert %[[D1856]], %[[D1854]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1858:.+]] = arith.addf %[[D1855]], %[[D1857]] : vector<1xf32>
// CHECK:          %[[D1859:.+]] = vector.extract %[[D1348]][1, 14, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1860:.+]] = vector.insert %[[D1859]], %[[D1857]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1861:.+]] = arith.addf %[[D1858]], %[[D1860]] : vector<1xf32>
// CHECK:          %[[D1862:.+]] = vector.extract %[[D1348]][1, 15, 1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1863:.+]] = vector.insert %[[D1862]], %[[D1860]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1864:.+]] = arith.addf %[[D1861]], %[[D1863]] : vector<1xf32>
// CHECK:          %[[D1865:.+]] = vector.extract %[[D1348]][1, 15, 1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1866:.+]] = vector.insert %[[D1865]], %[[D1863]] [0] : f32 into vector<1xf32>
// CHECK:          %[[D1867:.+]] = arith.addf %[[D1864]], %[[D1866]] : vector<1xf32>
// CHECK:          %[[D1868:.+]] = vector.bitcast %[[D1867]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1869:.+]] = vector.extract %[[D1868]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_40:.+]], %[[VALID_41:.+]] = gpu.shuffle  xor %[[D1869]], %[[C1_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1870:.+]] = vector.broadcast %[[SHUFFLERESULT_40]] : i32 to vector<1xi32>
// CHECK:          %[[D1871:.+]] = vector.bitcast %[[D1870]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1872:.+]] = arith.addf %[[D1871]], %[[D1867]] : vector<1xf32>
// CHECK:          %[[D1873:.+]] = vector.bitcast %[[D1872]] : vector<1xf32> to vector<1xi32>
// CHECK:          %[[D1874:.+]] = vector.extract %[[D1873]][0] : vector<1xi32>
// CHECK:          %[[SHUFFLERESULT_42:.+]], %[[VALID_43:.+]] = gpu.shuffle  xor %[[D1874]], %[[C2_I32]], %[[C32_I32]] :
// CHECK-SAME:       i32
// CHECK:          %[[D1875:.+]] = vector.broadcast %[[SHUFFLERESULT_42]] : i32 to vector<1xi32>
// CHECK:          %[[D1876:.+]] = vector.bitcast %[[D1875]] : vector<1xi32> to vector<1xf32>
// CHECK:          %[[D1877:.+]] = arith.addf %[[D1876]], %[[D1872]] : vector<1xf32>
// CHECK:          %[[D1878:.+]] = vector.extract %[[D1877]][0] : vector<1xf32>
// CHECK:          %[[D1879:.+]] = arith.addf %[[D1878]], %[[D1772]] : f32
// CHECK:          %[[D1880:.+]] = vector.insert %[[D1879]], %[[D1771]] [1, 0, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1881:.+]] = vector.insert %[[D1879]], %[[D1880]] [1, 0, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1882:.+]] = vector.insert %[[D1879]], %[[D1881]] [1, 1, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1883:.+]] = vector.insert %[[D1879]], %[[D1882]] [1, 1, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1884:.+]] = vector.insert %[[D1879]], %[[D1883]] [1, 2, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1885:.+]] = vector.insert %[[D1879]], %[[D1884]] [1, 2, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1886:.+]] = vector.insert %[[D1879]], %[[D1885]] [1, 3, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1887:.+]] = vector.insert %[[D1879]], %[[D1886]] [1, 3, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1888:.+]] = vector.insert %[[D1879]], %[[D1887]] [1, 4, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1889:.+]] = vector.insert %[[D1879]], %[[D1888]] [1, 4, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1890:.+]] = vector.insert %[[D1879]], %[[D1889]] [1, 5, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1891:.+]] = vector.insert %[[D1879]], %[[D1890]] [1, 5, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1892:.+]] = vector.insert %[[D1879]], %[[D1891]] [1, 6, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1893:.+]] = vector.insert %[[D1879]], %[[D1892]] [1, 6, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1894:.+]] = vector.insert %[[D1879]], %[[D1893]] [1, 7, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1895:.+]] = vector.insert %[[D1879]], %[[D1894]] [1, 7, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1896:.+]] = vector.insert %[[D1879]], %[[D1895]] [1, 8, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1897:.+]] = vector.insert %[[D1879]], %[[D1896]] [1, 8, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1898:.+]] = vector.insert %[[D1879]], %[[D1897]] [1, 9, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1899:.+]] = vector.insert %[[D1879]], %[[D1898]] [1, 9, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1900:.+]] = vector.insert %[[D1879]], %[[D1899]] [1, 10, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1901:.+]] = vector.insert %[[D1879]], %[[D1900]] [1, 10, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1902:.+]] = vector.insert %[[D1879]], %[[D1901]] [1, 11, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1903:.+]] = vector.insert %[[D1879]], %[[D1902]] [1, 11, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1904:.+]] = vector.insert %[[D1879]], %[[D1903]] [1, 12, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1905:.+]] = vector.insert %[[D1879]], %[[D1904]] [1, 12, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1906:.+]] = vector.insert %[[D1879]], %[[D1905]] [1, 13, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1907:.+]] = vector.insert %[[D1879]], %[[D1906]] [1, 13, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1908:.+]] = vector.insert %[[D1879]], %[[D1907]] [1, 14, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1909:.+]] = vector.insert %[[D1879]], %[[D1908]] [1, 14, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1910:.+]] = vector.insert %[[D1879]], %[[D1909]] [1, 15, 1, 0] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1911:.+]] = vector.insert %[[D1879]], %[[D1910]] [1, 15, 1, 1] : f32 into vector<2x16x2x2xf32>
// CHECK:          %[[D1912:.+]] = arith.mulf %[[D1348]], %[[D1911]] : vector<2x16x2x2xf32>
// CHECK-DAG:      %[[CST_44:.+]] = arith.constant dense<0.000000e+00> : vector<2x8x4x2xf32>
// CHECK:          %[[D1913:.+]] = vector.extract %[[D1912]][0, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1914:.+]] = vector.insert_strided_slice %[[D1913]], %[[CST_44]] {offsets = [0, 0, 0, 0], strides
// CHECK-SAME:       = [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1915:.+]] = vector.extract %[[D1912]][0, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1916:.+]] = vector.insert_strided_slice %[[D1915]], %[[D1914]] {offsets = [0, 0, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1917:.+]] = vector.extract %[[D1912]][0, 2] : vector<2x16x2x2xf32>
// CHECK:          %[[D1918:.+]] = vector.insert_strided_slice %[[D1917]], %[[D1916]] {offsets = [0, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1919:.+]] = vector.extract %[[D1912]][0, 3] : vector<2x16x2x2xf32>
// CHECK:          %[[D1920:.+]] = vector.insert_strided_slice %[[D1919]], %[[D1918]] {offsets = [0, 1, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1921:.+]] = vector.extract %[[D1912]][0, 4] : vector<2x16x2x2xf32>
// CHECK:          %[[D1922:.+]] = vector.insert_strided_slice %[[D1921]], %[[D1920]] {offsets = [0, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1923:.+]] = vector.extract %[[D1912]][0, 5] : vector<2x16x2x2xf32>
// CHECK:          %[[D1924:.+]] = vector.insert_strided_slice %[[D1923]], %[[D1922]] {offsets = [0, 2, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1925:.+]] = vector.extract %[[D1912]][0, 6] : vector<2x16x2x2xf32>
// CHECK:          %[[D1926:.+]] = vector.insert_strided_slice %[[D1925]], %[[D1924]] {offsets = [0, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1927:.+]] = vector.extract %[[D1912]][0, 7] : vector<2x16x2x2xf32>
// CHECK:          %[[D1928:.+]] = vector.insert_strided_slice %[[D1927]], %[[D1926]] {offsets = [0, 3, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1929:.+]] = vector.extract %[[D1912]][0, 8] : vector<2x16x2x2xf32>
// CHECK:          %[[D1930:.+]] = vector.insert_strided_slice %[[D1929]], %[[D1928]] {offsets = [0, 4, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1931:.+]] = vector.extract %[[D1912]][0, 9] : vector<2x16x2x2xf32>
// CHECK:          %[[D1932:.+]] = vector.insert_strided_slice %[[D1931]], %[[D1930]] {offsets = [0, 4, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1933:.+]] = vector.extract %[[D1912]][0, 10] : vector<2x16x2x2xf32>
// CHECK:          %[[D1934:.+]] = vector.insert_strided_slice %[[D1933]], %[[D1932]] {offsets = [0, 5, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1935:.+]] = vector.extract %[[D1912]][0, 11] : vector<2x16x2x2xf32>
// CHECK:          %[[D1936:.+]] = vector.insert_strided_slice %[[D1935]], %[[D1934]] {offsets = [0, 5, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1937:.+]] = vector.extract %[[D1912]][0, 12] : vector<2x16x2x2xf32>
// CHECK:          %[[D1938:.+]] = vector.insert_strided_slice %[[D1937]], %[[D1936]] {offsets = [0, 6, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1939:.+]] = vector.extract %[[D1912]][0, 13] : vector<2x16x2x2xf32>
// CHECK:          %[[D1940:.+]] = vector.insert_strided_slice %[[D1939]], %[[D1938]] {offsets = [0, 6, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1941:.+]] = vector.extract %[[D1912]][0, 14] : vector<2x16x2x2xf32>
// CHECK:          %[[D1942:.+]] = vector.insert_strided_slice %[[D1941]], %[[D1940]] {offsets = [0, 7, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1943:.+]] = vector.extract %[[D1912]][0, 15] : vector<2x16x2x2xf32>
// CHECK:          %[[D1944:.+]] = vector.insert_strided_slice %[[D1943]], %[[D1942]] {offsets = [0, 7, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1945:.+]] = vector.extract %[[D1912]][1, 0] : vector<2x16x2x2xf32>
// CHECK:          %[[D1946:.+]] = vector.insert_strided_slice %[[D1945]], %[[D1944]] {offsets = [1, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1947:.+]] = vector.extract %[[D1912]][1, 1] : vector<2x16x2x2xf32>
// CHECK:          %[[D1948:.+]] = vector.insert_strided_slice %[[D1947]], %[[D1946]] {offsets = [1, 0, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1949:.+]] = vector.extract %[[D1912]][1, 2] : vector<2x16x2x2xf32>
// CHECK:          %[[D1950:.+]] = vector.insert_strided_slice %[[D1949]], %[[D1948]] {offsets = [1, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1951:.+]] = vector.extract %[[D1912]][1, 3] : vector<2x16x2x2xf32>
// CHECK:          %[[D1952:.+]] = vector.insert_strided_slice %[[D1951]], %[[D1950]] {offsets = [1, 1, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1953:.+]] = vector.extract %[[D1912]][1, 4] : vector<2x16x2x2xf32>
// CHECK:          %[[D1954:.+]] = vector.insert_strided_slice %[[D1953]], %[[D1952]] {offsets = [1, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1955:.+]] = vector.extract %[[D1912]][1, 5] : vector<2x16x2x2xf32>
// CHECK:          %[[D1956:.+]] = vector.insert_strided_slice %[[D1955]], %[[D1954]] {offsets = [1, 2, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1957:.+]] = vector.extract %[[D1912]][1, 6] : vector<2x16x2x2xf32>
// CHECK:          %[[D1958:.+]] = vector.insert_strided_slice %[[D1957]], %[[D1956]] {offsets = [1, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1959:.+]] = vector.extract %[[D1912]][1, 7] : vector<2x16x2x2xf32>
// CHECK:          %[[D1960:.+]] = vector.insert_strided_slice %[[D1959]], %[[D1958]] {offsets = [1, 3, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1961:.+]] = vector.extract %[[D1912]][1, 8] : vector<2x16x2x2xf32>
// CHECK:          %[[D1962:.+]] = vector.insert_strided_slice %[[D1961]], %[[D1960]] {offsets = [1, 4, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1963:.+]] = vector.extract %[[D1912]][1, 9] : vector<2x16x2x2xf32>
// CHECK:          %[[D1964:.+]] = vector.insert_strided_slice %[[D1963]], %[[D1962]] {offsets = [1, 4, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1965:.+]] = vector.extract %[[D1912]][1, 10] : vector<2x16x2x2xf32>
// CHECK:          %[[D1966:.+]] = vector.insert_strided_slice %[[D1965]], %[[D1964]] {offsets = [1, 5, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1967:.+]] = vector.extract %[[D1912]][1, 11] : vector<2x16x2x2xf32>
// CHECK:          %[[D1968:.+]] = vector.insert_strided_slice %[[D1967]], %[[D1966]] {offsets = [1, 5, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1969:.+]] = vector.extract %[[D1912]][1, 12] : vector<2x16x2x2xf32>
// CHECK:          %[[D1970:.+]] = vector.insert_strided_slice %[[D1969]], %[[D1968]] {offsets = [1, 6, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1971:.+]] = vector.extract %[[D1912]][1, 13] : vector<2x16x2x2xf32>
// CHECK:          %[[D1972:.+]] = vector.insert_strided_slice %[[D1971]], %[[D1970]] {offsets = [1, 6, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1973:.+]] = vector.extract %[[D1912]][1, 14] : vector<2x16x2x2xf32>
// CHECK:          %[[D1974:.+]] = vector.insert_strided_slice %[[D1973]], %[[D1972]] {offsets = [1, 7, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1975:.+]] = vector.extract %[[D1912]][1, 15] : vector<2x16x2x2xf32>
// CHECK:          %[[D1976:.+]] = vector.insert_strided_slice %[[D1975]], %[[D1974]] {offsets = [1, 7, 2, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<2x2xf32> into vector<2x8x4x2xf32>
// CHECK:          %[[D1977:.+]] = arith.truncf %[[D1976]] : vector<2x8x4x2xf32> to vector<2x8x4x2xf16>
// CHECK:          %[[D1978:.+]] = arith.mulf %[[D1351]], %[[D1911]] : vector<2x16x2x2xf32>
// CHECK:          %[[D1979:.+]] = vector.extract_strided_slice %[[D1978]] {offsets = [0, 0, 0, 0], sizes = [2, 8, 2,
// CHECK-SAME:       2], strides = [1, 1, 1, 1]} : vector<2x16x2x2xf32> to vector<2x8x2x2xf32>
// CHECK:          %[[D1980:.+]] = arith.mulf %[[D1979]], %[[ARG6]] : vector<2x8x2x2xf32>
// CHECK-DAG:      %[[CST_45:.+]] = arith.constant dense<0.000000e+00> : vector<8x8x2x2xf16>
// CHECK:          %[[D1981:.+]] = arith.addi %[[D13]], %[[D220]] : index
// CHECK:          %[[D1982:.+]] = arith.addi %[[D219]], %[[D16]] : index
// CHECK:          %[[D1983:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1982]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D1984:.+]] = vector.insert_strided_slice %[[D1983]], %[[CST_45]] {offsets = [0, 0, 0, 0], strides
// CHECK-SAME:       = [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D1985:.+]] = arith.addi %[[D219]], %[[D21]] : index
// CHECK:          %[[D1986:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1985]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D1987:.+]] = vector.insert_strided_slice %[[D1986]], %[[D1984]] {offsets = [0, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D1988:.+]] = arith.addi %[[D219]], %[[D33]] : index
// CHECK:          %[[D1989:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1988]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D1990:.+]] = vector.insert_strided_slice %[[D1989]], %[[D1987]] {offsets = [0, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D1991:.+]] = arith.addi %[[D219]], %[[D38]] : index
// CHECK:          %[[D1992:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1991]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D1993:.+]] = vector.insert_strided_slice %[[D1992]], %[[D1990]] {offsets = [0, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D1994:.+]] = arith.addi %[[D219]], %[[D47]] : index
// CHECK:          %[[D1995:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1994]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D1996:.+]] = vector.insert_strided_slice %[[D1995]], %[[D1993]] {offsets = [0, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D1997:.+]] = arith.addi %[[D219]], %[[D52]] : index
// CHECK:          %[[D1998:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1997]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D1999:.+]] = vector.insert_strided_slice %[[D1998]], %[[D1996]] {offsets = [0, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2000:.+]] = arith.addi %[[D219]], %[[D61]] : index
// CHECK:          %[[D2001:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2000]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2002:.+]] = vector.insert_strided_slice %[[D2001]], %[[D1999]] {offsets = [0, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2003:.+]] = arith.addi %[[D219]], %[[D66]] : index
// CHECK:          %[[D2004:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2003]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2005:.+]] = vector.insert_strided_slice %[[D2004]], %[[D2002]] {offsets = [0, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK-DAG:      %[[D2006:.+]] = affine.apply #[[MAP33]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D2007:.+]] = arith.addi %[[D2006]], %[[C0]] : index
// CHECK:          %[[D2008:.+]] = arith.addi %[[D219]], %[[D2007]] : index
// CHECK:          %[[D2009:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2008]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2010:.+]] = vector.insert_strided_slice %[[D2009]], %[[D2005]] {offsets = [0, 4, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK-DAG:      %[[D2011:.+]] = affine.apply #[[MAP34]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D2012:.+]] = arith.addi %[[D2011]], %[[C0]] : index
// CHECK:          %[[D2013:.+]] = arith.addi %[[D219]], %[[D2012]] : index
// CHECK:          %[[D2014:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2013]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2015:.+]] = vector.insert_strided_slice %[[D2014]], %[[D2010]] {offsets = [0, 4, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK-DAG:      %[[D2016:.+]] = affine.apply #[[MAP35]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D2017:.+]] = arith.addi %[[D2016]], %[[C0]] : index
// CHECK:          %[[D2018:.+]] = arith.addi %[[D219]], %[[D2017]] : index
// CHECK:          %[[D2019:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2018]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2020:.+]] = vector.insert_strided_slice %[[D2019]], %[[D2015]] {offsets = [0, 5, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK-DAG:      %[[D2021:.+]] = affine.apply #[[MAP36]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D2022:.+]] = arith.addi %[[D2021]], %[[C0]] : index
// CHECK:          %[[D2023:.+]] = arith.addi %[[D219]], %[[D2022]] : index
// CHECK:          %[[D2024:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2023]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2025:.+]] = vector.insert_strided_slice %[[D2024]], %[[D2020]] {offsets = [0, 5, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK-DAG:      %[[D2026:.+]] = affine.apply #[[MAP37]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D2027:.+]] = arith.addi %[[D2026]], %[[C0]] : index
// CHECK:          %[[D2028:.+]] = arith.addi %[[D219]], %[[D2027]] : index
// CHECK:          %[[D2029:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2028]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2030:.+]] = vector.insert_strided_slice %[[D2029]], %[[D2025]] {offsets = [0, 6, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK-DAG:      %[[D2031:.+]] = affine.apply #[[MAP38]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D2032:.+]] = arith.addi %[[D2031]], %[[C0]] : index
// CHECK:          %[[D2033:.+]] = arith.addi %[[D219]], %[[D2032]] : index
// CHECK:          %[[D2034:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2033]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2035:.+]] = vector.insert_strided_slice %[[D2034]], %[[D2030]] {offsets = [0, 6, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK-DAG:      %[[D2036:.+]] = affine.apply #[[MAP39]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D2037:.+]] = arith.addi %[[D2036]], %[[C0]] : index
// CHECK:          %[[D2038:.+]] = arith.addi %[[D219]], %[[D2037]] : index
// CHECK:          %[[D2039:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2038]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2040:.+]] = vector.insert_strided_slice %[[D2039]], %[[D2035]] {offsets = [0, 7, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK-DAG:      %[[D2041:.+]] = affine.apply #[[MAP40]](%[[C0]], %[[C0]], %[[C0]])
// CHECK:          %[[D2042:.+]] = arith.addi %[[D2041]], %[[C0]] : index
// CHECK:          %[[D2043:.+]] = arith.addi %[[D219]], %[[D2042]] : index
// CHECK:          %[[D2044:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2043]], %[[D1981]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2045:.+]] = vector.insert_strided_slice %[[D2044]], %[[D2040]] {offsets = [0, 7, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2046:.+]] = arith.addi %[[D13]], %[[D239]] : index
// CHECK:          %[[D2047:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1982]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2048:.+]] = vector.insert_strided_slice %[[D2047]], %[[D2045]] {offsets = [1, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2049:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1985]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2050:.+]] = vector.insert_strided_slice %[[D2049]], %[[D2048]] {offsets = [1, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2051:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1988]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2052:.+]] = vector.insert_strided_slice %[[D2051]], %[[D2050]] {offsets = [1, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2053:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1991]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2054:.+]] = vector.insert_strided_slice %[[D2053]], %[[D2052]] {offsets = [1, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2055:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1994]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2056:.+]] = vector.insert_strided_slice %[[D2055]], %[[D2054]] {offsets = [1, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2057:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1997]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2058:.+]] = vector.insert_strided_slice %[[D2057]], %[[D2056]] {offsets = [1, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2059:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2000]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2060:.+]] = vector.insert_strided_slice %[[D2059]], %[[D2058]] {offsets = [1, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2061:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2003]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2062:.+]] = vector.insert_strided_slice %[[D2061]], %[[D2060]] {offsets = [1, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2063:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2008]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2064:.+]] = vector.insert_strided_slice %[[D2063]], %[[D2062]] {offsets = [1, 4, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2065:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2013]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2066:.+]] = vector.insert_strided_slice %[[D2065]], %[[D2064]] {offsets = [1, 4, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2067:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2018]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2068:.+]] = vector.insert_strided_slice %[[D2067]], %[[D2066]] {offsets = [1, 5, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2069:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2023]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2070:.+]] = vector.insert_strided_slice %[[D2069]], %[[D2068]] {offsets = [1, 5, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2071:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2028]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2072:.+]] = vector.insert_strided_slice %[[D2071]], %[[D2070]] {offsets = [1, 6, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2073:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2033]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2074:.+]] = vector.insert_strided_slice %[[D2073]], %[[D2072]] {offsets = [1, 6, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2075:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2038]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2076:.+]] = vector.insert_strided_slice %[[D2075]], %[[D2074]] {offsets = [1, 7, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2077:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2043]], %[[D2046]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2078:.+]] = vector.insert_strided_slice %[[D2077]], %[[D2076]] {offsets = [1, 7, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2079:.+]] = arith.addi %[[D13]], %[[D258]] : index
// CHECK:          %[[D2080:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1982]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2081:.+]] = vector.insert_strided_slice %[[D2080]], %[[D2078]] {offsets = [2, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2082:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1985]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2083:.+]] = vector.insert_strided_slice %[[D2082]], %[[D2081]] {offsets = [2, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2084:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1988]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2085:.+]] = vector.insert_strided_slice %[[D2084]], %[[D2083]] {offsets = [2, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2086:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1991]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2087:.+]] = vector.insert_strided_slice %[[D2086]], %[[D2085]] {offsets = [2, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2088:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1994]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2089:.+]] = vector.insert_strided_slice %[[D2088]], %[[D2087]] {offsets = [2, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2090:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1997]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2091:.+]] = vector.insert_strided_slice %[[D2090]], %[[D2089]] {offsets = [2, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2092:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2000]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2093:.+]] = vector.insert_strided_slice %[[D2092]], %[[D2091]] {offsets = [2, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2094:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2003]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2095:.+]] = vector.insert_strided_slice %[[D2094]], %[[D2093]] {offsets = [2, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2096:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2008]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2097:.+]] = vector.insert_strided_slice %[[D2096]], %[[D2095]] {offsets = [2, 4, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2098:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2013]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2099:.+]] = vector.insert_strided_slice %[[D2098]], %[[D2097]] {offsets = [2, 4, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2100:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2018]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2101:.+]] = vector.insert_strided_slice %[[D2100]], %[[D2099]] {offsets = [2, 5, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2102:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2023]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2103:.+]] = vector.insert_strided_slice %[[D2102]], %[[D2101]] {offsets = [2, 5, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2104:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2028]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2105:.+]] = vector.insert_strided_slice %[[D2104]], %[[D2103]] {offsets = [2, 6, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2106:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2033]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2107:.+]] = vector.insert_strided_slice %[[D2106]], %[[D2105]] {offsets = [2, 6, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2108:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2038]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2109:.+]] = vector.insert_strided_slice %[[D2108]], %[[D2107]] {offsets = [2, 7, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2110:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2043]], %[[D2079]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2111:.+]] = vector.insert_strided_slice %[[D2110]], %[[D2109]] {offsets = [2, 7, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2112:.+]] = arith.addi %[[D13]], %[[D277]] : index
// CHECK:          %[[D2113:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1982]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2114:.+]] = vector.insert_strided_slice %[[D2113]], %[[D2111]] {offsets = [3, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2115:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1985]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2116:.+]] = vector.insert_strided_slice %[[D2115]], %[[D2114]] {offsets = [3, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2117:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1988]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2118:.+]] = vector.insert_strided_slice %[[D2117]], %[[D2116]] {offsets = [3, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2119:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1991]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2120:.+]] = vector.insert_strided_slice %[[D2119]], %[[D2118]] {offsets = [3, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2121:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1994]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2122:.+]] = vector.insert_strided_slice %[[D2121]], %[[D2120]] {offsets = [3, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2123:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1997]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2124:.+]] = vector.insert_strided_slice %[[D2123]], %[[D2122]] {offsets = [3, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2125:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2000]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2126:.+]] = vector.insert_strided_slice %[[D2125]], %[[D2124]] {offsets = [3, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2127:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2003]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2128:.+]] = vector.insert_strided_slice %[[D2127]], %[[D2126]] {offsets = [3, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2129:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2008]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2130:.+]] = vector.insert_strided_slice %[[D2129]], %[[D2128]] {offsets = [3, 4, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2131:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2013]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2132:.+]] = vector.insert_strided_slice %[[D2131]], %[[D2130]] {offsets = [3, 4, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2133:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2018]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2134:.+]] = vector.insert_strided_slice %[[D2133]], %[[D2132]] {offsets = [3, 5, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2135:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2023]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2136:.+]] = vector.insert_strided_slice %[[D2135]], %[[D2134]] {offsets = [3, 5, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2137:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2028]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2138:.+]] = vector.insert_strided_slice %[[D2137]], %[[D2136]] {offsets = [3, 6, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2139:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2033]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2140:.+]] = vector.insert_strided_slice %[[D2139]], %[[D2138]] {offsets = [3, 6, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2141:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2038]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2142:.+]] = vector.insert_strided_slice %[[D2141]], %[[D2140]] {offsets = [3, 7, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2143:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2043]], %[[D2112]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2144:.+]] = vector.insert_strided_slice %[[D2143]], %[[D2142]] {offsets = [3, 7, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2145:.+]] = arith.addi %[[D13]], %[[D296]] : index
// CHECK:          %[[D2146:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1982]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2147:.+]] = vector.insert_strided_slice %[[D2146]], %[[D2144]] {offsets = [4, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2148:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1985]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2149:.+]] = vector.insert_strided_slice %[[D2148]], %[[D2147]] {offsets = [4, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2150:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1988]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2151:.+]] = vector.insert_strided_slice %[[D2150]], %[[D2149]] {offsets = [4, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2152:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1991]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2153:.+]] = vector.insert_strided_slice %[[D2152]], %[[D2151]] {offsets = [4, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2154:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1994]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2155:.+]] = vector.insert_strided_slice %[[D2154]], %[[D2153]] {offsets = [4, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2156:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1997]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2157:.+]] = vector.insert_strided_slice %[[D2156]], %[[D2155]] {offsets = [4, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2158:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2000]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2159:.+]] = vector.insert_strided_slice %[[D2158]], %[[D2157]] {offsets = [4, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2160:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2003]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2161:.+]] = vector.insert_strided_slice %[[D2160]], %[[D2159]] {offsets = [4, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2162:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2008]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2163:.+]] = vector.insert_strided_slice %[[D2162]], %[[D2161]] {offsets = [4, 4, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2164:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2013]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2165:.+]] = vector.insert_strided_slice %[[D2164]], %[[D2163]] {offsets = [4, 4, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2166:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2018]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2167:.+]] = vector.insert_strided_slice %[[D2166]], %[[D2165]] {offsets = [4, 5, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2168:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2023]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2169:.+]] = vector.insert_strided_slice %[[D2168]], %[[D2167]] {offsets = [4, 5, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2170:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2028]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2171:.+]] = vector.insert_strided_slice %[[D2170]], %[[D2169]] {offsets = [4, 6, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2172:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2033]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2173:.+]] = vector.insert_strided_slice %[[D2172]], %[[D2171]] {offsets = [4, 6, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2174:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2038]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2175:.+]] = vector.insert_strided_slice %[[D2174]], %[[D2173]] {offsets = [4, 7, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2176:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2043]], %[[D2145]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2177:.+]] = vector.insert_strided_slice %[[D2176]], %[[D2175]] {offsets = [4, 7, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2178:.+]] = arith.addi %[[D13]], %[[D315]] : index
// CHECK:          %[[D2179:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1982]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2180:.+]] = vector.insert_strided_slice %[[D2179]], %[[D2177]] {offsets = [5, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2181:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1985]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2182:.+]] = vector.insert_strided_slice %[[D2181]], %[[D2180]] {offsets = [5, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2183:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1988]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2184:.+]] = vector.insert_strided_slice %[[D2183]], %[[D2182]] {offsets = [5, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2185:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1991]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2186:.+]] = vector.insert_strided_slice %[[D2185]], %[[D2184]] {offsets = [5, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2187:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1994]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2188:.+]] = vector.insert_strided_slice %[[D2187]], %[[D2186]] {offsets = [5, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2189:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1997]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2190:.+]] = vector.insert_strided_slice %[[D2189]], %[[D2188]] {offsets = [5, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2191:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2000]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2192:.+]] = vector.insert_strided_slice %[[D2191]], %[[D2190]] {offsets = [5, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2193:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2003]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2194:.+]] = vector.insert_strided_slice %[[D2193]], %[[D2192]] {offsets = [5, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2195:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2008]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2196:.+]] = vector.insert_strided_slice %[[D2195]], %[[D2194]] {offsets = [5, 4, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2197:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2013]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2198:.+]] = vector.insert_strided_slice %[[D2197]], %[[D2196]] {offsets = [5, 4, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2199:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2018]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2200:.+]] = vector.insert_strided_slice %[[D2199]], %[[D2198]] {offsets = [5, 5, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2201:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2023]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2202:.+]] = vector.insert_strided_slice %[[D2201]], %[[D2200]] {offsets = [5, 5, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2203:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2028]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2204:.+]] = vector.insert_strided_slice %[[D2203]], %[[D2202]] {offsets = [5, 6, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2205:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2033]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2206:.+]] = vector.insert_strided_slice %[[D2205]], %[[D2204]] {offsets = [5, 6, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2207:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2038]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2208:.+]] = vector.insert_strided_slice %[[D2207]], %[[D2206]] {offsets = [5, 7, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2209:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2043]], %[[D2178]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2210:.+]] = vector.insert_strided_slice %[[D2209]], %[[D2208]] {offsets = [5, 7, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2211:.+]] = arith.addi %[[D13]], %[[D334]] : index
// CHECK:          %[[D2212:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1982]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2213:.+]] = vector.insert_strided_slice %[[D2212]], %[[D2210]] {offsets = [6, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2214:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1985]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2215:.+]] = vector.insert_strided_slice %[[D2214]], %[[D2213]] {offsets = [6, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2216:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1988]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2217:.+]] = vector.insert_strided_slice %[[D2216]], %[[D2215]] {offsets = [6, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2218:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1991]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2219:.+]] = vector.insert_strided_slice %[[D2218]], %[[D2217]] {offsets = [6, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2220:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1994]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2221:.+]] = vector.insert_strided_slice %[[D2220]], %[[D2219]] {offsets = [6, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2222:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1997]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2223:.+]] = vector.insert_strided_slice %[[D2222]], %[[D2221]] {offsets = [6, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2224:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2000]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2225:.+]] = vector.insert_strided_slice %[[D2224]], %[[D2223]] {offsets = [6, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2226:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2003]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2227:.+]] = vector.insert_strided_slice %[[D2226]], %[[D2225]] {offsets = [6, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2228:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2008]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2229:.+]] = vector.insert_strided_slice %[[D2228]], %[[D2227]] {offsets = [6, 4, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2230:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2013]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2231:.+]] = vector.insert_strided_slice %[[D2230]], %[[D2229]] {offsets = [6, 4, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2232:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2018]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2233:.+]] = vector.insert_strided_slice %[[D2232]], %[[D2231]] {offsets = [6, 5, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2234:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2023]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2235:.+]] = vector.insert_strided_slice %[[D2234]], %[[D2233]] {offsets = [6, 5, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2236:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2028]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2237:.+]] = vector.insert_strided_slice %[[D2236]], %[[D2235]] {offsets = [6, 6, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2238:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2033]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2239:.+]] = vector.insert_strided_slice %[[D2238]], %[[D2237]] {offsets = [6, 6, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2240:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2038]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2241:.+]] = vector.insert_strided_slice %[[D2240]], %[[D2239]] {offsets = [6, 7, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2242:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2043]], %[[D2211]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2243:.+]] = vector.insert_strided_slice %[[D2242]], %[[D2241]] {offsets = [6, 7, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2244:.+]] = arith.addi %[[D13]], %[[D353]] : index
// CHECK:          %[[D2245:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1982]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2246:.+]] = vector.insert_strided_slice %[[D2245]], %[[D2243]] {offsets = [7, 0, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2247:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1985]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2248:.+]] = vector.insert_strided_slice %[[D2247]], %[[D2246]] {offsets = [7, 0, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2249:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1988]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2250:.+]] = vector.insert_strided_slice %[[D2249]], %[[D2248]] {offsets = [7, 1, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2251:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1991]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2252:.+]] = vector.insert_strided_slice %[[D2251]], %[[D2250]] {offsets = [7, 1, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2253:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1994]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2254:.+]] = vector.insert_strided_slice %[[D2253]], %[[D2252]] {offsets = [7, 2, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2255:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D1997]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2256:.+]] = vector.insert_strided_slice %[[D2255]], %[[D2254]] {offsets = [7, 2, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2257:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2000]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2258:.+]] = vector.insert_strided_slice %[[D2257]], %[[D2256]] {offsets = [7, 3, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2259:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2003]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2260:.+]] = vector.insert_strided_slice %[[D2259]], %[[D2258]] {offsets = [7, 3, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2261:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2008]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2262:.+]] = vector.insert_strided_slice %[[D2261]], %[[D2260]] {offsets = [7, 4, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2263:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2013]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2264:.+]] = vector.insert_strided_slice %[[D2263]], %[[D2262]] {offsets = [7, 4, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2265:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2018]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2266:.+]] = vector.insert_strided_slice %[[D2265]], %[[D2264]] {offsets = [7, 5, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2267:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2023]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2268:.+]] = vector.insert_strided_slice %[[D2267]], %[[D2266]] {offsets = [7, 5, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2269:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2028]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2270:.+]] = vector.insert_strided_slice %[[D2269]], %[[D2268]] {offsets = [7, 6, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2271:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2033]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2272:.+]] = vector.insert_strided_slice %[[D2271]], %[[D2270]] {offsets = [7, 6, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2273:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2038]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2274:.+]] = vector.insert_strided_slice %[[D2273]], %[[D2272]] {offsets = [7, 7, 0, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2275:.+]] = nvgpu.ldmatrix %[[ALLOC_11]][%[[D2043]], %[[D2244]]] {numTiles = 1 : i32, transpose =
// CHECK-SAME:       true} : memref<128x64xf16, #[[GPU]].address_space<workgroup>> -> vector<1x2xf16>
// CHECK:          %[[D2276:.+]] = vector.insert_strided_slice %[[D2275]], %[[D2274]] {offsets = [7, 7, 1, 0], strides =
// CHECK-SAME:       [1, 1]} : vector<1x2xf16> into vector<8x8x2x2xf16>
// CHECK:          %[[D2277:.+]] = vector.extract %[[D1980]][0, 0] : vector<2x8x2x2xf32>
// CHECK:          %[[D2278:.+]] = vector.extract %[[D1977]][0, 0] : vector<2x8x4x2xf16>
// CHECK:          %[[D2279:.+]] = vector.extract %[[D2276]][0, 0] : vector<8x8x2x2xf16>
// CHECK:          %[[D2280:.+]] = nvgpu.mma.sync(%[[D2278]], %[[D2279]], %[[D2277]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2281:.+]] = vector.extract %[[D1977]][0, 1] : vector<2x8x4x2xf16>
// CHECK:          %[[D2282:.+]] = vector.extract %[[D2276]][0, 1] : vector<8x8x2x2xf16>
// CHECK:          %[[D2283:.+]] = nvgpu.mma.sync(%[[D2281]], %[[D2282]], %[[D2280]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2284:.+]] = vector.extract %[[D1977]][0, 2] : vector<2x8x4x2xf16>
// CHECK:          %[[D2285:.+]] = vector.extract %[[D2276]][0, 2] : vector<8x8x2x2xf16>
// CHECK:          %[[D2286:.+]] = nvgpu.mma.sync(%[[D2284]], %[[D2285]], %[[D2283]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2287:.+]] = vector.extract %[[D1977]][0, 3] : vector<2x8x4x2xf16>
// CHECK:          %[[D2288:.+]] = vector.extract %[[D2276]][0, 3] : vector<8x8x2x2xf16>
// CHECK:          %[[D2289:.+]] = nvgpu.mma.sync(%[[D2287]], %[[D2288]], %[[D2286]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2290:.+]] = vector.extract %[[D1977]][0, 4] : vector<2x8x4x2xf16>
// CHECK:          %[[D2291:.+]] = vector.extract %[[D2276]][0, 4] : vector<8x8x2x2xf16>
// CHECK:          %[[D2292:.+]] = nvgpu.mma.sync(%[[D2290]], %[[D2291]], %[[D2289]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2293:.+]] = vector.extract %[[D1977]][0, 5] : vector<2x8x4x2xf16>
// CHECK:          %[[D2294:.+]] = vector.extract %[[D2276]][0, 5] : vector<8x8x2x2xf16>
// CHECK:          %[[D2295:.+]] = nvgpu.mma.sync(%[[D2293]], %[[D2294]], %[[D2292]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2296:.+]] = vector.extract %[[D1977]][0, 6] : vector<2x8x4x2xf16>
// CHECK:          %[[D2297:.+]] = vector.extract %[[D2276]][0, 6] : vector<8x8x2x2xf16>
// CHECK:          %[[D2298:.+]] = nvgpu.mma.sync(%[[D2296]], %[[D2297]], %[[D2295]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2299:.+]] = vector.extract %[[D1977]][0, 7] : vector<2x8x4x2xf16>
// CHECK:          %[[D2300:.+]] = vector.extract %[[D2276]][0, 7] : vector<8x8x2x2xf16>
// CHECK:          %[[D2301:.+]] = nvgpu.mma.sync(%[[D2299]], %[[D2300]], %[[D2298]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2302:.+]] = vector.insert %[[D2301]], %[[CST]] [0, 0] : vector<2x2xf32> into vector<2x8x2x2xf32>
// CHECK:          %[[D2303:.+]] = vector.extract %[[D1980]][0, 1] : vector<2x8x2x2xf32>
// CHECK:          %[[D2304:.+]] = vector.extract %[[D2276]][1, 0] : vector<8x8x2x2xf16>
// CHECK:          %[[D2305:.+]] = nvgpu.mma.sync(%[[D2278]], %[[D2304]], %[[D2303]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2306:.+]] = vector.extract %[[D2276]][1, 1] : vector<8x8x2x2xf16>
// CHECK:          %[[D2307:.+]] = nvgpu.mma.sync(%[[D2281]], %[[D2306]], %[[D2305]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2308:.+]] = vector.extract %[[D2276]][1, 2] : vector<8x8x2x2xf16>
// CHECK:          %[[D2309:.+]] = nvgpu.mma.sync(%[[D2284]], %[[D2308]], %[[D2307]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2310:.+]] = vector.extract %[[D2276]][1, 3] : vector<8x8x2x2xf16>
// CHECK:          %[[D2311:.+]] = nvgpu.mma.sync(%[[D2287]], %[[D2310]], %[[D2309]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2312:.+]] = vector.extract %[[D2276]][1, 4] : vector<8x8x2x2xf16>
// CHECK:          %[[D2313:.+]] = nvgpu.mma.sync(%[[D2290]], %[[D2312]], %[[D2311]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2314:.+]] = vector.extract %[[D2276]][1, 5] : vector<8x8x2x2xf16>
// CHECK:          %[[D2315:.+]] = nvgpu.mma.sync(%[[D2293]], %[[D2314]], %[[D2313]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2316:.+]] = vector.extract %[[D2276]][1, 6] : vector<8x8x2x2xf16>
// CHECK:          %[[D2317:.+]] = nvgpu.mma.sync(%[[D2296]], %[[D2316]], %[[D2315]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2318:.+]] = vector.extract %[[D2276]][1, 7] : vector<8x8x2x2xf16>
// CHECK:          %[[D2319:.+]] = nvgpu.mma.sync(%[[D2299]], %[[D2318]], %[[D2317]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2320:.+]] = vector.insert %[[D2319]], %[[D2302]] [0, 1] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2321:.+]] = vector.extract %[[D1980]][0, 2] : vector<2x8x2x2xf32>
// CHECK:          %[[D2322:.+]] = vector.extract %[[D2276]][2, 0] : vector<8x8x2x2xf16>
// CHECK:          %[[D2323:.+]] = nvgpu.mma.sync(%[[D2278]], %[[D2322]], %[[D2321]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2324:.+]] = vector.extract %[[D2276]][2, 1] : vector<8x8x2x2xf16>
// CHECK:          %[[D2325:.+]] = nvgpu.mma.sync(%[[D2281]], %[[D2324]], %[[D2323]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2326:.+]] = vector.extract %[[D2276]][2, 2] : vector<8x8x2x2xf16>
// CHECK:          %[[D2327:.+]] = nvgpu.mma.sync(%[[D2284]], %[[D2326]], %[[D2325]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2328:.+]] = vector.extract %[[D2276]][2, 3] : vector<8x8x2x2xf16>
// CHECK:          %[[D2329:.+]] = nvgpu.mma.sync(%[[D2287]], %[[D2328]], %[[D2327]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2330:.+]] = vector.extract %[[D2276]][2, 4] : vector<8x8x2x2xf16>
// CHECK:          %[[D2331:.+]] = nvgpu.mma.sync(%[[D2290]], %[[D2330]], %[[D2329]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2332:.+]] = vector.extract %[[D2276]][2, 5] : vector<8x8x2x2xf16>
// CHECK:          %[[D2333:.+]] = nvgpu.mma.sync(%[[D2293]], %[[D2332]], %[[D2331]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2334:.+]] = vector.extract %[[D2276]][2, 6] : vector<8x8x2x2xf16>
// CHECK:          %[[D2335:.+]] = nvgpu.mma.sync(%[[D2296]], %[[D2334]], %[[D2333]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2336:.+]] = vector.extract %[[D2276]][2, 7] : vector<8x8x2x2xf16>
// CHECK:          %[[D2337:.+]] = nvgpu.mma.sync(%[[D2299]], %[[D2336]], %[[D2335]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2338:.+]] = vector.insert %[[D2337]], %[[D2320]] [0, 2] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2339:.+]] = vector.extract %[[D1980]][0, 3] : vector<2x8x2x2xf32>
// CHECK:          %[[D2340:.+]] = vector.extract %[[D2276]][3, 0] : vector<8x8x2x2xf16>
// CHECK:          %[[D2341:.+]] = nvgpu.mma.sync(%[[D2278]], %[[D2340]], %[[D2339]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2342:.+]] = vector.extract %[[D2276]][3, 1] : vector<8x8x2x2xf16>
// CHECK:          %[[D2343:.+]] = nvgpu.mma.sync(%[[D2281]], %[[D2342]], %[[D2341]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2344:.+]] = vector.extract %[[D2276]][3, 2] : vector<8x8x2x2xf16>
// CHECK:          %[[D2345:.+]] = nvgpu.mma.sync(%[[D2284]], %[[D2344]], %[[D2343]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2346:.+]] = vector.extract %[[D2276]][3, 3] : vector<8x8x2x2xf16>
// CHECK:          %[[D2347:.+]] = nvgpu.mma.sync(%[[D2287]], %[[D2346]], %[[D2345]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2348:.+]] = vector.extract %[[D2276]][3, 4] : vector<8x8x2x2xf16>
// CHECK:          %[[D2349:.+]] = nvgpu.mma.sync(%[[D2290]], %[[D2348]], %[[D2347]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2350:.+]] = vector.extract %[[D2276]][3, 5] : vector<8x8x2x2xf16>
// CHECK:          %[[D2351:.+]] = nvgpu.mma.sync(%[[D2293]], %[[D2350]], %[[D2349]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2352:.+]] = vector.extract %[[D2276]][3, 6] : vector<8x8x2x2xf16>
// CHECK:          %[[D2353:.+]] = nvgpu.mma.sync(%[[D2296]], %[[D2352]], %[[D2351]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2354:.+]] = vector.extract %[[D2276]][3, 7] : vector<8x8x2x2xf16>
// CHECK:          %[[D2355:.+]] = nvgpu.mma.sync(%[[D2299]], %[[D2354]], %[[D2353]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2356:.+]] = vector.insert %[[D2355]], %[[D2338]] [0, 3] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2357:.+]] = vector.extract %[[D1980]][0, 4] : vector<2x8x2x2xf32>
// CHECK:          %[[D2358:.+]] = vector.extract %[[D2276]][4, 0] : vector<8x8x2x2xf16>
// CHECK:          %[[D2359:.+]] = nvgpu.mma.sync(%[[D2278]], %[[D2358]], %[[D2357]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2360:.+]] = vector.extract %[[D2276]][4, 1] : vector<8x8x2x2xf16>
// CHECK:          %[[D2361:.+]] = nvgpu.mma.sync(%[[D2281]], %[[D2360]], %[[D2359]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2362:.+]] = vector.extract %[[D2276]][4, 2] : vector<8x8x2x2xf16>
// CHECK:          %[[D2363:.+]] = nvgpu.mma.sync(%[[D2284]], %[[D2362]], %[[D2361]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2364:.+]] = vector.extract %[[D2276]][4, 3] : vector<8x8x2x2xf16>
// CHECK:          %[[D2365:.+]] = nvgpu.mma.sync(%[[D2287]], %[[D2364]], %[[D2363]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2366:.+]] = vector.extract %[[D2276]][4, 4] : vector<8x8x2x2xf16>
// CHECK:          %[[D2367:.+]] = nvgpu.mma.sync(%[[D2290]], %[[D2366]], %[[D2365]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2368:.+]] = vector.extract %[[D2276]][4, 5] : vector<8x8x2x2xf16>
// CHECK:          %[[D2369:.+]] = nvgpu.mma.sync(%[[D2293]], %[[D2368]], %[[D2367]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2370:.+]] = vector.extract %[[D2276]][4, 6] : vector<8x8x2x2xf16>
// CHECK:          %[[D2371:.+]] = nvgpu.mma.sync(%[[D2296]], %[[D2370]], %[[D2369]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2372:.+]] = vector.extract %[[D2276]][4, 7] : vector<8x8x2x2xf16>
// CHECK:          %[[D2373:.+]] = nvgpu.mma.sync(%[[D2299]], %[[D2372]], %[[D2371]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2374:.+]] = vector.insert %[[D2373]], %[[D2356]] [0, 4] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2375:.+]] = vector.extract %[[D1980]][0, 5] : vector<2x8x2x2xf32>
// CHECK:          %[[D2376:.+]] = vector.extract %[[D2276]][5, 0] : vector<8x8x2x2xf16>
// CHECK:          %[[D2377:.+]] = nvgpu.mma.sync(%[[D2278]], %[[D2376]], %[[D2375]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2378:.+]] = vector.extract %[[D2276]][5, 1] : vector<8x8x2x2xf16>
// CHECK:          %[[D2379:.+]] = nvgpu.mma.sync(%[[D2281]], %[[D2378]], %[[D2377]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2380:.+]] = vector.extract %[[D2276]][5, 2] : vector<8x8x2x2xf16>
// CHECK:          %[[D2381:.+]] = nvgpu.mma.sync(%[[D2284]], %[[D2380]], %[[D2379]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2382:.+]] = vector.extract %[[D2276]][5, 3] : vector<8x8x2x2xf16>
// CHECK:          %[[D2383:.+]] = nvgpu.mma.sync(%[[D2287]], %[[D2382]], %[[D2381]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2384:.+]] = vector.extract %[[D2276]][5, 4] : vector<8x8x2x2xf16>
// CHECK:          %[[D2385:.+]] = nvgpu.mma.sync(%[[D2290]], %[[D2384]], %[[D2383]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2386:.+]] = vector.extract %[[D2276]][5, 5] : vector<8x8x2x2xf16>
// CHECK:          %[[D2387:.+]] = nvgpu.mma.sync(%[[D2293]], %[[D2386]], %[[D2385]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2388:.+]] = vector.extract %[[D2276]][5, 6] : vector<8x8x2x2xf16>
// CHECK:          %[[D2389:.+]] = nvgpu.mma.sync(%[[D2296]], %[[D2388]], %[[D2387]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2390:.+]] = vector.extract %[[D2276]][5, 7] : vector<8x8x2x2xf16>
// CHECK:          %[[D2391:.+]] = nvgpu.mma.sync(%[[D2299]], %[[D2390]], %[[D2389]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2392:.+]] = vector.insert %[[D2391]], %[[D2374]] [0, 5] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2393:.+]] = vector.extract %[[D1980]][0, 6] : vector<2x8x2x2xf32>
// CHECK:          %[[D2394:.+]] = vector.extract %[[D2276]][6, 0] : vector<8x8x2x2xf16>
// CHECK:          %[[D2395:.+]] = nvgpu.mma.sync(%[[D2278]], %[[D2394]], %[[D2393]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2396:.+]] = vector.extract %[[D2276]][6, 1] : vector<8x8x2x2xf16>
// CHECK:          %[[D2397:.+]] = nvgpu.mma.sync(%[[D2281]], %[[D2396]], %[[D2395]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2398:.+]] = vector.extract %[[D2276]][6, 2] : vector<8x8x2x2xf16>
// CHECK:          %[[D2399:.+]] = nvgpu.mma.sync(%[[D2284]], %[[D2398]], %[[D2397]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2400:.+]] = vector.extract %[[D2276]][6, 3] : vector<8x8x2x2xf16>
// CHECK:          %[[D2401:.+]] = nvgpu.mma.sync(%[[D2287]], %[[D2400]], %[[D2399]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2402:.+]] = vector.extract %[[D2276]][6, 4] : vector<8x8x2x2xf16>
// CHECK:          %[[D2403:.+]] = nvgpu.mma.sync(%[[D2290]], %[[D2402]], %[[D2401]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2404:.+]] = vector.extract %[[D2276]][6, 5] : vector<8x8x2x2xf16>
// CHECK:          %[[D2405:.+]] = nvgpu.mma.sync(%[[D2293]], %[[D2404]], %[[D2403]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2406:.+]] = vector.extract %[[D2276]][6, 6] : vector<8x8x2x2xf16>
// CHECK:          %[[D2407:.+]] = nvgpu.mma.sync(%[[D2296]], %[[D2406]], %[[D2405]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2408:.+]] = vector.extract %[[D2276]][6, 7] : vector<8x8x2x2xf16>
// CHECK:          %[[D2409:.+]] = nvgpu.mma.sync(%[[D2299]], %[[D2408]], %[[D2407]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2410:.+]] = vector.insert %[[D2409]], %[[D2392]] [0, 6] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2411:.+]] = vector.extract %[[D1980]][0, 7] : vector<2x8x2x2xf32>
// CHECK:          %[[D2412:.+]] = vector.extract %[[D2276]][7, 0] : vector<8x8x2x2xf16>
// CHECK:          %[[D2413:.+]] = nvgpu.mma.sync(%[[D2278]], %[[D2412]], %[[D2411]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2414:.+]] = vector.extract %[[D2276]][7, 1] : vector<8x8x2x2xf16>
// CHECK:          %[[D2415:.+]] = nvgpu.mma.sync(%[[D2281]], %[[D2414]], %[[D2413]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2416:.+]] = vector.extract %[[D2276]][7, 2] : vector<8x8x2x2xf16>
// CHECK:          %[[D2417:.+]] = nvgpu.mma.sync(%[[D2284]], %[[D2416]], %[[D2415]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2418:.+]] = vector.extract %[[D2276]][7, 3] : vector<8x8x2x2xf16>
// CHECK:          %[[D2419:.+]] = nvgpu.mma.sync(%[[D2287]], %[[D2418]], %[[D2417]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2420:.+]] = vector.extract %[[D2276]][7, 4] : vector<8x8x2x2xf16>
// CHECK:          %[[D2421:.+]] = nvgpu.mma.sync(%[[D2290]], %[[D2420]], %[[D2419]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2422:.+]] = vector.extract %[[D2276]][7, 5] : vector<8x8x2x2xf16>
// CHECK:          %[[D2423:.+]] = nvgpu.mma.sync(%[[D2293]], %[[D2422]], %[[D2421]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2424:.+]] = vector.extract %[[D2276]][7, 6] : vector<8x8x2x2xf16>
// CHECK:          %[[D2425:.+]] = nvgpu.mma.sync(%[[D2296]], %[[D2424]], %[[D2423]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2426:.+]] = vector.extract %[[D2276]][7, 7] : vector<8x8x2x2xf16>
// CHECK:          %[[D2427:.+]] = nvgpu.mma.sync(%[[D2299]], %[[D2426]], %[[D2425]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2428:.+]] = vector.insert %[[D2427]], %[[D2410]] [0, 7] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2429:.+]] = vector.extract %[[D1980]][1, 0] : vector<2x8x2x2xf32>
// CHECK:          %[[D2430:.+]] = vector.extract %[[D1977]][1, 0] : vector<2x8x4x2xf16>
// CHECK:          %[[D2431:.+]] = nvgpu.mma.sync(%[[D2430]], %[[D2279]], %[[D2429]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2432:.+]] = vector.extract %[[D1977]][1, 1] : vector<2x8x4x2xf16>
// CHECK:          %[[D2433:.+]] = nvgpu.mma.sync(%[[D2432]], %[[D2282]], %[[D2431]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2434:.+]] = vector.extract %[[D1977]][1, 2] : vector<2x8x4x2xf16>
// CHECK:          %[[D2435:.+]] = nvgpu.mma.sync(%[[D2434]], %[[D2285]], %[[D2433]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2436:.+]] = vector.extract %[[D1977]][1, 3] : vector<2x8x4x2xf16>
// CHECK:          %[[D2437:.+]] = nvgpu.mma.sync(%[[D2436]], %[[D2288]], %[[D2435]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2438:.+]] = vector.extract %[[D1977]][1, 4] : vector<2x8x4x2xf16>
// CHECK:          %[[D2439:.+]] = nvgpu.mma.sync(%[[D2438]], %[[D2291]], %[[D2437]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2440:.+]] = vector.extract %[[D1977]][1, 5] : vector<2x8x4x2xf16>
// CHECK:          %[[D2441:.+]] = nvgpu.mma.sync(%[[D2440]], %[[D2294]], %[[D2439]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2442:.+]] = vector.extract %[[D1977]][1, 6] : vector<2x8x4x2xf16>
// CHECK:          %[[D2443:.+]] = nvgpu.mma.sync(%[[D2442]], %[[D2297]], %[[D2441]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2444:.+]] = vector.extract %[[D1977]][1, 7] : vector<2x8x4x2xf16>
// CHECK:          %[[D2445:.+]] = nvgpu.mma.sync(%[[D2444]], %[[D2300]], %[[D2443]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2446:.+]] = vector.insert %[[D2445]], %[[D2428]] [1, 0] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2447:.+]] = vector.extract %[[D1980]][1, 1] : vector<2x8x2x2xf32>
// CHECK:          %[[D2448:.+]] = nvgpu.mma.sync(%[[D2430]], %[[D2304]], %[[D2447]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2449:.+]] = nvgpu.mma.sync(%[[D2432]], %[[D2306]], %[[D2448]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2450:.+]] = nvgpu.mma.sync(%[[D2434]], %[[D2308]], %[[D2449]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2451:.+]] = nvgpu.mma.sync(%[[D2436]], %[[D2310]], %[[D2450]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2452:.+]] = nvgpu.mma.sync(%[[D2438]], %[[D2312]], %[[D2451]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2453:.+]] = nvgpu.mma.sync(%[[D2440]], %[[D2314]], %[[D2452]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2454:.+]] = nvgpu.mma.sync(%[[D2442]], %[[D2316]], %[[D2453]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2455:.+]] = nvgpu.mma.sync(%[[D2444]], %[[D2318]], %[[D2454]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2456:.+]] = vector.insert %[[D2455]], %[[D2446]] [1, 1] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2457:.+]] = vector.extract %[[D1980]][1, 2] : vector<2x8x2x2xf32>
// CHECK:          %[[D2458:.+]] = nvgpu.mma.sync(%[[D2430]], %[[D2322]], %[[D2457]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2459:.+]] = nvgpu.mma.sync(%[[D2432]], %[[D2324]], %[[D2458]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2460:.+]] = nvgpu.mma.sync(%[[D2434]], %[[D2326]], %[[D2459]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2461:.+]] = nvgpu.mma.sync(%[[D2436]], %[[D2328]], %[[D2460]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2462:.+]] = nvgpu.mma.sync(%[[D2438]], %[[D2330]], %[[D2461]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2463:.+]] = nvgpu.mma.sync(%[[D2440]], %[[D2332]], %[[D2462]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2464:.+]] = nvgpu.mma.sync(%[[D2442]], %[[D2334]], %[[D2463]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2465:.+]] = nvgpu.mma.sync(%[[D2444]], %[[D2336]], %[[D2464]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2466:.+]] = vector.insert %[[D2465]], %[[D2456]] [1, 2] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2467:.+]] = vector.extract %[[D1980]][1, 3] : vector<2x8x2x2xf32>
// CHECK:          %[[D2468:.+]] = nvgpu.mma.sync(%[[D2430]], %[[D2340]], %[[D2467]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2469:.+]] = nvgpu.mma.sync(%[[D2432]], %[[D2342]], %[[D2468]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2470:.+]] = nvgpu.mma.sync(%[[D2434]], %[[D2344]], %[[D2469]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2471:.+]] = nvgpu.mma.sync(%[[D2436]], %[[D2346]], %[[D2470]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2472:.+]] = nvgpu.mma.sync(%[[D2438]], %[[D2348]], %[[D2471]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2473:.+]] = nvgpu.mma.sync(%[[D2440]], %[[D2350]], %[[D2472]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2474:.+]] = nvgpu.mma.sync(%[[D2442]], %[[D2352]], %[[D2473]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2475:.+]] = nvgpu.mma.sync(%[[D2444]], %[[D2354]], %[[D2474]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2476:.+]] = vector.insert %[[D2475]], %[[D2466]] [1, 3] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2477:.+]] = vector.extract %[[D1980]][1, 4] : vector<2x8x2x2xf32>
// CHECK:          %[[D2478:.+]] = nvgpu.mma.sync(%[[D2430]], %[[D2358]], %[[D2477]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2479:.+]] = nvgpu.mma.sync(%[[D2432]], %[[D2360]], %[[D2478]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2480:.+]] = nvgpu.mma.sync(%[[D2434]], %[[D2362]], %[[D2479]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2481:.+]] = nvgpu.mma.sync(%[[D2436]], %[[D2364]], %[[D2480]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2482:.+]] = nvgpu.mma.sync(%[[D2438]], %[[D2366]], %[[D2481]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2483:.+]] = nvgpu.mma.sync(%[[D2440]], %[[D2368]], %[[D2482]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2484:.+]] = nvgpu.mma.sync(%[[D2442]], %[[D2370]], %[[D2483]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2485:.+]] = nvgpu.mma.sync(%[[D2444]], %[[D2372]], %[[D2484]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2486:.+]] = vector.insert %[[D2485]], %[[D2476]] [1, 4] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2487:.+]] = vector.extract %[[D1980]][1, 5] : vector<2x8x2x2xf32>
// CHECK:          %[[D2488:.+]] = nvgpu.mma.sync(%[[D2430]], %[[D2376]], %[[D2487]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2489:.+]] = nvgpu.mma.sync(%[[D2432]], %[[D2378]], %[[D2488]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2490:.+]] = nvgpu.mma.sync(%[[D2434]], %[[D2380]], %[[D2489]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2491:.+]] = nvgpu.mma.sync(%[[D2436]], %[[D2382]], %[[D2490]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2492:.+]] = nvgpu.mma.sync(%[[D2438]], %[[D2384]], %[[D2491]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2493:.+]] = nvgpu.mma.sync(%[[D2440]], %[[D2386]], %[[D2492]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2494:.+]] = nvgpu.mma.sync(%[[D2442]], %[[D2388]], %[[D2493]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2495:.+]] = nvgpu.mma.sync(%[[D2444]], %[[D2390]], %[[D2494]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2496:.+]] = vector.insert %[[D2495]], %[[D2486]] [1, 5] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2497:.+]] = vector.extract %[[D1980]][1, 6] : vector<2x8x2x2xf32>
// CHECK:          %[[D2498:.+]] = nvgpu.mma.sync(%[[D2430]], %[[D2394]], %[[D2497]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2499:.+]] = nvgpu.mma.sync(%[[D2432]], %[[D2396]], %[[D2498]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2500:.+]] = nvgpu.mma.sync(%[[D2434]], %[[D2398]], %[[D2499]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2501:.+]] = nvgpu.mma.sync(%[[D2436]], %[[D2400]], %[[D2500]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2502:.+]] = nvgpu.mma.sync(%[[D2438]], %[[D2402]], %[[D2501]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2503:.+]] = nvgpu.mma.sync(%[[D2440]], %[[D2404]], %[[D2502]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2504:.+]] = nvgpu.mma.sync(%[[D2442]], %[[D2406]], %[[D2503]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2505:.+]] = nvgpu.mma.sync(%[[D2444]], %[[D2408]], %[[D2504]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2506:.+]] = vector.insert %[[D2505]], %[[D2496]] [1, 6] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          %[[D2507:.+]] = vector.extract %[[D1980]][1, 7] : vector<2x8x2x2xf32>
// CHECK:          %[[D2508:.+]] = nvgpu.mma.sync(%[[D2430]], %[[D2412]], %[[D2507]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2509:.+]] = nvgpu.mma.sync(%[[D2432]], %[[D2414]], %[[D2508]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2510:.+]] = nvgpu.mma.sync(%[[D2434]], %[[D2416]], %[[D2509]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2511:.+]] = nvgpu.mma.sync(%[[D2436]], %[[D2418]], %[[D2510]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2512:.+]] = nvgpu.mma.sync(%[[D2438]], %[[D2420]], %[[D2511]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2513:.+]] = nvgpu.mma.sync(%[[D2440]], %[[D2422]], %[[D2512]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2514:.+]] = nvgpu.mma.sync(%[[D2442]], %[[D2424]], %[[D2513]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2515:.+]] = nvgpu.mma.sync(%[[D2444]], %[[D2426]], %[[D2514]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
// CHECK:          %[[D2516:.+]] = vector.insert %[[D2515]], %[[D2506]] [1, 7] : vector<2x2xf32> into
// CHECK-SAME:       vector<2x8x2x2xf32>
// CHECK:          gpu.barrier
// CHECK:          memref.dealloc %[[ALLOC_10]] : memref<128x64xf16, #[[GPU]].address_space<workgroup>>
// CHECK:          memref.dealloc %[[ALLOC_11]] : memref<128x64xf16, #[[GPU]].address_space<workgroup>>
// CHECK:          scf.yield %[[CST_6]], %[[CST_6]], %[[CST_7]], %[[D1346]], %[[D1911]], %[[D2516]] : vector<32xf32>,
// CHECK-SAME:       vector<32xf32>, vector<32x64xf32>, vector<2x16x2x2xf32>, vector<2x16x2x2xf32>, vector<2x8x2x2xf32>
// CHECK:        }
// CHECK:        %[[D113:.+]] = arith.truncf %[[D112]]#[[D5:.+]] : vector<2x8x2x2xf32> to vector<2x8x2x2xf16>
// CHECK:        %[[D114:.+]] = vector.extract %[[D113]][0, 0, 0, 0] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D115:.+]] = affine.apply #[[MAP3]](%[[D5]], %[[D6]], %[[D7]])
// CHECK-DAG:    %[[D116:.+]] = affine.apply #[[MAP4]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D117:.+]] = arith.addi %[[D115]], %[[D8]] : index
// CHECK:        %[[D118:.+]] = arith.addi %[[D116]], %[[C0]] : index
// CHECK:        memref.store %[[D114]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D118]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D119:.+]] = vector.extract %[[D113]][0, 0, 0, 1] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D120:.+]] = affine.apply #[[MAP41]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D121:.+]] = arith.addi %[[D120]], %[[C0]] : index
// CHECK:        memref.store %[[D119]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D121]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D122:.+]] = vector.extract %[[D113]][0, 0, 1, 0] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D123:.+]] = affine.apply #[[MAP7]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D124:.+]] = arith.addi %[[D123]], %[[D8]] : index
// CHECK:        memref.store %[[D122]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D118]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D125:.+]] = vector.extract %[[D113]][0, 0, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D125]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D121]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D126:.+]] = vector.extract %[[D113]][0, 1, 0, 0] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D127:.+]] = affine.apply #[[MAP6]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D128:.+]] = arith.addi %[[D127]], %[[C0]] : index
// CHECK:        memref.store %[[D126]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D128]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D129:.+]] = vector.extract %[[D113]][0, 1, 0, 1] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D130:.+]] = affine.apply #[[MAP42]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D131:.+]] = arith.addi %[[D130]], %[[C0]] : index
// CHECK:        memref.store %[[D129]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D131]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D132:.+]] = vector.extract %[[D113]][0, 1, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D132]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D128]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D133:.+]] = vector.extract %[[D113]][0, 1, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D133]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D131]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D134:.+]] = vector.extract %[[D113]][0, 2, 0, 0] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D135:.+]] = affine.apply #[[MAP8]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D136:.+]] = arith.addi %[[D135]], %[[C0]] : index
// CHECK:        memref.store %[[D134]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D136]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D137:.+]] = vector.extract %[[D113]][0, 2, 0, 1] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D138:.+]] = affine.apply #[[MAP43]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D139:.+]] = arith.addi %[[D138]], %[[C0]] : index
// CHECK:        memref.store %[[D137]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D139]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D140:.+]] = vector.extract %[[D113]][0, 2, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D140]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D136]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D141:.+]] = vector.extract %[[D113]][0, 2, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D141]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D139]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D142:.+]] = vector.extract %[[D113]][0, 3, 0, 0] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D143:.+]] = affine.apply #[[MAP9]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D144:.+]] = arith.addi %[[D143]], %[[C0]] : index
// CHECK:        memref.store %[[D142]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D144]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D145:.+]] = vector.extract %[[D113]][0, 3, 0, 1] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D146:.+]] = affine.apply #[[MAP44]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D147:.+]] = arith.addi %[[D146]], %[[C0]] : index
// CHECK:        memref.store %[[D145]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D147]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D148:.+]] = vector.extract %[[D113]][0, 3, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D148]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D144]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D149:.+]] = vector.extract %[[D113]][0, 3, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D149]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D147]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D150:.+]] = vector.extract %[[D113]][0, 4, 0, 0] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D151:.+]] = affine.apply #[[MAP10]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D152:.+]] = arith.addi %[[D151]], %[[C0]] : index
// CHECK:        memref.store %[[D150]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D152]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D153:.+]] = vector.extract %[[D113]][0, 4, 0, 1] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D154:.+]] = affine.apply #[[MAP45]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D155:.+]] = arith.addi %[[D154]], %[[C0]] : index
// CHECK:        memref.store %[[D153]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D155]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D156:.+]] = vector.extract %[[D113]][0, 4, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D156]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D152]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D157:.+]] = vector.extract %[[D113]][0, 4, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D157]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D155]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D158:.+]] = vector.extract %[[D113]][0, 5, 0, 0] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D159:.+]] = affine.apply #[[MAP11]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D160:.+]] = arith.addi %[[D159]], %[[C0]] : index
// CHECK:        memref.store %[[D158]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D160]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D161:.+]] = vector.extract %[[D113]][0, 5, 0, 1] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D162:.+]] = affine.apply #[[MAP46]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D163:.+]] = arith.addi %[[D162]], %[[C0]] : index
// CHECK:        memref.store %[[D161]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D163]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D164:.+]] = vector.extract %[[D113]][0, 5, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D164]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D160]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D165:.+]] = vector.extract %[[D113]][0, 5, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D165]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D163]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D166:.+]] = vector.extract %[[D113]][0, 6, 0, 0] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D167:.+]] = affine.apply #[[MAP12]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D168:.+]] = arith.addi %[[D167]], %[[C0]] : index
// CHECK:        memref.store %[[D166]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D168]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D169:.+]] = vector.extract %[[D113]][0, 6, 0, 1] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D170:.+]] = affine.apply #[[MAP47]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D171:.+]] = arith.addi %[[D170]], %[[C0]] : index
// CHECK:        memref.store %[[D169]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D171]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D172:.+]] = vector.extract %[[D113]][0, 6, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D172]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D168]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D173:.+]] = vector.extract %[[D113]][0, 6, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D173]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D171]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D174:.+]] = vector.extract %[[D113]][0, 7, 0, 0] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D175:.+]] = affine.apply #[[MAP13]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D176:.+]] = arith.addi %[[D175]], %[[C0]] : index
// CHECK:        memref.store %[[D174]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D176]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D177:.+]] = vector.extract %[[D113]][0, 7, 0, 1] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D178:.+]] = affine.apply #[[MAP48]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D179:.+]] = arith.addi %[[D178]], %[[C0]] : index
// CHECK:        memref.store %[[D177]], %[[ALLOC_4]][%[[C0]], %[[D117]], %[[D179]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D180:.+]] = vector.extract %[[D113]][0, 7, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D180]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D176]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D181:.+]] = vector.extract %[[D113]][0, 7, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D181]], %[[ALLOC_4]][%[[C0]], %[[D124]], %[[D179]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D182:.+]] = vector.extract %[[D113]][1, 0, 0, 0] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D183:.+]] = affine.apply #[[MAP14]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D184:.+]] = arith.addi %[[D183]], %[[D8]] : index
// CHECK:        memref.store %[[D182]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D118]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D185:.+]] = vector.extract %[[D113]][1, 0, 0, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D185]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D121]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D186:.+]] = vector.extract %[[D113]][1, 0, 1, 0] : vector<2x8x2x2xf16>
// CHECK-DAG:    %[[D187:.+]] = affine.apply #[[MAP15]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D188:.+]] = arith.addi %[[D187]], %[[D8]] : index
// CHECK:        memref.store %[[D186]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D118]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D189:.+]] = vector.extract %[[D113]][1, 0, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D189]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D121]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D190:.+]] = vector.extract %[[D113]][1, 1, 0, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D190]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D128]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D191:.+]] = vector.extract %[[D113]][1, 1, 0, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D191]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D131]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D192:.+]] = vector.extract %[[D113]][1, 1, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D192]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D128]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D193:.+]] = vector.extract %[[D113]][1, 1, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D193]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D131]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D194:.+]] = vector.extract %[[D113]][1, 2, 0, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D194]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D136]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D195:.+]] = vector.extract %[[D113]][1, 2, 0, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D195]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D139]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D196:.+]] = vector.extract %[[D113]][1, 2, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D196]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D136]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D197:.+]] = vector.extract %[[D113]][1, 2, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D197]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D139]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D198:.+]] = vector.extract %[[D113]][1, 3, 0, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D198]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D144]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D199:.+]] = vector.extract %[[D113]][1, 3, 0, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D199]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D147]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D200:.+]] = vector.extract %[[D113]][1, 3, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D200]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D144]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D201:.+]] = vector.extract %[[D113]][1, 3, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D201]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D147]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D202:.+]] = vector.extract %[[D113]][1, 4, 0, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D202]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D152]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D203:.+]] = vector.extract %[[D113]][1, 4, 0, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D203]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D155]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D204:.+]] = vector.extract %[[D113]][1, 4, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D204]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D152]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D205:.+]] = vector.extract %[[D113]][1, 4, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D205]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D155]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D206:.+]] = vector.extract %[[D113]][1, 5, 0, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D206]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D160]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D207:.+]] = vector.extract %[[D113]][1, 5, 0, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D207]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D163]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D208:.+]] = vector.extract %[[D113]][1, 5, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D208]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D160]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D209:.+]] = vector.extract %[[D113]][1, 5, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D209]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D163]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D210:.+]] = vector.extract %[[D113]][1, 6, 0, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D210]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D168]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D211:.+]] = vector.extract %[[D113]][1, 6, 0, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D211]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D171]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D212:.+]] = vector.extract %[[D113]][1, 6, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D212]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D168]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D213:.+]] = vector.extract %[[D113]][1, 6, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D213]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D171]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D214:.+]] = vector.extract %[[D113]][1, 7, 0, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D214]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D176]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D215:.+]] = vector.extract %[[D113]][1, 7, 0, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D215]], %[[ALLOC_4]][%[[C0]], %[[D184]], %[[D179]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D216:.+]] = vector.extract %[[D113]][1, 7, 1, 0] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D216]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D176]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        %[[D217:.+]] = vector.extract %[[D113]][1, 7, 1, 1] : vector<2x8x2x2xf16>
// CHECK:        memref.store %[[D217]], %[[ALLOC_4]][%[[C0]], %[[D188]], %[[D179]]] : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        gpu.barrier
// CHECK:        memref.dealloc %[[ALLOC]] : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>
// CHECK:        gpu.barrier
// CHECK:        linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel",
// CHECK-SAME:     "parallel"]} ins(%[[ALLOC_4]] : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>)
// CHECK-SAME:     outs(%[[SUBVIEW_3]] : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>>) {
// CHECK:        ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:          linalg.yield %[[IN]] : f16
// CHECK:        }
// CHECK:        gpu.barrier
// CHECK:        memref.dealloc %[[ALLOC_4]] : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>
// CHECK:        return
// CHECK:      }
