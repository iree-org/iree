// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-optimize-tensor-insert-extract-slices))" --split-input-file %s | FileCheck %s

func.func @fold_extract_slice_consumer_into_xfer_write(%arg0: vector<1x64x128xf16>, %arg1: index) -> tensor<1x?x128xf16> {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<1x64x128xf16>
  %1 = vector.transfer_write %arg0, %0[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x64x128xf16>, tensor<1x64x128xf16>
  %extracted_slice = tensor.extract_slice %1[0, 0, 0] [1, %arg1, 128] [1, 1, 1] : tensor<1x64x128xf16> to tensor<1x?x128xf16>
  return %extracted_slice : tensor<1x?x128xf16>
}
// CHECK-LABEL: func.func @fold_extract_slice_consumer_into_xfer_write
// CHECK-SAME:    %[[VEC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[SZ:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[INIT:.+]] = tensor.empty(%[[SZ]]) : tensor<1x?x128xf16>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[VEC]], %[[INIT]]
// CHECK-SAME:      [%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, false, true]}
// CHECK-SAME:      : vector<1x64x128xf16>, tensor<1x?x128xf16>
// CHECK:         return %[[WRITE]]

// -----

// Test the case where we write out of bounds because large index 
func.func @fold_extract_slice_consumer_into_xfer_write_2(%arg0: vector<1x64x128xf16>, %arg1: index) -> tensor<1x?x128xf16> {
  %c0 = arith.constant 0 : index
  %c127 = arith.constant 127 : index
  %0 = tensor.empty() : tensor<1x64x128xf16>
  %1 = vector.transfer_write %arg0, %0[%c0, %c0, %c127] {in_bounds = [true, true, true]} : vector<1x64x128xf16>, tensor<1x64x128xf16>
  %extracted_slice = tensor.extract_slice %1[0, 0, 0] [1, %arg1, 128] [1, 1, 1] : tensor<1x64x128xf16> to tensor<1x?x128xf16>
  return %extracted_slice : tensor<1x?x128xf16>
}

// CHECK-LABEL: func.func @fold_extract_slice_consumer_into_xfer_write_2
// CHECK-SAME:    %[[VEC2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[SZ2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C127:.+]] = arith.constant 127 : index
// CHECK:         %[[INIT2:.+]] = tensor.empty(%[[SZ2]]) : tensor<1x?x128xf16>
// CHECK:         %[[WRITE2:.+]] = vector.transfer_write %[[VEC2]], %[[INIT2]]
// CHECK-SAME:      [%[[C0]], %[[C0]], %[[C127]]] {in_bounds = [true, false, false]}
// CHECK-SAME:      : vector<1x64x128xf16>, tensor<1x?x128xf16>
// CHECK:         return %[[WRITE2]]

// -----

// Test the case where we conservatively set in_bounds attribute
func.func @fold_extract_slice_consumer_into_xfer_write_3(%arg0: vector<1x64x128xf16>, %arg1: index, %arg2: index) -> tensor<1x?x?xf16> {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<1x64x128xf16>
  %1 = vector.transfer_write %arg0, %0[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x64x128xf16>, tensor<1x64x128xf16>
  %extracted_slice = tensor.extract_slice %1[0, 0, 0] [1, %arg1, %arg2] [1, 1, 1] : tensor<1x64x128xf16> to tensor<1x?x?xf16>
  return %extracted_slice : tensor<1x?x?xf16>
}

// CHECK-LABEL: func.func @fold_extract_slice_consumer_into_xfer_write_3
// CHECK-SAME:    %[[VEC2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[SZ2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[INIT2:.+]] = tensor.empty(%arg1, %arg2) : tensor<1x?x?xf16>
// CHECK:         %[[WRITE2:.+]] = vector.transfer_write %[[VEC2]], %[[INIT2]]
// CHECK-SAME:      [%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, false, false]}
// CHECK-SAME:      : vector<1x64x128xf16>, tensor<1x?x?xf16>
// CHECK:         return %[[WRITE2]]

// -----

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0] -> (s0 * -64 + 968, 64)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @batch_matmul_with_padding_strategy(%arg0: tensor<1x?x1280xf16>, %arg1: tensor<1x1280x128xf16>) {
  %cst = arith.constant dense<0.000000e+00> : vector<1x64x128xf16>
  %c20 = arith.constant 20 : index
  %c1 = arith.constant 1 : index
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x968x1280xf16>>
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %1 = affine.apply #map()[%workgroup_id_y]
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %2 = affine.apply #map1()[%workgroup_id_x]
  %3 = affine.min #map2()[%workgroup_id_y]
  %4 = tensor.empty() : tensor<1x64x128xf16>
  %5 = vector.transfer_write %cst, %4[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x64x128xf16>, tensor<1x64x128xf16>
  %6 = scf.for %arg2 = %c0 to %c20 step %c1 iter_args(%arg3 = %5) -> (tensor<1x64x128xf16>) {
    %7 = affine.delinearize_index %arg2 into (%c20) : index
    %8 = affine.apply #map()[%7]
    %extracted_slice_1 = tensor.extract_slice %arg1[0, %8, 0] [1, 64, 128] [1, 1, 1] : tensor<1x1280x128xf16> to tensor<1x64x128xf16>
    %extracted_slice_2 = tensor.extract_slice %arg0[0, 0, %8] [1, %3, 64] [1, 1, 1] : tensor<1x?x1280xf16> to tensor<1x?x64xf16>
    %9 = vector.transfer_read %extracted_slice_2[%c0, %c0, %c0], %cst_0 {in_bounds = [true, false, true]} : tensor<1x?x64xf16>, vector<1x64x64xf16>
    %10 = vector.transfer_read %extracted_slice_1[%c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true]} : tensor<1x64x128xf16>, vector<1x64x128xf16>
    %11 = vector.transfer_read %arg3[%c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true]} : tensor<1x64x128xf16>, vector<1x64x128xf16>
    %12 = vector.contract {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %9, %10, %11 : vector<1x64x64xf16>, vector<1x64x128xf16> into vector<1x64x128xf16>
    %13 = vector.transfer_write %12, %arg3[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x64x128xf16>, tensor<1x64x128xf16>
    scf.yield %13 : tensor<1x64x128xf16>
  }
  %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, %3, 128] [1, 1, 1] : tensor<1x64x128xf16> to tensor<1x?x128xf16>
  flow.dispatch.tensor.store %extracted_slice, %0, offsets = [%workgroup_id_z, %1, %2], sizes = [1, %3, 128], strides = [1, 1, 1] : tensor<1x?x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x968x1280xf16>>
  return
}
// CHECK-LABEL: func.func @batch_matmul_with_padding_strategy
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[SCF:.+]] = scf.for {{.+}} -> (vector<1x64x128xf16>) {
// CHECK:         }
// CHECK:         %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<1x?x128xf16>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[SCF]], %[[INIT]]
// CHECK-SAME:      [%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, false, true]}
// CHECK-SAME:      : vector<1x64x128xf16>, tensor<1x?x128xf16>
// CHECK:         flow.dispatch.tensor.store %[[WRITE]]

// -----

func.func @_batch_matmul_narrow_n_2_dispatch_4_unpack_i32() attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
  %c0_i32 = arith.constant 0 : i32
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c128) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1x1x2x8xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x3x2xi32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  scf.for %arg0 = %workgroup_id_x to %c2 step %workgroup_count_x {
    %2 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0, 0], sizes = [1, 3, 2], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<2x3x2xi32>> -> tensor<1x3x2xi32>
    %3 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0, 0, 0, 0], sizes = [1, 1, 1, 2, 8], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1x1x2x8xi32>> -> tensor<1x1x1x2x8xi32>
    %4 = vector.transfer_read %3[%c0, %c0, %c0, %c0, %c0], %c0_i32 {in_bounds = [true, true]} : tensor<1x1x1x2x8xi32>, vector<2x8xi32>
    %5 = vector.transpose %4, [1, 0] : vector<2x8xi32> to vector<8x2xi32>
    %6 = tensor.empty() : tensor<3x2xi32>
    %7 = vector.transfer_write %5, %6[%c0, %c0] {in_bounds = [false, true]} : vector<8x2xi32>, tensor<3x2xi32>
    %inserted_slice = tensor.insert_slice %7 into %2[0, 0, 0] [1, 3, 2] [1, 1, 1] : tensor<3x2xi32> into tensor<1x3x2xi32>
    flow.dispatch.tensor.store %inserted_slice, %1, offsets = [%arg0, 0, 0], sizes = [1, 3, 2], strides = [1, 1, 1] : tensor<1x3x2xi32> -> !flow.dispatch.tensor<writeonly:tensor<2x3x2xi32>>
  }
  return
}

// CHECK-LABEL: func.func @_batch_matmul_narrow_n_2_dispatch_4_unpack_i32
// CHECK: %[[EMPTY:[a-zA-Z0-9]+]] = tensor.empty() : tensor<3x2xi32>
// CHECK: %[[TRANS:[a-zA-Z0-9]+]] = vector.transpose %5, [1, 0] : vector<2x8xi32> to vector<8x2xi32>
// CHECK: vector.transfer_write %[[TRANS]], %[[EMPTY]][%c0, %c0] {in_bounds = [false, true]} : vector<8x2xi32>, tensor<3x2xi32>
