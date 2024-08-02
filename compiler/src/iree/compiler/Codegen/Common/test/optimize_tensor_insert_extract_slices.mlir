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

func.func @fold_insert_slice_into_transfer_write_static(%v: vector<4x5xf32>, %t1: tensor<4x5xf32>, %t2: tensor<?x?xf32>, %a: index, %b: index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %t1[%c0, %c0] {in_bounds = [true, true]} : vector<4x5xf32>, tensor<4x5xf32>
  %1 = tensor.insert_slice %0 into %t2[%a, %b] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @fold_insert_slice_into_transfer_write_static
// CHECK-SAME:    %[[VEC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[T1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[T2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]
// CHECK-NEXT:    %[[WRITE:.+]] = vector.transfer_write %[[VEC]], %[[T2]][%[[A]], %[[B]]] {in_bounds = [true, true]} : vector<4x5xf32>, tensor<?x?xf32>
// CHECK-NEXT:    return %[[WRITE]]
// -----

#aarch64_sve = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", target_triple = "aarch64-none-elf"}>

// CHECK-LABEL: func.func @fold_insert_slice_into_transfer_write_scalable
// CHECK-SAME:    %[[VEC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[T1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[T2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]
// CHECK-NEXT:    %[[WRITE:.+]] = vector.transfer_write %[[VEC]], %[[T2]][%[[A]], %[[B]]] {in_bounds = [true, true]} : vector<4x[4]xf32>, tensor<?x?xf32>
// CHECK-NEXT:    return %[[WRITE]]
func.func @fold_insert_slice_into_transfer_write_scalable(%v: vector<4x[4]xf32>, %t1: tensor<?x?xf32>, %t2: tensor<?x?xf32>, %a: index, %b: index) -> tensor<?x?xf32>
  attributes {hal.executable.target = #aarch64_sve}
{
  %vscale = vector.vscale
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %extract_slice = tensor.extract_slice %t1[0, 0] [4, %c4_vscale] [1, 1] : tensor<?x?xf32> to tensor<4x?xf32>
  %0 = vector.transfer_write %v, %extract_slice[%c0, %c0] {in_bounds = [true, true]} : vector<4x[4]xf32>, tensor<4x?xf32>
  %1 = tensor.insert_slice %0 into %t2[%a, %b] [4, %c4_vscale] [1, 1] : tensor<4x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @fold_insert_slice_into_transfer_write_dynamic
// CHECK-SAME:    %[[VEC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[T1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[T2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]
// CHECK-NEXT:    %[[WRITE:.+]] = vector.transfer_write %[[VEC]], %[[T2]][%[[A]], %[[B]]] {in_bounds = [true, true]} : vector<4x8xf32>, tensor<?x?xf32>
// CHECK-NEXT:    return %[[WRITE]]
func.func @fold_insert_slice_into_transfer_write_dynamic(%v: vector<4x8xf32>, %t1: tensor<?x?xf32>, %t2: tensor<?x?xf32>, %a: index, %b: index, %size: index) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %slice_size = affine.min affine_map<(d0) -> (d0, 8)>(%size)
  %extract_slice = tensor.extract_slice %t1[0, 0] [4, %slice_size] [1, 1] : tensor<?x?xf32> to tensor<4x?xf32>
  %0 = vector.transfer_write %v, %extract_slice[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xf32>, tensor<4x?xf32>
  %1 = tensor.insert_slice %0 into %t2[%a, %b] [4, %slice_size] [1, 1] : tensor<4x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

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
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x968x1280xf16>>
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
func.func @_batch_matmul_narrow_n_2_dispatch_4_unpack_i32() attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
  %c0_i32 = arith.constant 0 : i32
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c128) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1x1x2x8xi32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x3x2xi32>>
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

// -----

func.func @subset_hoisting_invariant_tensor(%init: tensor<64x64xf32>, %t: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty() : tensor<8x1xf32>
  %loop = scf.for %i = %c0 to %c8 step %c1 iter_args(%arg1 = %init) -> tensor<64x64xf32> {
    %slice1 = tensor.extract_slice %arg1[1, 0][8, 1][1, 1] : tensor<64x64xf32> to tensor<8x1xf32>
    %slice2 = tensor.extract_slice %init[0, %i][8, 1][1, 1] : tensor<64x64xf32> to tensor<8x1xf32>
    %out = linalg.add ins(%slice1, %slice2 : tensor<8x1xf32>, tensor<8x1xf32>) outs(%empty: tensor<8x1xf32>) -> tensor<8x1xf32>
    %td = tensor.insert_slice %out into %t[1, 0][8, 1][1, 1] : tensor<8x1xf32> into tensor<64x64xf32>
    scf.yield %td : tensor<64x64xf32>
  }
  return %loop : tensor<64x64xf32>
}

// CHECK-LABEL: @subset_hoisting_invariant_tensor
// CHECK:   tensor.extract_slice
// CHECK:   scf.for {{.*}} iter_args(%[[IV:.+]] = {{.*}})
// CHECK:     %[[SLICE:.+]] = tensor.extract_slice
// CHECK-NOT: tensor.extract_slice
// CHECK:     linalg.add ins(%[[IV]], %[[SLICE]] : {{.*}})
// CHECK:   scf.yield
// CHECK:   tensor.insert_slice

// -----

func.func @subset_hoisting_invariant_tensor_nonequivalent_subset(%init: tensor<64x64xf32>, %t: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty() : tensor<8x1xf32>
  %loop = scf.for %i = %c0 to %c8 step %c1 iter_args(%arg1 = %init) -> tensor<64x64xf32> {
    %slice1 = tensor.extract_slice %arg1[1, 0][8, 1][1, 1] : tensor<64x64xf32> to tensor<8x1xf32>
    %slice2 = tensor.extract_slice %init[0, %i][8, 1][1, 1] : tensor<64x64xf32> to tensor<8x1xf32>
    %out = linalg.add ins(%slice1, %slice2 : tensor<8x1xf32>, tensor<8x1xf32>) outs(%empty: tensor<8x1xf32>) -> tensor<8x1xf32>
    %td = tensor.insert_slice %out into %t[0, 0][8, 1][1, 1] : tensor<8x1xf32> into tensor<64x64xf32>
    scf.yield %td : tensor<64x64xf32>
  }
  return %loop : tensor<64x64xf32>
}

// CHECK-LABEL: @subset_hoisting_invariant_tensor_nonequivalent_subset
// CHECK:     scf.for
// CHECK:       tensor.extract_slice
// CHECK:       tensor.extract_slice
// CHECK:       tensor.insert_slice
// CHECK:       scf.yield
// CHECK-NOT: tensor.insert_slice
