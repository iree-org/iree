// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-optimize-tensor-insert-extract-slices{fold-identity-slices=true}))" --split-input-file %s | FileCheck %s

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
// CHECK-SAME:    %[[VEC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[SZ1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[SZ2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[INIT:.+]] = tensor.empty(%[[SZ1]], %[[SZ2]]) : tensor<1x?x?xf16>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[VEC]], %[[INIT]]
// CHECK-SAME:      [%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, false, false]}
// CHECK-SAME:      : vector<1x64x128xf16>, tensor<1x?x?xf16>
// CHECK:         return %[[WRITE]]

// -----

func.func @fold_insert_slice_into_transfer_write_static(%v: vector<4x5xf32>, %t1: tensor<4x5xf32>, %t2: tensor<?x?xf32>, %a: index, %b: index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %t1[%c0, %c0] {in_bounds = [true, true]} : vector<4x5xf32>, tensor<4x5xf32>
  %1 = tensor.insert_slice %0 into %t2[%a, %b] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @fold_insert_slice_into_transfer_write_static
// CHECK-SAME:    %[[VEC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %{{[a-zA-Z0-9]+}}
// CHECK-SAME:    %[[T2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]
// CHECK-NEXT:    %[[WRITE:.+]] = vector.transfer_write %[[VEC]], %[[T2]][%[[A]], %[[B]]] {in_bounds = [true, true]} : vector<4x5xf32>, tensor<?x?xf32>
// CHECK-NEXT:    return %[[WRITE]]

// -----

#aarch64_sve = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", target_triple = "aarch64-none-elf"}>

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
// CHECK-LABEL: func.func @fold_insert_slice_into_transfer_write_scalable
// CHECK-SAME:    %[[VEC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %{{[a-zA-Z0-9]+}}
// CHECK-SAME:    %[[T2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]
// CHECK-NEXT:    %[[WRITE:.+]] = vector.transfer_write %[[VEC]], %[[T2]][%[[A]], %[[B]]] {in_bounds = [true, true]} : vector<4x[4]xf32>, tensor<?x?xf32>
// CHECK-NEXT:    return %[[WRITE]]

// -----

func.func @fold_insert_slice_into_transfer_write_dynamic(%v: vector<4x8xf32>, %t1: tensor<?x?xf32>, %t2: tensor<?x?xf32>, %a: index, %b: index, %size: index) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %slice_size = affine.min affine_map<(d0) -> (d0, 8)>(%size)
  %extract_slice = tensor.extract_slice %t1[0, 0] [4, %slice_size] [1, 1] : tensor<?x?xf32> to tensor<4x?xf32>
  %0 = vector.transfer_write %v, %extract_slice[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xf32>, tensor<4x?xf32>
  %1 = tensor.insert_slice %0 into %t2[%a, %b] [4, %slice_size] [1, 1] : tensor<4x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @fold_insert_slice_into_transfer_write_dynamic
// CHECK-SAME:    %[[VEC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %{{[a-zA-Z0-9]+}}
// CHECK-SAME:    %[[T2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]
// CHECK-SAME:    %{{[a-zA-Z0-9]+}}
// CHECK-NEXT:    %[[WRITE:.+]] = vector.transfer_write %[[VEC]], %[[T2]][%[[A]], %[[B]]] {in_bounds = [true, true]} : vector<4x8xf32>, tensor<?x?xf32>
// CHECK-NEXT:    return %[[WRITE]]

// -----

func.func @negative_fold_insert_slice_into_transfer_write_static(%v: vector<3x5xf32>, %t1: tensor<4x5xf32>, %t2: tensor<?x?xf32>, %a: index, %b: index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %t1[%c0, %c0] {in_bounds = [true, true]} : vector<3x5xf32>, tensor<4x5xf32>
  %1 = tensor.insert_slice %0 into %t2[%a, %b] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @negative_fold_insert_slice_into_transfer_write_static
// CHECK: %[[WRITE:.*]] = vector.transfer_write
// CHECK: tensor.insert_slice %[[WRITE]]

// -----

#aarch64_sve = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", target_triple = "aarch64-none-elf"}>

func.func @negative_fold_insert_slice_into_transfer_write_scalable(%v: vector<4x[2]xf32>, %t1: tensor<?x?xf32>, %t2: tensor<?x?xf32>, %a: index, %b: index) -> tensor<?x?xf32>
  attributes {hal.executable.target = #aarch64_sve}
{
  %vscale = vector.vscale
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %extract_slice = tensor.extract_slice %t1[0, 0] [4, %c4_vscale] [1, 1] : tensor<?x?xf32> to tensor<4x?xf32>
  %0 = vector.transfer_write %v, %extract_slice[%c0, %c0] {in_bounds = [true, true]} : vector<4x[2]xf32>, tensor<4x?xf32>
  %1 = tensor.insert_slice %0 into %t2[%a, %b] [4, %c4_vscale] [1, 1] : tensor<4x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @negative_fold_insert_slice_into_transfer_write_scalable
// CHECK: %[[WRITE:.*]] = vector.transfer_write
// CHECK: tensor.insert_slice %[[WRITE]]

// -----

func.func @negative_fold_insert_slice_into_transfer_write_dynamic(%v: vector<4x7xf32>, %t1: tensor<?x?xf32>, %t2: tensor<?x?xf32>, %a: index, %b: index, %size: index) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %slice_size = affine.min affine_map<(d0) -> (d0, 8)>(%size)
  %extract_slice = tensor.extract_slice %t1[0, 0] [4, %slice_size] [1, 1] : tensor<?x?xf32> to tensor<4x?xf32>
  %0 = vector.transfer_write %v, %extract_slice[%c0, %c0] {in_bounds = [true, true]} : vector<4x7xf32>, tensor<4x?xf32>
  %1 = tensor.insert_slice %0 into %t2[%a, %b] [4, %slice_size] [1, 1] : tensor<4x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @negative_fold_insert_slice_into_transfer_write_dynamic
// CHECK: %[[WRITE:.*]] = vector.transfer_write
// CHECK: tensor.insert_slice %[[WRITE]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
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
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x968x1280xf16>>
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %1 = affine.apply #map()[%workgroup_id_y]
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %2 = affine.apply #map1()[%workgroup_id_x]
  %3 = affine.min #map2()[%workgroup_id_y]
  %4 = tensor.empty() : tensor<1x64x128xf16>
  %5 = vector.transfer_write %cst, %4[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x64x128xf16>, tensor<1x64x128xf16>
  %6 = scf.for %arg2 = %c0 to %c20 step %c1 iter_args(%arg3 = %5) -> (tensor<1x64x128xf16>) {
    %7 = affine.delinearize_index %arg2 into (20) : index
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
  iree_tensor_ext.dispatch.tensor.store %extracted_slice, %0, offsets = [%workgroup_id_z, %1, %2], sizes = [1, %3, 128], strides = [1, 1, 1] : tensor<1x?x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x968x1280xf16>>
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[WRITE]]

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

// -----

func.func @fold_identity_extract_slice(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %slice = tensor.extract_slice %arg0[0][%dim][1] : tensor<?xf32> to tensor<?xf32>
  return %slice : tensor<?xf32>
}

// CHECK-LABEL: @fold_identity_extract_slice
//       CHECK:   %[[ARG0:.+]]: tensor<?xf32>
//       CHECK:   return %[[ARG0]]

// -----

func.func @push_up_extract_slice(%arg0: index, %arg1: vector<64x64xf32>, %arg2: tensor<2x4096x10x64xf16>) -> tensor<1x64x1x64xf16> {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<64x64xf16>
  %c2 = arith.constant 2 : index
  %1 = arith.addi %arg0, %c2 : index
  %2 = arith.truncf %arg1 : vector<64x64xf32> to vector<64x64xf16>
  %3 = vector.transfer_write %2, %0[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xf16>, tensor<64x64xf16>
  %extracted_slice = tensor.extract_slice %arg2[%arg0, %c2, %1, %arg0] [1, 64, 1, 64] [1, 1, 1, 1] : tensor<2x4096x10x64xf16> to tensor<1x64x1x64xf16>
  %inserted_slice = tensor.insert_slice %3 into %extracted_slice[0, 0, 0, 0] [1, 64, 1, 64] [1, 1, 1, 1] : tensor<64x64xf16> into tensor<1x64x1x64xf16>
  return %inserted_slice : tensor<1x64x1x64xf16>
}

// CHECK-LABEL: @push_up_extract_slice
//       CHECK:   tensor.extract_slice
//       CHECK:   vector.transfer_write

// -----

#trait = {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
}

func.func @licm_generic(%source: tensor<32x32xf16>, %idx : index) -> tensor<32x32xf16> {
  %empty = tensor.empty() : tensor<32x32xf16>
  %c2 = arith.constant 2 : index
  %out = linalg.generic #trait outs(%empty : tensor<32x32xf16>) {
  ^bb0(%o: f16):
    %rem = arith.remsi %idx, %c2 : index
    %div = arith.divsi %idx, %c2 : index
    %extracted = tensor.extract %source[%rem, %div] : tensor<32x32xf16>
    linalg.yield %extracted : f16
  } -> tensor<32x32xf16>
  func.return %out : tensor<32x32xf16>
}

// CHECK-LABEL: @licm_generic
// CHECK-NOT: linalg.generic
// CHECK: arith.remsi
// CHECK: arith.divsi
// CHECK: tensor.extract
// CHECK: linalg.generic
// CHECK-NOT: tensor.extract
// CHECK: return

// -----

// Verify that loop invariant ops are not hoisted from regions that may not be
// executed.
func.func @no_hoist_from_possibly_unexecuted_region(%arg0: tensor<4x8xi32>) -> tensor<8x4xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %0 = tensor.empty() : tensor<8x4xi32>
  %1 = scf.for %arg1 = %workgroup_id_x to %c1 step %c100 iter_args(%arg2 = %0) -> tensor<8x4xi32> {
    %2 = vector.transfer_read %arg0[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : tensor<4x8xi32>, vector<2x8xi32>
    %3 = vector.transpose %2, [1, 0] : vector<2x8xi32> to vector<8x2xi32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x2xi32>, tensor<8x4xi32>
    scf.yield %4 : tensor<8x4xi32>
  }
  return %1 : tensor<8x4xi32>
}

// CHECK-LABEL: func.func @no_hoist_from_possibly_unexecuted_region
// CHECK:       scf.for {{.*}} {
// CHECK:         vector.transfer_read
// CHECK:         vector.transpose
// CHECK:         vector.transfer_write
// CHECK:       }

// -----

// Test for FoldMaskedTransferRAW.
// Both write and read are masked with the same mask: replace with select(mask, val, broadcast(pad)).
func.func @fold_masked_transfer_raw_both_masked(%t: tensor<128xf16>, %mask: vector<128xi1>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %val = arith.constant dense<1.0> : vector<128xf16>
  %w = vector.transfer_write %val, %t[%c0], %mask {in_bounds = [true]}
     : vector<128xf16>, tensor<128xf16>
  %r = vector.transfer_read %w[%c0], %cst, %mask {in_bounds = [true]}
     : tensor<128xf16>, vector<128xf16>
  return %r : vector<128xf16>
}

// CHECK-LABEL: func.func @fold_masked_transfer_raw_both_masked
// CHECK-SAME:    %[[T:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[MASK:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : vector<128xf16>
// CHECK-DAG:     %[[CST_1:.*]] = arith.constant dense<1.000000e+00> : vector<128xf16>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[CST_1]], %[[CST_0]]
// CHECK:         return %[[SEL]]

// -----

// Test for FoldMaskedTransferRAW.
// Masked write, unmasked read: replace with select(wMask, val, read(original_tensor)).
func.func @fold_masked_transfer_raw_masked_write_unmasked_read(%t: tensor<128xf16>, %mask: vector<128xi1>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %val = arith.constant dense<1.0> : vector<128xf16>
  %w = vector.transfer_write %val, %t[%c0], %mask {in_bounds = [true]}
     : vector<128xf16>, tensor<128xf16>
  %r = vector.transfer_read %w[%c0], %cst {in_bounds = [true]}
     : tensor<128xf16>, vector<128xf16>
  return %r : vector<128xf16>
}
// CHECK-LABEL: func.func @fold_masked_transfer_raw_masked_write_unmasked_read
// CHECK-SAME:    %[[T:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[MASK:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[CST:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:     %[[VAL:.*]] = arith.constant dense<1.000000e+00> : vector<128xf16>
// CHECK:         %[[READ:.*]] = vector.transfer_read %[[T]]{{.*}}, %[[CST]] {in_bounds = [true]}
// CHECK-SAME:      : tensor<128xf16>, vector<128xf16>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[VAL]], %[[READ]]
// CHECK:         return %[[SEL]]

// -----

// Test for FoldMaskedTransferRAW.
// Both unmasked: the read is directly replaced by the written value.
func.func @fold_masked_transfer_raw_both_unmasked(%t: tensor<128xf16>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %val = arith.constant dense<1.0> : vector<128xf16>
  %w = vector.transfer_write %val, %t[%c0] {in_bounds = [true]}
     : vector<128xf16>, tensor<128xf16>
  %r = vector.transfer_read %w[%c0], %cst {in_bounds = [true]}
     : tensor<128xf16>, vector<128xf16>
  return %r : vector<128xf16>
}
// CHECK-LABEL: func.func @fold_masked_transfer_raw_both_unmasked
// CHECK-DAG:     %[[VAL:.*]] = arith.constant dense<1.000000e+00> : vector<128xf16>
// CHECK-NOT:     vector.transfer_write
// CHECK-NOT:     vector.transfer_read
// CHECK:         return %[[VAL]]

// -----

// Test for FoldMaskedTransferRAW.
// Unmasked write, masked read: re-read the original tensor with the read's mask.
func.func @fold_masked_transfer_raw_unmasked_write_masked_read(%t: tensor<128xf16>, %mask: vector<128xi1>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %val = arith.constant dense<1.0> : vector<128xf16>
  %w = vector.transfer_write %val, %t[%c0] {in_bounds = [true]}
     : vector<128xf16>, tensor<128xf16>
  %r = vector.transfer_read %w[%c0], %cst, %mask {in_bounds = [true]}
     : tensor<128xf16>, vector<128xf16>
  return %r : vector<128xf16>
}

// CHECK-LABEL: func.func @fold_masked_transfer_raw_unmasked_write_masked_read
// CHECK-SAME:    %[[T:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[MASK:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[VAL:.*]] = arith.constant dense<1.000000e+00> : vector<128xf16>
// CHECK-DAG:     %[[PAD:.*]] = arith.constant dense<0.000000e+00> : vector<128xf16>
// CHECK-NOT:     vector.transfer_write
// CHECK-NOT:     vector.transfer_read
// CHECK:         %[[RES:.+]] = arith.select %[[MASK]]
// CHECK-DAG:         %[[VAL]]
// CHECK-DAG:         %[[PAD]]
// CHECK-SAME:       vector<128xi1>, vector<128xf16>
// CHECK:         return %[[RES]]

// -----

// transfer_read from a memref (not tensor semantics): pattern must not fire.
func.func @negative_read_empty_not_tensor_semantics(%m: memref<128xf16>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %r = vector.transfer_read %m[%c0], %cst {in_bounds = [true]}
     : memref<128xf16>, vector<128xf16>
  return %r : vector<128xf16>
}
// CHECK-LABEL: func.func @negative_read_empty_not_tensor_semantics
// CHECK:         vector.transfer_read

// -----

// transfer_read from a regular tensor (not tensor.empty): pattern must not fire.
func.func @negative_read_not_empty_tensor(%t: tensor<128xf16>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %r = vector.transfer_read %t[%c0], %cst {in_bounds = [true]}
     : tensor<128xf16>, vector<128xf16>
  return %r : vector<128xf16>
}
// CHECK-LABEL: func.func @negative_read_not_empty_tensor
// CHECK:         vector.transfer_read

// -----

// transfer_read from tensor.empty with a transposing permutation map: bail.
func.func @negative_read_empty_non_identity_map() -> vector<64x128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %e = tensor.empty() : tensor<128x64xf16>
  %r = vector.transfer_read %e[%c0, %c0], %cst
     {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>}
     : tensor<128x64xf16>, vector<64x128xf16>
  return %r : vector<64x128xf16>
}
// CHECK-LABEL: func.func @negative_read_empty_non_identity_map
// CHECK:         tensor.empty
// CHECK:         vector.transfer_read

// -----

// Unmasked, in-bounds read from tensor.empty -> ub.poison.
func.func @fold_read_empty_unmasked_inbounds() -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %e = tensor.empty() : tensor<128xf16>
  %r = vector.transfer_read %e[%c0], %cst {in_bounds = [true]}
     : tensor<128xf16>, vector<128xf16>
  return %r : vector<128xf16>
}
// CHECK-LABEL: func.func @fold_read_empty_unmasked_inbounds
// CHECK-NOT:     tensor.empty
// CHECK-NOT:     vector.transfer_read
// CHECK:         %[[POISON:.*]] = ub.poison : vector<128xf16>
// CHECK:         return %[[POISON]]

// -----

// Unmasked, out-of-bounds read from tensor.empty -> ub.poison.
func.func @fold_read_empty_unmasked_outofbounds() -> vector<256xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %e = tensor.empty() : tensor<128xf16>
  %r = vector.transfer_read %e[%c0], %cst
     : tensor<128xf16>, vector<256xf16>
  return %r : vector<256xf16>
}
// CHECK-LABEL: func.func @fold_read_empty_unmasked_outofbounds
// CHECK-NOT:     tensor.empty
// CHECK-NOT:     vector.transfer_read
// CHECK:         %[[PAD:.+]] = arith.constant dense<0.000000e+00> : vector<256xf16>
// CHECK:         return %[[PAD]]

// -----

// Masked read from tensor.empty where padding is ub.poison -> just ub.poison.
func.func @fold_read_empty_masked_poison_pad(%mask: vector<128xi1>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %pad = ub.poison : f16
  %e = tensor.empty() : tensor<128xf16>
  %r = vector.transfer_read %e[%c0], %pad, %mask {in_bounds = [true]}
     : tensor<128xf16>, vector<128xf16>
  return %r : vector<128xf16>
}
// CHECK-LABEL: func.func @fold_read_empty_masked_poison_pad
// CHECK-NOT:     tensor.empty
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     arith.select
// CHECK:         %[[POISON:.*]] = ub.poison : vector<128xf16>
// CHECK:         return %[[POISON]]

// -----

// Masked read from tensor.empty with a concrete pad value -> select(mask, poison, broadcast(pad)).
// Followed by: select cond, poison, X -> X
func.func @fold_read_empty_masked_real_pad(%mask: vector<128xi1>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %e = tensor.empty() : tensor<128xf16>
  %r = vector.transfer_read %e[%c0], %cst, %mask {in_bounds = [true]}
     : tensor<128xf16>, vector<128xf16>
  return %r : vector<128xf16>
}
// CHECK-LABEL: func.func @fold_read_empty_masked_real_pad
// CHECK:         %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<128xf16>
// CHECK:         return %[[CST]]

// -----

// Unmasked read from a dynamically-shaped tensor.empty -> ub.poison.
func.func @fold_read_empty_dynamic_unmasked(%sz: index) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %e = tensor.empty(%sz) : tensor<?xf16>
  %r = vector.transfer_read %e[%c0], %cst {in_bounds = [true]}
     : tensor<?xf16>, vector<128xf16>
  return %r : vector<128xf16>
}
// CHECK-LABEL: func.func @fold_read_empty_dynamic_unmasked
// CHECK-NOT:     tensor.empty
// CHECK-NOT:     vector.transfer_read
// CHECK:         %[[POISON:.*]] = ub.poison : vector<128xf16>
// CHECK:         return %[[POISON]]

// -----

// Multiple chained gathers: the index vectors computed for the first two
// gathers (and the clamp ops derived from their results) must be reused
// directly as vector SSA values by subsequent gathers. Vectorization may
// introduce tensor.empty<...xindex> intermediaries with write-read chains
// for materialized index vectors; these are cleaned up by the
// optimize-tensor-insert-extract-slices pass that follows.

#map = affine_map<(d0, d1, d2)[s0, s1, s2] -> (s0, s1, s2, 0)>
#map1 = affine_map<(d0, d1, d2)[s0, s1, s2] -> ()>
#map2 = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2)>
#map4 = affine_map<(d0, d1, d2)[s0, s1, s2] -> (s0, s1, s2)>
module {
  func.func @three_gathers_index_materialization(%arg0: tensor<1x8x?xf32>, %arg1: tensor<1x8xf32>, %arg2: tensor<1x8xf32>, %arg3: tensor<1x8x?xf32>, %arg4: tensor<50x32x25x2xi32>, %arg5: tensor<50x40x40xi8>, %arg6: index, %arg7: index, %arg8: index, %arg9: index) -> tensor<1x8x?xf32> {
    %cst = arith.constant dense<0> : vector<1x8x8xi8>
    %0 = ub.poison : i8
    %cst_0 = arith.constant dense<39> : vector<1x8x8xindex>
    %cst_1 = arith.constant dense<0> : vector<1x8x8xindex>
    %1 = ub.poison : i32
    %cst_2 = arith.constant dense<0.000000e+00> : vector<1x8x8xf32>
    %2 = ub.poison : index
    %3 = ub.poison : f32
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %4 = tensor.empty() : tensor<1x8x8xf32>
    %5 = tensor.empty() : tensor<1x8x8xindex>
    %6 = tensor.empty() : tensor<1x8x8xindex>
    %7 = tensor.empty() : tensor<1x8x8xindex>
    %dim = tensor.dim %arg0, %c2 : tensor<1x8x?xf32>
    %8 = vector.transfer_read %arg1[%c0, %c0], %3 {in_bounds = [true, true]} : tensor<1x8xf32>, vector<1x8xf32>
    %9 = vector.transfer_read %arg2[%c0, %c0], %3 {in_bounds = [true, true]} : tensor<1x8xf32>, vector<1x8xf32>
    %10 = vector.create_mask %c1, %c8, %dim : vector<1x8x8xi1>
    %11 = vector.transfer_read %arg0[%c0, %c0, %c0], %3, %10 {in_bounds = [true, true, true]} : tensor<1x8x?xf32>, vector<1x8x8xf32>
    %12 = arith.divf %8, %9 : vector<1x8xf32>
    %13 = vector.broadcast %12 : vector<1x8xf32> to vector<8x1x8xf32>
    %14 = vector.transpose %13, [1, 2, 0] : vector<8x1x8xf32> to vector<1x8x8xf32>
    %15 = arith.cmpf une, %11, %cst_2 : vector<1x8x8xf32>
    %16 = arith.select %15, %14, %cst_2 : vector<1x8x8xi1>, vector<1x8x8xf32>
    %17 = arith.addi %arg6, %arg7 : index
    %18 = vector.step : vector<8xindex>
    %19 = vector.broadcast %18 : vector<8xindex> to vector<1x8x8xindex>
    %20 = vector.transpose %19, [0, 2, 1] : vector<1x8x8xindex> to vector<1x8x8xindex>
    %21 = vector.broadcast %arg8 : index to vector<1x8x8xindex>
    %22 = arith.addi %21, %20 : vector<1x8x8xindex>
    %23 = vector.step : vector<8xindex>
    %24 = vector.broadcast %arg9 : index to vector<8xindex>
    %25 = arith.addi %24, %23 : vector<8xindex>
    %26 = vector.transfer_write %16, %4[%c0, %c0, %c0], %10 {in_bounds = [true, true, true]} : vector<1x8x8xf32>, tensor<1x8x8xf32>
    %27 = vector.broadcast %17 : index to vector<1x8x8xindex>
    %28 = vector.transfer_write %27, %5[%c0, %c0, %c0], %10 {in_bounds = [true, true, true]} : vector<1x8x8xindex>, tensor<1x8x8xindex>
    %29 = vector.transfer_write %22, %6[%c0, %c0, %c0], %10 {in_bounds = [true, true, true]} : vector<1x8x8xindex>, tensor<1x8x8xindex>
    %30 = vector.broadcast %25 : vector<8xindex> to vector<1x8x8xindex>
    %31 = vector.transfer_write %30, %7[%c0, %c0, %c0], %10 {in_bounds = [true, true, true]} : vector<1x8x8xindex>, tensor<1x8x8xindex>
    %32 = iree_vector_ext.transfer_gather %arg4[%c0, %c0, %c0, %c0] [%17, %22, %25 : index, vector<1x8x8xindex>, vector<8xindex>], %1 {indexing_maps = [#map, #map1, #map2, #map3]} : tensor<50x32x25x2xi32>, vector<1x8x8xi32>
    %33 = arith.index_cast %32 : vector<1x8x8xi32> to vector<1x8x8xindex>
    %34 = vector.transfer_read %28[%c0, %c0, %c0], %2 {in_bounds = [true, true, true]} : tensor<1x8x8xindex>, vector<1x8x8xindex>
    %35 = vector.transfer_read %29[%c0, %c0, %c0], %2 {in_bounds = [true, true, true]} : tensor<1x8x8xindex>, vector<1x8x8xindex>
    %36 = vector.transfer_read %31[%c0, %c0, %c0], %2 {in_bounds = [true, true, true]} : tensor<1x8x8xindex>, vector<1x8x8xindex>
    %37 = iree_vector_ext.transfer_gather %arg4[%c0, %c0, %c0, %c1] [%34, %35, %36 : vector<1x8x8xindex>, vector<1x8x8xindex>, vector<1x8x8xindex>], %1 {indexing_maps = [#map, #map2, #map2, #map2]} : tensor<50x32x25x2xi32>, vector<1x8x8xi32>
    %38 = arith.index_cast %37 : vector<1x8x8xi32> to vector<1x8x8xindex>
    %39 = arith.maxsi %33, %cst_1 : vector<1x8x8xindex>
    %40 = arith.minui %39, %cst_0 : vector<1x8x8xindex>
    %41 = arith.maxsi %38, %cst_1 : vector<1x8x8xindex>
    %42 = arith.minui %41, %cst_0 : vector<1x8x8xindex>
    %43 = vector.transfer_read %28[%c0, %c0, %c0], %2 {in_bounds = [true, true, true]} : tensor<1x8x8xindex>, vector<1x8x8xindex>
    %44 = iree_vector_ext.transfer_gather %arg5[%c0, %c0, %c0] [%43, %40, %42 : vector<1x8x8xindex>, vector<1x8x8xindex>, vector<1x8x8xindex>], %0 {indexing_maps = [#map4, #map2, #map2, #map2]} : tensor<50x40x40xi8>, vector<1x8x8xi8>
    %45 = vector.transfer_read %26[%c0, %c0, %c0], %3 {in_bounds = [true, true, true]} : tensor<1x8x8xf32>, vector<1x8x8xf32>
    %46 = arith.cmpi ugt, %44, %cst : vector<1x8x8xi8>
    %47 = arith.select %46, %45, %cst_2 : vector<1x8x8xi1>, vector<1x8x8xf32>
    %48 = vector.transfer_write %47, %arg3[%c0, %c0, %c0] {in_bounds = [true, true, false]} : vector<1x8x8xf32>, tensor<1x8x?xf32>
    return %48 : tensor<1x8x?xf32>
  }
}

// Verify three transfer_gather ops are produced. Index vectors from the first
// two gathers (and the clamp ops on their results) feed directly into the
// third gather as vector SSA values — no tensor.empty<...xindex> or
// write-read chains.
//
// CHECK-LABEL: func.func @three_gathers_index_materialization
// CHECK-SAME:    %[[IN0:[a-zA-Z0-9]+]]: tensor<1x8x?xf32>
// CHECK-SAME:    %[[IN1:[a-zA-Z0-9]+]]: tensor<1x8xf32>
// CHECK-SAME:    %[[IN2:[a-zA-Z0-9]+]]: tensor<1x8xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x8x?xf32>
// CHECK-SAME:    %[[TABLE:[a-zA-Z0-9]+]]: tensor<50x32x25x2xi32>
// CHECK-SAME:    %[[LUT:[a-zA-Z0-9]+]]: tensor<50x40x40xi8>
//
//     No index-typed tensors should appear anywhere in the output.
// CHECK-NOT:     tensor<{{.*}}xindex>
//
//     Gather #1 from %indir_table — index vecs consumed directly.
// CHECK:         %[[G1:.+]] = iree_vector_ext.transfer_gather %[[TABLE]]
// CHECK-SAME:      : tensor<50x32x25x2xi32>, vector<1x8x8xi32>
// CHECK-NOT:     tensor<{{.*}}xindex>
// CHECK:         %[[G1_IDX:.+]] = arith.index_cast %[[G1]] : vector<1x8x8xi32> to vector<1x8x8xindex>
// CHECK-NOT:     tensor<{{.*}}xindex>
//
//     Gather #2 from %indir_table — reuses the same index vecs as #1.
// CHECK:         %[[G2:.+]] = iree_vector_ext.transfer_gather %[[TABLE]]
// CHECK-SAME:      : tensor<50x32x25x2xi32>, vector<1x8x8xi32>
// CHECK-NOT:     tensor<{{.*}}xindex>
// CHECK:         %[[G2_IDX:.+]] = arith.index_cast %[[G2]] : vector<1x8x8xi32> to vector<1x8x8xindex>
// CHECK-NOT:     tensor<{{.*}}xindex>
//
//     Clamp results of gather #1 and #2 — pure vector ops, no tensor roundtrip.
//     G1_IDX -> maxsi -> minui = CLAMP_A, G2_IDX -> maxsi -> minui = CLAMP_B.
// CHECK:         %[[G1_MAX:.+]] = arith.maxsi %[[G1_IDX]], {{.*}} : vector<1x8x8xindex>
// CHECK-NOT:     tensor<{{.*}}xindex>
// CHECK:         %[[CLAMP_A:.+]] = arith.minui %[[G1_MAX]], {{.*}} : vector<1x8x8xindex>
// CHECK-NOT:     tensor<{{.*}}xindex>
// CHECK:         %[[G2_MAX:.+]] = arith.maxsi %[[G2_IDX]], {{.*}} : vector<1x8x8xindex>
// CHECK-NOT:     tensor<{{.*}}xindex>
// CHECK:         %[[CLAMP_B:.+]] = arith.minui %[[G2_MAX]], {{.*}} : vector<1x8x8xindex>
// CHECK-NOT:     tensor<{{.*}}xindex>
//
//     Gather #3 from %lut — takes clamped results directly as index vectors.
// CHECK:         %[[G3:.+]] = iree_vector_ext.transfer_gather %[[LUT]]
// CHECK-SAME:      [{{.*}}, %[[CLAMP_A]], %[[CLAMP_B]] : {{.*}}]
// CHECK-SAME:      : tensor<50x40x40xi8>, vector<1x8x8xi8>
// CHECK-NOT:     tensor<{{.*}}xindex>
//
//     Final select + write.
// CHECK:         %[[GATE:.+]] = arith.cmpi ugt, %[[G3]]
// CHECK-NOT:     tensor<{{.*}}xindex>
// CHECK:         %[[RES:.+]] = arith.select %[[GATE]]
// CHECK-NOT:     tensor<{{.*}}xindex>
// CHECK:         vector.transfer_write %[[RES]], %[[OUT]]
// CHECK-NOT:     tensor<{{.*}}xindex>
