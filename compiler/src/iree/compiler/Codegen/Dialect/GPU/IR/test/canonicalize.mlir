// RUN: iree-opt %s --canonicalize --split-input-file | FileCheck %s

func.func @fold_cast_into_resource_cast(%input: tensor<32xf16>) -> tensor<?xf16> {
  %cast = tensor.cast %input : tensor<32xf16> to tensor<?xf16>
  %resource_cast = iree_gpu.buffer_resource_cast %cast : tensor<?xf16>
  return %resource_cast : tensor<?xf16>
}

// CHECK-LABEL: func @fold_cast_into_resource_cast
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<32xf16>
//       CHECK:   %[[RESOURCE_CAST:.+]] = iree_gpu.buffer_resource_cast %[[INPUT]] : tensor<32xf16>
//       CHECK:   %[[CAST:.+]] = tensor.cast %[[RESOURCE_CAST]] : tensor<32xf16> to tensor<?xf16>
//       CHECK:   return %[[CAST]]

// -----

func.func @no_fold_static_cast_into_resource_cast(%input: tensor<?xf16>) -> tensor<32xf16> {
  %cast = tensor.cast %input : tensor<?xf16> to tensor<32xf16>
  %resource_cast = iree_gpu.buffer_resource_cast %cast : tensor<32xf16>
  return %resource_cast : tensor<32xf16>
}

// CHECK-LABEL: func @no_fold_static_cast_into_resource_cast
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<?xf16>
//       CHECK:   %[[CAST:.+]] = tensor.cast %[[INPUT]] : tensor<?xf16> to tensor<32xf16>
//       CHECK:   %[[RESOURCE_CAST:.+]] = iree_gpu.buffer_resource_cast %[[CAST]] : tensor<32xf16>
//       CHECK:   return %[[RESOURCE_CAST]]

// -----

func.func @split_cast_adding_and_removing_static_info(%input: tensor<?x32xf16>) -> tensor<32x?xf16> {
  %cast = tensor.cast %input : tensor<?x32xf16> to tensor<32x?xf16>
  %resource_cast = iree_gpu.buffer_resource_cast %cast : tensor<32x?xf16>
  return %resource_cast : tensor<32x?xf16>
}

// CHECK-LABEL: func @split_cast_adding_and_removing_static_info
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<?x32xf16>
//       CHECK:   %[[CAST_IN:.+]] = tensor.cast %[[INPUT]] : tensor<?x32xf16> to tensor<32x32xf16>
//       CHECK:   %[[RESOURCE_CAST:.+]] = iree_gpu.buffer_resource_cast %[[CAST_IN]] : tensor<32x32xf16>
//       CHECK:   %[[CAST_OUT:.+]] = tensor.cast %[[RESOURCE_CAST]] : tensor<32x32xf16> to tensor<32x?xf16>
//       CHECK:   return %[[CAST_OUT]]
