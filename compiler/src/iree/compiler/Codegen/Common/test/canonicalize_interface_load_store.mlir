// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-cleanup-buffer-alloc-view))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
// CHECK-LABEL: func.func @fold_reshape_load()
func.func @fold_reshape_load() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: %[[ARG:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) : !flow.dispatch.tensor<readonly:tensor<3x3x96xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<3x3x1x96xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<writeonly:tensor<3x3x96xf32>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[ARG]], {{.*}} : !flow.dispatch.tensor<readonly:tensor<3x3x96xf32>> -> tensor<3x3x96xf32>
  %3 = flow.dispatch.tensor.load %1, offsets=[0, 0, 0, 0], sizes =[3, 3, 1, 96], strides=[1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x1x96xf32>> -> tensor<3x3x1x96xf32>
  %4 = tensor.collapse_shape %3 [[0, 1, 2, 3]] : tensor<3x3x1x96xf32> into tensor<864xf32>
  %5 = tensor.expand_shape %4 [[0, 1, 2]] output_shape [3, 3, 96] : tensor<864xf32> into tensor<3x3x96xf32>
  //  CHECK: %[[FILL:.+]] = linalg.fill ins(%{{.+}}) outs(%[[LOAD]] : tensor<3x3x96xf32>)
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<3x3x96xf32>) -> tensor<3x3x96xf32>
  //  CHECK: flow.dispatch.tensor.store %[[FILL]], {{.*}}
  flow.dispatch.tensor.store %6, %2, offsets = [%c0, %c0, %c0], sizes = [3, 3, 96], strides = [%c1, %c1, %c1] : tensor<3x3x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<3x3x96xf32>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
// CHECK-LABEL: func.func @fold_reshape_store()
func.func @fold_reshape_store() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<3x3x1x96xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<writeonly:tensor<3x3x96xf32>>
  // CHECK: %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) : !flow.dispatch.tensor<writeonly:tensor<3x3x1x96xf32>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %{{.*}}, {{.*}}
  %3 = flow.dispatch.tensor.load %1, offsets=[0, 0, 0, 0], sizes =[3, 3, 1, 96], strides=[1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x1x96xf32>> -> tensor<3x3x1x96xf32>
  //  CHECK: %[[FILL:.+]] = linalg.fill ins(%{{.+}}) outs(%[[LOAD]] : tensor<3x3x1x96xf32>)
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<3x3x1x96xf32>) -> tensor<3x3x1x96xf32>
  %5 = tensor.collapse_shape %4 [[0, 1, 2, 3]] : tensor<3x3x1x96xf32> into tensor<864xf32>
  %6 = tensor.expand_shape %5 [[0, 1, 2]] output_shape [3, 3, 96] : tensor<864xf32> into tensor<3x3x96xf32>
  //  CHECK: flow.dispatch.tensor.store %[[FILL]], %[[OUT]], {{.+}} : tensor<3x3x1x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<3x3x1x96xf32>>
  flow.dispatch.tensor.store %6, %2, offsets = [%c0, %c0, 0], sizes = [3, 3, 96], strides = [1, %c1, 1] : tensor<3x3x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<3x3x96xf32>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
// CHECK-LABEL: func.func @dont_fold_reshape_with_not_full_load()
func.func @dont_fold_reshape_with_not_full_load() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c96 = arith.constant 96 : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<6x3x1x96xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<writeonly:tensor<4x3x96xf32>>
  %3 = flow.dispatch.tensor.load %1, offsets = [3, 0, 0, 0], sizes = [3, 3, 1, 96], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<6x3x1x96xf32>> -> tensor<3x3x1x96xf32>
  // CHECK: tensor.collapse_shape
  // CHECK: tensor.expand_shape
  %4 = tensor.collapse_shape %3 [[0, 1, 2, 3]] : tensor<3x3x1x96xf32> into tensor<864xf32>
  %5 = tensor.expand_shape %4 [[0, 1, 2]] output_shape [3, 3, 96] : tensor<864xf32> into tensor<3x3x96xf32>
  flow.dispatch.tensor.store %5, %2, offsets = [%c0, %c0, %c0], sizes = [3, 3, 96], strides = [%c1, %c1, %c1] : tensor<3x3x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x3x96xf32>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fold_dynamic_reshape() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %dim1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %dim2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?x96xf32>>{%dim0, %dim1}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<writeonly:tensor<?x12x8xf32>>{%dim2}
  %3 = flow.dispatch.tensor.load %1, offsets=[0, 0, 0], sizes =[%dim0, %dim1, 96], strides=[1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x96xf32>>{%dim0, %dim1} -> tensor<?x?x96xf32>
  %4 = tensor.collapse_shape %3 [[0, 1], [2]] : tensor<?x?x96xf32> into tensor<?x96xf32>
  %dyn = tensor.dim %4, %c0 : tensor<?x96xf32>
  %5 = tensor.expand_shape %4 [[0], [1, 2]] output_shape [%dyn, 12, 8] : tensor<?x96xf32> into tensor<?x12x8xf32>
  flow.dispatch.tensor.store %5, %2, offsets = [0, 0, 0], sizes = [%dim2, 12, 8], strides = [1, 1, 1] : tensor<?x12x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x12x8xf32>>{%dim2}
  return
}
//       CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//       CHECK: func.func @fold_dynamic_reshape()
//   CHECK-DAG:   %[[CST0:.+]] = hal.interface.constant.load layout(#{{.+}}) ordinal(0)
//   CHECK-DAG:   %[[CST1:.+]] = hal.interface.constant.load layout(#{{.+}}) ordinal(1)
//   CHECK-DAG:   %[[CST2:.+]] = hal.interface.constant.load layout(#{{.+}}) ordinal(2)
//       CHECK:   %[[COLLAPSED:.+]] = affine.apply #[[MAP]]()[%[[CST0]], %[[CST1]]]
//       CHECK:   %[[IN_BINDING:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       binding(0) : !flow.dispatch.tensor<readonly:tensor<?x96xf32>>{%[[COLLAPSED]]}
//       CHECK:   %[[OUT_BINDING:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       binding(1) : !flow.dispatch.tensor<writeonly:tensor<?x96xf32>>{%[[CST2]]}
//       CHECK:   %[[IN:.+]] = flow.dispatch.tensor.load %[[IN_BINDING]]
//       CHECK:   flow.dispatch.tensor.store %[[IN]], %[[OUT_BINDING]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
// CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 288)>
// CHECK-LABEL: func.func @fold_reshape_slice_store
func.func @fold_reshape_slice_store(%x: index) {
  // CHECK-SAME: %[[X:[A-Za-z0-9]+]]: index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<3x3x1x96xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<writeonly:tensor<1728xf32>>
  // CHECK: %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) : !flow.dispatch.tensor<writeonly:tensor<6x3x1x96xf32>>
  // CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %{{.*}}, {{.*}}
  %3 = flow.dispatch.tensor.load %1, offsets=[0, 0, 0, 0], sizes =[3, 3, 1, 96], strides=[1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x1x96xf32>> -> tensor<3x3x1x96xf32>
  //  CHECK: %[[FILL:.+]] = linalg.fill ins(%{{.+}}) outs(%[[LOAD]] : tensor<3x3x1x96xf32>)
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<3x3x1x96xf32>) -> tensor<3x3x1x96xf32>
  %5 = tensor.collapse_shape %4 [[0, 1, 2, 3]] : tensor<3x3x1x96xf32> into tensor<864xf32>
  //  CHECK: %[[XDIV:.+]] = affine.apply #[[$MAP]]()[%[[X]]]
  //  CHECK: flow.dispatch.tensor.store %[[FILL]], %[[OUT]], offsets = [%[[XDIV]], 0, 0, 0], sizes = [3, 3, 1, 96]
  //  CHECK-SAME: tensor<3x3x1x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<6x3x1x96xf32>>
  flow.dispatch.tensor.store %5, %2, offsets = [%x], sizes = [864], strides = [1] : tensor<864xf32> -> !flow.dispatch.tensor<writeonly:tensor<1728xf32>>
  return
}
