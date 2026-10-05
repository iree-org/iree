// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-collapse-contraction-dimensions))" %s | FileCheck %s --implicit-check-not=tensor.collapse_shape --implicit-check-not=tensor.expand_shape

#compilation = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0, 0]]>,
  translation_info = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Default>>>

// A preset compilation info refers to the original loop dimensions.
// CHECK-LABEL: @no_collapse_compilation_info(
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:      compilation_info =
// CHECK:         util.return {{.*}} : tensor<2x3x4xf32>
util.func public @no_collapse_compilation_info(%a: tensor<2x3x8xf32>, %b: tensor<8x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %result = linalg.generic {compilation_info = #compilation,
    indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, m1, k)>,
                     affine_map<(m0, m1, n, k) -> (k, n)>,
                     affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x3x8xf32>, tensor<8x4xf32>)
    outs(%init : tensor<2x3x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x3x4xf32>
  util.return %result : tensor<2x3x4xf32>
}

// -----

// A lowering config without a translation info also refers to the original
// loop dimensions.
// CHECK-LABEL: @no_collapse_lowering_config(
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:      lowering_config =
// CHECK:         util.return {{.*}} : tensor<2x3x4xf32>
util.func public @no_collapse_lowering_config(%a: tensor<2x3x8xf32>, %b: tensor<8x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %result = linalg.generic {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0, 0]]>,
    indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, m1, k)>,
                     affine_map<(m0, m1, n, k) -> (k, n)>,
                     affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x3x8xf32>, tensor<8x4xf32>)
    outs(%init : tensor<2x3x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x3x4xf32>
  util.return %result : tensor<2x3x4xf32>
}

// -----

// Dispatch formation already owns normalization inside this region.
// CHECK-LABEL: @no_collapse_in_dispatch_region(
// CHECK:         flow.dispatch.region
// CHECK:           linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK:           flow.return {{.*}} : tensor<2x3x4xf32>
util.func public @no_collapse_in_dispatch_region(%a: tensor<2x3x8xf32>, %b: tensor<8x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %dispatch = flow.dispatch.region -> (tensor<2x3x4xf32>) {
    %result = linalg.generic {
      indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, m1, k)>,
                       affine_map<(m0, m1, n, k) -> (k, n)>,
                       affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<2x3x8xf32>, tensor<8x4xf32>)
      outs(%init : tensor<2x3x4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %mul = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %mul, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<2x3x4xf32>
    flow.return %result : tensor<2x3x4xf32>
  }
  util.return %dispatch : tensor<2x3x4xf32>
}

// -----

// Explicitly scheduled workgroups are not revisited either.
// CHECK-LABEL: @no_collapse_in_dispatch_workgroups(
// CHECK:         flow.dispatch.workgroups
// CHECK:           linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK:           iree_tensor_ext.dispatch.tensor.store
util.func public @no_collapse_in_dispatch_workgroups(%a: tensor<2x3x8xf32>, %b: tensor<8x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %dispatch = flow.dispatch.workgroups(%a, %b, %init)
    : (tensor<2x3x8xf32>, tensor<8x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32> = (
        %a_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3x8xf32>>,
        %b_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>>,
        %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3x4xf32>>,
        %result_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x3x4xf32>>
      ) {
    %a_tile = iree_tensor_ext.dispatch.tensor.load %a_binding, offsets = [0, 0, 0], sizes = [2, 3, 8], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3x8xf32>> -> tensor<2x3x8xf32>
    %b_tile = iree_tensor_ext.dispatch.tensor.load %b_binding, offsets = [0, 0], sizes = [8, 4], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>> -> tensor<8x4xf32>
    %init_tile = iree_tensor_ext.dispatch.tensor.load %init_binding, offsets = [0, 0, 0], sizes = [2, 3, 4], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3x4xf32>> -> tensor<2x3x4xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, m1, k)>,
                       affine_map<(m0, m1, n, k) -> (k, n)>,
                       affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%a_tile, %b_tile : tensor<2x3x8xf32>, tensor<8x4xf32>)
      outs(%init_tile : tensor<2x3x4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %mul = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %mul, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<2x3x4xf32>
    iree_tensor_ext.dispatch.tensor.store %result, %result_binding, offsets = [0, 0, 0], sizes = [2, 3, 4], strides = [1, 1, 1]
      : tensor<2x3x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x3x4xf32>>
    flow.return
  }
  util.return %dispatch : tensor<2x3x4xf32>
}

// -----

// This pass normalizes contractions, not unrelated elementwise operations.
// CHECK-LABEL: @no_collapse_pointwise(
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
// CHECK:           math.absf
// CHECK:         util.return {{.*}} : tensor<2x3x4xf32>
util.func public @no_collapse_pointwise(%input: tensor<2x3x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, j, k)>, affine_map<(i, j, k) -> (i, j, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%input : tensor<2x3x4xf32>) outs(%init : tensor<2x3x4xf32>) {
  ^bb0(%element: f32, %output: f32):
    %abs = math.absf %element : f32
    linalg.yield %abs : f32
  } -> tensor<2x3x4xf32>
  util.return %result : tensor<2x3x4xf32>
}
