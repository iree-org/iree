// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=false}))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=true use-configured-vector-sizes=false}))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=true use-configured-vector-sizes=true}))" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @strided(
// CHECK: vector.transfer_read {{.*}} : tensor<1x4x1xf32>, vector<1x4x1xf32>
// CHECK: vector.extract_strided_slice {{.*}} offsets = [0, 1, 0]
// CHECK: vector.contract
// CHECK: vector.transfer_write
// CHECK: tensor.collapse_shape {{.*}} : tensor<1x2x1xf32> into tensor<2xf32>
// CHECK: return
func.func @strided(%input: tensor<4xf32>, %filter: tensor<2xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(w, kw) -> (w * 2 + kw)>,
                     affine_map<(w, kw) -> (kw)>,
                     affine_map<(w, kw) -> (w)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input, %filter : tensor<4xf32>, tensor<2xf32>) outs(%init : tensor<2xf32>)
    attrs = {lowering_config = #iree_gpu.lowering_config<{thread = [2, 0], reduction = [0, 2]}>} {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %p = arith.mulf %a, %b : f32
    %s = arith.addf %c, %p : f32
    linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @dilated(
// CHECK: vector.transfer_read {{.*}} : tensor<1x4x1xf32>, vector<1x4x1xf32>
// CHECK: vector.extract_strided_slice {{.*}} offsets = [0, 2, 0]
// CHECK: vector.contract
// CHECK: vector.transfer_write
// CHECK: tensor.collapse_shape {{.*}} : tensor<1x2x1xf32> into tensor<2xf32>
// CHECK: return
func.func @dilated(%input: tensor<4xf32>, %filter: tensor<2xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(w, kw) -> (w + kw * 2)>,
                     affine_map<(w, kw) -> (kw)>,
                     affine_map<(w, kw) -> (w)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input, %filter : tensor<4xf32>, tensor<2xf32>) outs(%init : tensor<2xf32>)
    attrs = {lowering_config = #iree_gpu.lowering_config<{thread = [2, 0], reduction = [0, 2]}>} {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %p = arith.mulf %a, %b : f32
    %s = arith.addf %c, %p : f32
    linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @strided_dilated(
// CHECK: vector.transfer_read {{.*}} : tensor<1x6x1xf32>, vector<1x6x1xf32>
// CHECK: vector.extract_strided_slice {{.*}} offsets = [0, 3, 0]
// CHECK: vector.contract
// CHECK: vector.transfer_write
// CHECK: tensor.collapse_shape {{.*}} : tensor<1x2x1xf32> into tensor<2xf32>
// CHECK: return
func.func @strided_dilated(%input: tensor<6xf32>, %filter: tensor<2xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(w, kw) -> (w * 2 + kw * 3)>,
                     affine_map<(w, kw) -> (kw)>,
                     affine_map<(w, kw) -> (w)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input, %filter : tensor<6xf32>, tensor<2xf32>) outs(%init : tensor<2xf32>)
    attrs = {lowering_config = #iree_gpu.lowering_config<{thread = [2, 0], reduction = [0, 2]}>} {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %p = arith.mulf %a, %b : f32
    %s = arith.addf %c, %p : f32
    linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @strided_integer(
// CHECK: vector.transfer_read {{.*}} : tensor<1x6x1xi32>, vector<1x6x1xi32>
// CHECK: vector.extract_strided_slice {{.*}} offsets = [0, 2, 0]
// CHECK: vector.contract
// CHECK: vector.transfer_write
// CHECK: tensor.collapse_shape {{.*}} : tensor<1x2x1xi32> into tensor<2xi32>
// CHECK: return
func.func @strided_integer(%input: tensor<6xi32>, %filter: tensor<2xi32>, %init: tensor<2xi32>) -> tensor<2xi32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(w, kw) -> (w * 3 + kw * 2)>,
                     affine_map<(w, kw) -> (kw)>,
                     affine_map<(w, kw) -> (w)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input, %filter : tensor<6xi32>, tensor<2xi32>) outs(%init : tensor<2xi32>)
    attrs = {lowering_config = #iree_gpu.lowering_config<{thread = [2, 0], reduction = [0, 2]}>} {
  ^bb0(%a: i32, %b: i32, %c: i32):
    %p = arith.muli %a, %b : i32
    %s = arith.addi %c, %p : i32
    linalg.yield %s : i32
  } -> tensor<2xi32>
  return %r : tensor<2xi32>
}

// -----

// Matching indexing maps alone must not turn a different body into a convolution.
// CHECK-LABEL: func.func @not_convolution(
// CHECK: linalg.generic
// CHECK: arith.subf
// CHECK: return
func.func @not_convolution(%input: tensor<4xf32>, %filter: tensor<2xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(w, kw) -> (w * 2 + kw)>,
                     affine_map<(w, kw) -> (kw)>,
                     affine_map<(w, kw) -> (w)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input, %filter : tensor<4xf32>, tensor<2xf32>) outs(%init : tensor<2xf32>)
    attrs = {lowering_config = #iree_gpu.lowering_config<{thread = [2, 0], reduction = [0, 2]}>} {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %p = arith.mulf %a, %b : f32
    %s = arith.subf %c, %p : f32
    linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}
