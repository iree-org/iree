// RUN: iree-opt %s --iree-transform-dialect-interpreter='transform-file-name=%p/reductions_codegen_spec.mlir' --split-input-file | FileCheck %s
// RUN: iree-opt %s --iree-transform-dialect-interpreter='transform-file-name=%p/reductions_match_spec.mlir' --split-input-file --verify-diagnostics

// Check that the same transform script applies to reductions with optional
// leading and trailing elementwise operations, potentially reordered
// producers and interleaving operations. This only checks for the matching
// and the right fusion structure without the surrounding IREE dispatch, which
// may fuse earlier, to decrease fragility.

!in_tensor_t = tensor<8x64xf32>
!out_tensor_t = tensor<8xf32>

func.func @reduce(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->   !out_tensor_t
  // expected-remark @below {{reduction}}
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      } -> !out_tensor_t
  return %2 : !out_tensor_t
}

// CHECK-LABEL: @reduce
// CHECK: scf.forall
// CHECK:   scf.forall
// CHECK:     linalg.fill
// CHECK:     linalg.generic
// CHECK:   scf.forall
// CHECK:     linalg.fill
// CHECK:     linalg.generic

// -----

!in_tensor_t = tensor<8x64xf32>
!out_tensor_t = tensor<8xf32>

func.func @eltwise_reduce(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->  !out_tensor_t
  %2 = tensor.empty() : !in_tensor_t
  // expected-remark @below {{leading}}
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg : !in_tensor_t) outs(%2 : !in_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg3 : f32
      %5 = arith.addf %4, %4 : f32
      linalg.yield %5 : f32
    } -> !in_tensor_t

  // expected-remark @below {{reduction}}
  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3 : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  return %6 : !out_tensor_t
}

// CHECK-LABEL: @eltwise_reduce
// CHECK: scf.forall
// CHECK:   scf.forall
// CHECK:     linalg.generic
// CHECK:     linalg.fill
// CHECK:     linalg.generic
// CHECK:   scf.forall
// CHECK:     linalg.fill
// CHECK:     linalg.generic

// -----

!in_tensor_t = tensor<8x64xf32>
!out_tensor_t = tensor<8xf32>

func.func @reduce_eltwise(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) -> !out_tensor_t
  // expected-remark @below {{reduction}}
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  %6 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{trailing}}
  %7 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%5 : !out_tensor_t) outs(%6 : !out_tensor_t) {  
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = math.sqrt %arg3 : f32
      linalg.yield %4 : f32
    } -> !out_tensor_t
  return %7 : !out_tensor_t
}


// CHECK-LABEL: @reduce_eltwise
// CHECK: scf.forall
// CHECK:   scf.forall
// CHECK:     linalg.fill
// CHECK:     linalg.generic
// CHECK:   scf.forall
// CHECK:     linalg.fill
// CHECK:     linalg.generic
// CHECK:     linalg.generic

// -----

!in_tensor_t = tensor<8x64xf32>
!out_tensor_t = tensor<8xf32>

func.func @eltwise_reduce_eltwise(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->  !out_tensor_t
  %2 = tensor.empty() : !in_tensor_t
  // expected-remark @below {{leading}}
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg : !in_tensor_t) outs(%2 : !in_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg3 : f32
      %5 = arith.addf %4, %4 : f32
      linalg.yield %5 : f32
    } -> !in_tensor_t

  // expected-remark @below {{reduction}}
  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3 : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  %7 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{trailing}}
  %8 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%6 : !out_tensor_t) outs(%7 : !out_tensor_t) {  
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = math.sqrt %arg3 : f32
      linalg.yield %4 : f32
    } -> !out_tensor_t


  return %8 : !out_tensor_t
}

// CHECK-LABEL: @eltwise_reduce_eltwise
// CHECK: scf.forall
// CHECK:   scf.forall
// CHECK:     linalg.generic
// CHECK:     linalg.fill
// CHECK:     linalg.generic
// CHECK:   scf.forall
// CHECK:     linalg.fill
// CHECK:     linalg.generic
// CHECK:     linalg.generic

// -----

!in_tensor_t = tensor<8x64xf32>
!out_tensor_t = tensor<8xf32>

func.func @eltwise_reduce_eltwise_swapped(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %2 = tensor.empty() : !in_tensor_t
  // expected-remark @below {{leading}}
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg : !in_tensor_t) outs(%2 : !in_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg3 : f32
      %5 = arith.addf %4, %4 : f32
      linalg.yield %5 : f32
    } -> !in_tensor_t

  %0 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->  !out_tensor_t
  // expected-remark @below {{reduction}}
  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3 : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  %7 = tensor.empty() : !out_tensor_t
  // expected-remark @below {{trailing}}
  %8 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%6 : !out_tensor_t) outs(%7 : !out_tensor_t) {  
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = math.sqrt %arg3 : f32
      linalg.yield %4 : f32
    } -> !out_tensor_t


  return %8 : !out_tensor_t
}

// CHECK-LABEL: @eltwise_reduce_eltwise_swapped
// CHECK: scf.forall
// CHECK:   scf.forall
// CHECK:     linalg.generic
// CHECK:     linalg.fill
// CHECK:     linalg.generic
// CHECK:   scf.forall
// CHECK:     linalg.fill
// CHECK:     linalg.generic
// CHECK:     linalg.generic
