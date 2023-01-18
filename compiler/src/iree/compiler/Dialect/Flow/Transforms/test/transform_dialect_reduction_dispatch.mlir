// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-flow-dispatch-with-transform-dialect),canonicalize,cse)" %s | \
// RUN: FileCheck %s

// CHECK-LABEL: @reduction
func.func @reduction(%arg: tensor<8x479xf32>) -> tensor<8xf32> {
  // CHECK: flow.dispatch.workgroups
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   linalg.fill
  // CHECK:   linalg.generic
  // CHECK:   flow.dispatch.tensor.store
  // CHECK:   flow.return
  // CHECK: count

  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]} 
    ins(%arg : tensor<8x479xf32>)
    outs(%fill : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// CHECK-LABEL: @multiple_reductions
func.func @multiple_reductions(%arg0: tensor<8x479xf32>, %arg1: tensor<32x32xf32>) -> (tensor<8xf32>, tensor<32xf32>) {
  // CHECK: flow.dispatch.workgroups
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   linalg.fill
  // CHECK:   linalg.generic
  // CHECK:   flow.dispatch.tensor.store
  // CHECK:   flow.return
  // CHECK: count

  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]} 
    ins(%arg0 : tensor<8x479xf32>)
    outs(%fill : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8xf32>

  // CHECK: flow.dispatch.workgroups
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   linalg.fill
  // CHECK:   linalg.generic
  // CHECK:   flow.dispatch.tensor.store
  // CHECK:   flow.return
  // CHECK: count

  %empty2 = tensor.empty() : tensor<32xf32>
  %fill2 = linalg.fill ins(%cst : f32) outs(%empty2 : tensor<32xf32>) -> tensor<32xf32>
  %result2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]} 
    ins(%arg1 : tensor<32x32xf32>)
    outs(%fill2 : tensor<32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<32xf32>

  return %result, %result2 : tensor<8xf32>, tensor<32xf32>
}

// CHECK-LABEL: @eltwise_reduction
func.func @eltwise_reduction(%arg: tensor<8x479xf32>) -> tensor<8xf32> {
  // CHECK: flow.dispatch.workgroups
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   linalg.generic
  // CHECK:   linalg.fill
  // CHECK:   linalg.generic
  // CHECK:   flow.dispatch.tensor.store
  // CHECK:   flow.return
  // CHECK: count

  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %eltwise_output = tensor.empty() : tensor<8x479xf32>
  %eltwise = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg : tensor<8x479xf32>)
    outs(%eltwise_output : tensor<8x479xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.mulf %in, %in : f32
    linalg.yield %0 : f32
  } -> tensor<8x479xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]} 
    ins(%eltwise : tensor<8x479xf32>)
    outs(%fill : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8xf32>
  return %result : tensor<8xf32>
}

func.func @reduction_eltwise(%arg: tensor<8x479xf32>) -> tensor<8xf32> {
  // CHECK: flow.dispatch.workgroups
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   linalg.fill
  // CHECK:   linalg.generic
  // CHECK:   linalg.generic
  // CHECK:   flow.dispatch.tensor.store
  // CHECK:   flow.return
  // CHECK: count

  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %reduction = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]} 
    ins(%arg : tensor<8x479xf32>)
    outs(%fill : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8xf32>
  %eltwise_output = tensor.empty() : tensor<8xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%reduction : tensor<8xf32>)
    outs(%eltwise_output : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = math.sqrt %in : f32
    linalg.yield %0 : f32
  } -> tensor<8xf32>
  return %result : tensor<8xf32>
}

func.func @eltwise_reduction_eltwise(%arg: tensor<8x479xf32>) -> tensor<8xf32> {
  // CHECK: flow.dispatch.workgroups
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   flow.dispatch.tensor.load
  // CHECK:   linalg.generic
  // CHECK:   linalg.fill
  // CHECK:   linalg.generic
  // CHECK:   linalg.generic
  // CHECK:   flow.dispatch.tensor.store
  // CHECK:   flow.return
  // CHECK: count

  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %eltwise_output = tensor.empty() : tensor<8x479xf32>
  %eltwise = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg : tensor<8x479xf32>)
    outs(%eltwise_output : tensor<8x479xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.mulf %in, %in : f32
    linalg.yield %0 : f32
  } -> tensor<8x479xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %reduction = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]} 
    ins(%eltwise : tensor<8x479xf32>)
    outs(%fill : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8xf32>
  %eltwise_output2 = tensor.empty() : tensor<8xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%reduction : tensor<8xf32>)
    outs(%eltwise_output2 : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = math.sqrt %in : f32
    linalg.yield %0 : f32
  } -> tensor<8xf32>
  return %result : tensor<8xf32>
}
