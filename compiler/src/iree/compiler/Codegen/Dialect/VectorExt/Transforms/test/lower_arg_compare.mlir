// RUN: iree-opt %s -pass-pipeline='builtin.module(func.func(iree-vector-ext-lower-arg-compare-to-vector))' --split-input-file --mlir-print-local-scope | FileCheck %s

func.func @argmax_1d_i32(%input: vector<128xf32>,
                         %init_val: vector<f32>,
                         %init_idx: vector<i32>)
    -> (vector<f32>, vector<i32>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(0)
    ins(%input : vector<128xf32>)
    inits(%init_val, %init_idx : vector<f32>, vector<i32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<f32>, vector<i32>
  return %result#0, %result#1 : vector<f32>, vector<i32>
}
// CHECK-LABEL: func.func @argmax_1d_i32
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C128:.*]] = arith.constant 128 : index
// CHECK:         %[[INIT_VAL:.*]] = vector.extract %{{.*}}[] : f32 from vector<f32>
// CHECK:         %[[INIT_IDX:.*]] = vector.extract %{{.*}}[] : i32 from vector<i32>
// CHECK:         %[[FOR:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK-SAME:        iter_args(%[[BV:.*]] = %[[INIT_VAL]], %[[BI:.*]] = %[[INIT_IDX]])
// CHECK:           %[[CAND:.*]] = vector.extract %{{.*}}[%[[IV]]] : f32 from vector<128xf32>
// CHECK:           %[[CMP:.*]] = arith.cmpf ogt, %[[CAND]], %[[BV]] : f32
// CHECK:           %[[SEL_VAL:.*]] = arith.select %[[CMP]], %[[CAND]], %[[BV]] : f32
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[IV]] : index to i32
// CHECK:           %[[SEL_IDX:.*]] = arith.select %[[CMP]], %[[IDX]], %[[BI]] : i32
// CHECK:           scf.yield %[[SEL_VAL]], %[[SEL_IDX]]
// CHECK:         vector.broadcast %[[FOR]]#0 : f32 to vector<f32>
// CHECK:         vector.broadcast %[[FOR]]#1 : i32 to vector<i32>

// -----

func.func @argmax_1d_i64(%input: vector<128xf32>,
                         %init_val: vector<f32>,
                         %init_idx: vector<i64>)
    -> (vector<f32>, vector<i64>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(0)
    ins(%input : vector<128xf32>)
    inits(%init_val, %init_idx : vector<f32>, vector<i64>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<f32>, vector<i64>
  return %result#0, %result#1 : vector<f32>, vector<i64>
}
// CHECK-LABEL: func.func @argmax_1d_i64
// CHECK:         arith.index_cast %{{.*}} : index to i64

// -----

func.func @argmax_2d_reduce_last_dim(%input: vector<4x128xf32>,
                                     %init_val: vector<4xf32>,
                                     %init_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi32>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(1)
    ins(%input : vector<4x128xf32>)
    inits(%init_val, %init_idx : vector<4xf32>, vector<4xi32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}
// CHECK-LABEL: func.func @argmax_2d_reduce_last_dim
// CHECK-NOT:     vector.transpose
// CHECK:         arith.constant 128 : index
// CHECK-COUNT-4: scf.for {{.*}} to %{{.*}} step
// CHECK-NOT:     scf.for
// CHECK:         return %{{.*}}, %{{.*}} : vector<4xf32>, vector<4xi32>

// -----

func.func @argmax_2d_reduce_first_dim(%input: vector<2x4xf32>,
                                      %init_val: vector<4xf32>,
                                      %init_idx: vector<4xi64>)
    -> (vector<4xf32>, vector<4xi64>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(0)
    ins(%input : vector<2x4xf32>)
    inits(%init_val, %init_idx : vector<4xf32>, vector<4xi64>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi64>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi64>
}
// CHECK-LABEL: func.func @argmax_2d_reduce_first_dim
// CHECK-DAG:     arith.constant 2 : index
// CHECK-DAG:     vector.transpose %{{.*}}, [1, 0] : vector<2x4xf32> to vector<4x2xf32>
// CHECK-COUNT-4: scf.for {{.*}} to %{{.*}} step
// CHECK-NOT:     scf.for
// CHECK:         arith.index_cast %{{.*}} : index to i64
// CHECK:         return %{{.*}}, %{{.*}} : vector<4xf32>, vector<4xi64>

// -----

func.func @argmax_3d_reduce_first_dim(%input: vector<2x3x4xf32>,
                                      %init_val: vector<3x4xf32>,
                                      %init_idx: vector<3x4xi64>)
    -> (vector<3x4xf32>, vector<3x4xi64>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(0)
    ins(%input : vector<2x3x4xf32>)
    inits(%init_val, %init_idx : vector<3x4xf32>, vector<3x4xi64>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<3x4xf32>, vector<3x4xi64>
  return %result#0, %result#1 : vector<3x4xf32>, vector<3x4xi64>
}
// CHECK-LABEL: func.func @argmax_3d_reduce_first_dim
// CHECK-DAG:     arith.constant 2 : index
// CHECK-DAG:     vector.transpose %{{.*}}, [1, 2, 0] : vector<2x3x4xf32> to vector<3x4x2xf32>
// CHECK-COUNT-12: scf.for {{.*}} to %{{.*}} step
// CHECK-NOT:     scf.for
// CHECK:         return %{{.*}}, %{{.*}} : vector<3x4xf32>, vector<3x4xi64>

// -----

func.func @argmax_3d_reduce_middle_dim(%input: vector<3x4x5xf32>,
                                       %init_val: vector<3x5xf32>,
                                       %init_idx: vector<3x5xi32>)
    -> (vector<3x5xf32>, vector<3x5xi32>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(1)
    ins(%input : vector<3x4x5xf32>)
    inits(%init_val, %init_idx : vector<3x5xf32>, vector<3x5xi32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<3x5xf32>, vector<3x5xi32>
  return %result#0, %result#1 : vector<3x5xf32>, vector<3x5xi32>
}
// CHECK-LABEL: func.func @argmax_3d_reduce_middle_dim
// CHECK-DAG:     arith.constant 4 : index
// CHECK-DAG:     vector.transpose %{{.*}}, [0, 2, 1] : vector<3x4x5xf32> to vector<3x5x4xf32>
// CHECK-COUNT-15: scf.for {{.*}} to %{{.*}} step
// CHECK-NOT:     scf.for
// CHECK:         return %{{.*}}, %{{.*}} : vector<3x5xf32>, vector<3x5xi32>

// -----

func.func @argmax_explicit_index_last_dim(%input_val: vector<4x32xf32>,
                                          %input_idx: vector<4x32xi32>,
                                          %init_val: vector<4xf32>,
                                          %init_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi32>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(1)
    ins(%input_val, %input_idx : vector<4x32xf32>, vector<4x32xi32>)
    inits(%init_val, %init_idx : vector<4xf32>, vector<4xi32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}
// CHECK-LABEL: func.func @argmax_explicit_index_last_dim
// CHECK-NOT:     vector.transpose
// CHECK-NOT:     arith.index_cast
// CHECK-COUNT-4: scf.for
// CHECK-NOT:     scf.for
// CHECK:         vector.extract %{{.*}}[%{{.*}}] : i32 from vector<32xi32>
// CHECK:         return %{{.*}}, %{{.*}} : vector<4xf32>, vector<4xi32>

// -----

func.func @argmax_explicit_index_first_dim(%input_val: vector<2x4xf32>,
                                           %input_idx: vector<2x4xi32>,
                                           %init_val: vector<4xf32>,
                                           %init_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi32>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(0)
    ins(%input_val, %input_idx : vector<2x4xf32>, vector<2x4xi32>)
    inits(%init_val, %init_idx : vector<4xf32>, vector<4xi32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}
// CHECK-LABEL: func.func @argmax_explicit_index_first_dim
// CHECK-DAG:     vector.transpose %{{.*}}, [1, 0] : vector<2x4xf32> to vector<4x2xf32>
// CHECK-DAG:     vector.transpose %{{.*}}, [1, 0] : vector<2x4xi32> to vector<4x2xi32>
// CHECK-NOT:     arith.index_cast
// CHECK-COUNT-4: scf.for
// CHECK-NOT:     scf.for
// CHECK:         return %{{.*}}, %{{.*}} : vector<4xf32>, vector<4xi32>

// -----

func.func @argmax_index_base(%input: vector<128xf32>,
                             %init_val: vector<f32>,
                             %init_idx: vector<i32>,
                             %base: index)
    -> (vector<f32>, vector<i32>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(0)
    ins(%input : vector<128xf32>)
    inits(%init_val, %init_idx : vector<f32>, vector<i32>)
    index_base(%base : index) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<f32>, vector<i32>
  return %result#0, %result#1 : vector<f32>, vector<i32>
}
// CHECK-LABEL: func.func @argmax_index_base
// CHECK-SAME:    %[[BASE:[a-zA-Z0-9]+]]: index
// CHECK:         scf.for %[[IV:[a-zA-Z0-9]+]] =
// CHECK:           arith.addi %[[BASE]], %[[IV]] : index
// CHECK:           arith.index_cast %{{.*}} : index to i32
// CHECK:         vector.broadcast %{{.*}} : f32 to vector<f32>
// CHECK:         vector.broadcast %{{.*}} : i32 to vector<i32>

// -----

func.func @argmin_1d(%input: vector<128xf32>,
                     %init_val: vector<f32>,
                     %init_idx: vector<i32>)
    -> (vector<f32>, vector<i32>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(0)
    ins(%input : vector<128xf32>)
    inits(%init_val, %init_idx : vector<f32>, vector<i32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf olt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<f32>, vector<i32>
  return %result#0, %result#1 : vector<f32>, vector<i32>
}
// CHECK-LABEL: func.func @argmin_1d
// CHECK:         scf.for
// CHECK:           arith.cmpf olt, %{{.*}}, %{{.*}} : f32
// CHECK:           arith.select
// CHECK:           arith.select
// CHECK:           scf.yield
// CHECK:         vector.broadcast %{{.*}} : f32 to vector<f32>
// CHECK:         vector.broadcast %{{.*}} : i32 to vector<i32>

// -----

func.func @argmax_captured_predicate(%input: vector<128xf32>,
                                     %init_val: vector<f32>,
                                     %init_idx: vector<i32>,
                                     %external_cmp: i1)
    -> (vector<f32>, vector<i32>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(0)
    ins(%input : vector<128xf32>)
    inits(%init_val, %init_idx : vector<f32>, vector<i32>) {
  ^bb0(%a: f32, %b: f32):
    iree_vector_ext.yield %external_cmp : i1
  } -> vector<f32>, vector<i32>
  return %result#0, %result#1 : vector<f32>, vector<i32>
}
// CHECK-LABEL: func.func @argmax_captured_predicate
// CHECK-SAME:    %[[EXT_CMP:[a-zA-Z0-9]+]]: i1
// CHECK:         scf.for
// CHECK:           arith.select %[[EXT_CMP]]
// CHECK:           arith.select %[[EXT_CMP]]
// CHECK:           scf.yield

// -----

func.func @argmax_multi_op_comparator(%input: vector<128xf32>,
                                      %init_val: vector<f32>,
                                      %init_idx: vector<i32>)
    -> (vector<f32>, vector<i32>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(0)
    ins(%input : vector<128xf32>)
    inits(%init_val, %init_idx : vector<f32>, vector<i32>) {
  ^bb0(%a: f32, %b: f32):
    %diff = arith.subf %a, %b : f32
    %zero = arith.constant 0.0 : f32
    %cmp = arith.cmpf ogt, %diff, %zero : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<f32>, vector<i32>
  return %result#0, %result#1 : vector<f32>, vector<i32>
}
// CHECK-LABEL: func.func @argmax_multi_op_comparator
// CHECK:         scf.for
// CHECK:           arith.subf
// CHECK:           arith.cmpf ogt
// CHECK:           arith.select
// CHECK:           arith.select
// CHECK:           scf.yield

// -----

func.func @argmax_index_base_2d(%input: vector<4x128xf32>,
                                 %init_val: vector<4xf32>,
                                 %init_idx: vector<4xi32>,
                                 %base: index)
    -> (vector<4xf32>, vector<4xi32>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(1)
    ins(%input : vector<4x128xf32>)
    inits(%init_val, %init_idx : vector<4xf32>, vector<4xi32>)
    index_base(%base : index) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}
// CHECK-LABEL: func.func @argmax_index_base_2d
// CHECK-COUNT-4: arith.addi
// CHECK-NOT:     arith.addi
// CHECK-NOT:     iree_vector_ext.arg_compare

// -----

func.func @argmax_scalable_rejection(%input: vector<[128]xf32>,
                                     %init_val: vector<f32>,
                                     %init_idx: vector<i32>)
    -> (vector<f32>, vector<i32>) {
  %result:2 = iree_vector_ext.arg_compare
    dimension(0)
    ins(%input : vector<[128]xf32>)
    inits(%init_val, %init_idx : vector<f32>, vector<i32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_vector_ext.yield %cmp : i1
  } -> vector<f32>, vector<i32>
  return %result#0, %result#1 : vector<f32>, vector<i32>
}
// CHECK-LABEL: func.func @argmax_scalable_rejection
// CHECK:         iree_vector_ext.arg_compare
// CHECK-NOT:     scf.for
