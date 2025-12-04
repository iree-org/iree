// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-fuse-consumers)" --split-input-file | FileCheck %s

// Positive Tests:
//* - DPS input
//* - DPS init
//*  - With dim alloc and with init alloc
//  - ReifyResultShapes user TODO: Implement pad
//* - Multiple operands
//* - Multiple fusion sites
//* - Multiple fusion sites across different blocks
//* - Multiple fusion sites with repeated result use
//* - Chained fusion
//* - Multiple fused operation results

func.func @fuse_into_generic(%arg0: tensor<?x?xi32>, %dest: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = linalg.copy ins(%0: tensor<?x?xi32>) outs(%dest: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @fuse_into_generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<?x?xi32>

//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<4x5xi32>
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[SLICE:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], %[[ID1]]] [4, 5]
//       CHECK:    %[[COPY:.+]] = linalg.copy ins(%[[CST]]{{.*}} outs(%[[SLICE]]
//       CHECK:    pcf.write_slice %[[COPY]] into %[[REF]][%[[ID0]], %[[ID1]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

func.func @fuse_into_loop(%arg0: tensor<?x?xi32>, %dest: tensor<?x?xi32>, %n0: index, %n1: index) -> tensor<?x?xi32> {
  %0 = pcf.loop scope(#pcf.dummy_scope) count(%n0, %n1)
    execute(%ref = %arg0)[%id0: index, %id1: index]
            : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
           -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = linalg.copy ins(%0: tensor<?x?xi32>) outs(%dest: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @fuse_into_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<?x?xi32>

//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<4x5xi32>
//       CHECK:  %[[LOOP:.+]] = pcf.loop scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[SLICE:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], %[[ID1]]] [4, 5]
//       CHECK:    %[[COPY:.+]] = linalg.copy ins(%[[CST]]{{.*}} outs(%[[SLICE]]
//       CHECK:    pcf.write_slice %[[COPY]] into %[[REF]][%[[ID0]], %[[ID1]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[LOOP]]

// -----

func.func @fuse_dps_init_into_generic(%arg0: tensor<?x?xi32>, %src: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = linalg.copy ins(%src: tensor<?x?xi32>) outs(%0: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @fuse_dps_init_into_generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9_]+]]: tensor<?x?xi32>

//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<4x5xi32>
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[ARG0]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[SLICE:.+]] = tensor.extract_slice %[[SRC]][%[[ID0]], %[[ID1]]] [4, 5]
//       CHECK:    %[[COPY:.+]] = linalg.copy ins(%[[SLICE]]{{.*}} outs(%[[CST]]
//       CHECK:    pcf.write_slice %[[COPY]] into %[[REF]][%[[ID0]], %[[ID1]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

func.func @fuse_dps_init_into_generic_with_alloc_dims(%d0: index, %d1: index, %src: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xi32>{%d0, %d1}) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = linalg.copy ins(%src: tensor<?x?xi32>) outs(%0: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @fuse_dps_init_into_generic_with_alloc_dims
//  CHECK-SAME:   %[[D0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9_]+]]: tensor<?x?xi32>

//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<4x5xi32>
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//  CHECK-NEXT:         : (!pcf.sref<?x?xi32
//  CHECK-NEXT:        -> (tensor<?x?xi32>{%[[D0]], %[[D1]]}
//       CHECK:    %[[SLICE:.+]] = tensor.extract_slice %[[SRC]][%[[ID0]], %[[ID1]]] [4, 5]
//       CHECK:    %[[COPY:.+]] = linalg.copy ins(%[[SLICE]]{{.*}} outs(%[[CST]]
//       CHECK:    pcf.write_slice %[[COPY]] into %[[REF]][%[[ID0]], %[[ID1]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

func.func @fuse_multi_operand_consumer_into_generic(%arg0: tensor<?x?xi32>, %d0: index, %d1: index, %dest: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0:2 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0, %ref1)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>, !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xi32>, tensor<?x?xi32>{%d0, %d1}) {
    %cst_5 = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst_5 into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    %true = arith.constant 1 : i1
    cf.assert %true, "hello"
    %cst_7 = arith.constant dense<7> : tensor<4x5xi32>
    pcf.write_slice %cst_7 into %ref1[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = linalg.add ins(%0#0, %0#1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%dest: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @fuse_multi_operand_consumer_into_generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[D0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<?x?xi32>

//   CHECK-DAG:  %[[CST5:.+]] = arith.constant dense<5> : tensor<4x5xi32>
//   CHECK-DAG:  %[[CST7:.+]] = arith.constant dense<7> : tensor<4x5xi32>
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index

// Use the assert to check that we get the correct insertion point.
//       CHECK:    cf.assert

//       CHECK:    %[[SLICE:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], %[[ID1]]] [4, 5]
//       CHECK:    %[[COPY:.+]] = linalg.add ins(%[[CST5]], %[[CST7]]{{.*}} outs(%[[SLICE]]
//       CHECK:    pcf.write_slice %[[COPY]] into %[[REF]][%[[ID0]], %[[ID1]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

func.func @fuse_into_generic_multiple_fusion_sites(%arg0: tensor<?xi32>, %dest: tensor<?xi32>) -> tensor<?xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?xi32>) {
    %cst_5 = arith.constant dense<5> : tensor<5xi32>
    %cst_7 = arith.constant dense<7> : tensor<7xi32>
    pcf.write_slice %cst_5 into %ref[%id0] [5] [1] : tensor<5xi32> into !pcf.sref<?xi32, sync(#pcf.dummy_scope)>
    %true = arith.constant 1 : i1
    cf.assert %true, "hello"
    pcf.write_slice %cst_7 into %ref[%id1] [7] [1] : tensor<7xi32> into !pcf.sref<?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = linalg.copy ins(%0: tensor<?xi32>) outs(%dest: tensor<?xi32>) -> tensor<?xi32>
  return %1 : tensor<?xi32>
}

// CHECK-LABEL: @fuse_into_generic_multiple_fusion_sites
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<?xi32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<?xi32>

//   CHECK-DAG:  %[[CST5:.+]] = arith.constant dense<5> : tensor<5xi32>
//   CHECK-DAG:  %[[CST7:.+]] = arith.constant dense<7> : tensor<7xi32>
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index

// First fusion point.
//       CHECK:    %[[SLICE5:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]]] [5]
//       CHECK:    %[[COPY5:.+]] = linalg.copy ins(%[[CST5]]{{.*}} outs(%[[SLICE5]]
//       CHECK:    pcf.write_slice %[[COPY5]] into %[[REF]][%[[ID0]]]

//       CHECK:    cf.assert

// Second fusion point.
//       CHECK:    %[[SLICE7:.+]] = tensor.extract_slice %[[DEST]][%[[ID1]]] [7]
//       CHECK:    %[[COPY7:.+]] = linalg.copy ins(%[[CST7]]{{.*}} outs(%[[SLICE7]]
//       CHECK:    pcf.write_slice %[[COPY7]] into %[[REF]][%[[ID1]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

func.func @fuse_into_generic_multiple_fusion_sites_in_control_flow(%arg0: tensor<?xi32>, %dest: tensor<?xi32>, %b: i1) -> tensor<?xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?xi32>) {
    %cst_5 = arith.constant dense<5> : tensor<3xi32>
    %cst_7 = arith.constant dense<7> : tensor<3xi32>
    cf.cond_br %b, ^bb1, ^bb2
   ^bb1:
    pcf.write_slice %cst_5 into %ref[%id] [3] [1] : tensor<3xi32> into !pcf.sref<?xi32, sync(#pcf.dummy_scope)>
    pcf.return
   ^bb2:
    pcf.write_slice %cst_7 into %ref[%id] [3] [1] : tensor<3xi32> into !pcf.sref<?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = linalg.copy ins(%0: tensor<?xi32>) outs(%dest: tensor<?xi32>) -> tensor<?xi32>
  return %1 : tensor<?xi32>
}

// CHECK-LABEL: @fuse_into_generic_multiple_fusion_sites
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<?xi32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<?xi32>

//   CHECK-DAG:  %[[CST5:.+]] = arith.constant dense<5> : tensor<3xi32>
//   CHECK-DAG:  %[[CST7:.+]] = arith.constant dense<7> : tensor<3xi32>
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID:[A-Za-z0-9_]+]]: index

//       CHECK:   ^bb1
//       CHECK:    %[[SLICE5:.+]] = tensor.extract_slice %[[DEST]][%[[ID]]] [3]
//       CHECK:    %[[COPY5:.+]] = linalg.copy ins(%[[CST5]]{{.*}} outs(%[[SLICE5]]
//       CHECK:    pcf.write_slice %[[COPY5]] into %[[REF]][%[[ID]]]
//       CHECK:    pcf.return

//       CHECK:   ^bb2
//       CHECK:    %[[SLICE7:.+]] = tensor.extract_slice %[[DEST]][%[[ID]]] [3]
//       CHECK:    %[[COPY7:.+]] = linalg.copy ins(%[[CST7]]{{.*}} outs(%[[SLICE7]]
//       CHECK:    pcf.write_slice %[[COPY7]] into %[[REF]][%[[ID]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

func.func @fuse_into_generic_multiple_fusion_sites_with_repeated_result_use(
    %arg0: tensor<?xi32>, %dest: tensor<?xi32>, %b: i1) -> tensor<?xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?xi32>) {
    %cst_5 = arith.constant dense<5> : tensor<3xi32>
    %cst_7 = arith.constant dense<7> : tensor<3xi32>
    cf.cond_br %b, ^bb1, ^bb2
   ^bb1:
    pcf.write_slice %cst_5 into %ref[%id] [3] [1] : tensor<3xi32> into !pcf.sref<?xi32, sync(#pcf.dummy_scope)>
    pcf.return
   ^bb2:
    pcf.write_slice %cst_7 into %ref[%id] [3] [1] : tensor<3xi32> into !pcf.sref<?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = linalg.add ins(%0, %0 : tensor<?xi32>, tensor<?xi32>) outs(%dest: tensor<?xi32>) -> tensor<?xi32>
  return %1 : tensor<?xi32>
}

// CHECK-LABEL: @fuse_into_generic_multiple_fusion_sites_with_repeated_result_use
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<?xi32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<?xi32>

//   CHECK-DAG:  %[[CST5:.+]] = arith.constant dense<5> : tensor<3xi32>
//   CHECK-DAG:  %[[CST7:.+]] = arith.constant dense<7> : tensor<3xi32>
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID:[A-Za-z0-9_]+]]: index

//       CHECK:   ^bb1
//       CHECK:    %[[SLICE5:.+]] = tensor.extract_slice %[[DEST]][%[[ID]]] [3]
//       CHECK:    %[[ADD5:.+]] = linalg.add ins(%[[CST5]], %[[CST5]]{{.*}} outs(%[[SLICE5]]
//       CHECK:    pcf.write_slice %[[ADD5]] into %[[REF]][%[[ID]]]
//       CHECK:    pcf.return

//       CHECK:   ^bb2
//       CHECK:    %[[SLICE7:.+]] = tensor.extract_slice %[[DEST]][%[[ID]]] [3]
//       CHECK:    %[[ADD7:.+]] = linalg.add ins(%[[CST7]], %[[CST7]]{{.*}} outs(%[[SLICE7]]
//       CHECK:    pcf.write_slice %[[ADD7]] into %[[REF]][%[[ID]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

func.func @fuse_into_generic_diamond(%arg0: tensor<?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xf32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xf32>) {
    %cst = arith.constant dense<5.0> : tensor<4x5xf32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xf32> into !pcf.sref<?x?xf32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = linalg.exp ins(%0: tensor<?x?xf32>) outs(%dest: tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.sqrt ins(%0: tensor<?x?xf32>) outs(%dest: tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = linalg.add ins(%1, %2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%dest: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// CHECK-LABEL: @fuse_into_generic_diamond
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<?x?xf32>

//       CHECK:  %[[CST:.+]] = arith.constant dense<{{.*}}> : tensor<4x5xf32>
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index

//       CHECK:    %[[SLICE1:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], %[[ID1]]] [4, 5]
//       CHECK:    %[[SQRT:.+]] = linalg.sqrt ins(%[[CST]]{{.*}} outs(%[[SLICE1]]
//       CHECK:    %[[SLICE0:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], %[[ID1]]] [4, 5]
//       CHECK:    %[[EXP:.+]] = linalg.exp ins(%[[CST]]{{.*}} outs(%[[SLICE0]]
//       CHECK:    %[[SLICE2:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], %[[ID1]]] [4, 5]
//       CHECK:    %[[ADD:.+]] = linalg.add ins(%[[EXP]], %[[SQRT]]{{.*}} outs(%[[SLICE2]]
//       CHECK:    pcf.write_slice %[[ADD]] into %[[REF]][%[[ID0]], %[[ID1]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

func.func @fuse_into_generic(%arg0: tensor<?x?xf32>, %dest: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xf32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xf32>) {
    %cst = arith.constant dense<5.0> : tensor<4x5xf32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xf32> into !pcf.sref<?x?xf32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1:2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d1, d0)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1, d0)>
    ], iterator_types = ["parallel", "parallel"]}
    ins(%0 : tensor<?x?xf32>) outs(%dest, %dest : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32, %out_0: f32):
    %exp = math.exp %in : f32
    %sqrt = math.sqrt %in : f32
    linalg.yield %exp, %sqrt : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  return %1#0, %1#1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// CHECK-LABEL: @fuse_into_generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<?x?xf32>

//       CHECK:  %[[CST:.+]] = arith.constant dense<{{.*}}> : tensor<4x5xf32>
//       CHECK:  %[[GENERIC:.+]]:2 = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%[[REF0:[A-Za-z0-9_]+]] = %[[DEST]], %[[REF1:[A-Za-z0-9_]+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[SLICE0:.+]] = tensor.extract_slice %[[DEST]][%[[ID1]], %[[ID0]]] [5, 4]
//       CHECK:    %[[SLICE1:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], %[[ID1]]] [4, 5]
//       CHECK:    %[[FUSED:.+]]:2 = linalg.generic {{.*}} ins(%[[CST]]{{.*}} outs(%[[SLICE0]], %[[SLICE1]]
//       CHECK:    pcf.write_slice %[[FUSED]]#0 into %[[REF0]][%[[ID1]], %[[ID0]]]
//       CHECK:    pcf.write_slice %[[FUSED]]#1 into %[[REF1]][%[[ID0]], %[[ID1]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]#0, %[[GENERIC]]#1

// -----

// Negative Tests:
//  - Unsupported sync scope
//  - Unsupported ref user
//  - Operand dominance
//  - Region dominance
//  - Multiple operands iteration space mismatch
//  - Multiple operands without dominant insertion point

func.func @no_fuse_non_parent_sync_scope(%arg0: tensor<?x?xi32>, %dest: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, #pcf.dummy_scope>
    pcf.return
  }
  %1 = linalg.copy ins(%0: tensor<?x?xi32>) outs(%dest: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @no_fuse_non_parent_sync_scope

//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//       CHECK:  %[[COPY:.+]] = linalg.copy ins(%[[GENERIC]]
//       CHECK:  return %[[COPY]]

// -----

func.func @no_fuse_unsupported_user(%arg0: tensor<?x?xi32>, %dest: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    util.optimization_barrier %ref : !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = linalg.copy ins(%0: tensor<?x?xi32>) outs(%dest: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @no_fuse_unsupported_user

//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//       CHECK:  %[[COPY:.+]] = linalg.copy ins(%[[GENERIC]]
//       CHECK:  return %[[COPY]]

// -----

func.func @no_fuse_dominating_operand(%arg0: tensor<?x?xi32>, %d0: index, %d1: index) -> tensor<?x?xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %empty = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  %1 = linalg.copy ins(%0: tensor<?x?xi32>) outs(%empty: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @no_fuse_dominating_operand

//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//       CHECK:  %[[COPY:.+]] = linalg.copy ins(%[[GENERIC]]
//       CHECK:  return %[[COPY]]

// -----

func.func @no_fuse_iteration_space_mismatch(%arg0: tensor<?x?xi32>, %d0: index, %d1: index, %dest: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0:2 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0, %ref1)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>, !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xi32>, tensor<?x?xi32>{%d0, %d1}) {
    %cst_5 = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst_5 into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    %cst_7 = arith.constant dense<7> : tensor<5x4xi32>
    pcf.write_slice %cst_7 into %ref1[%id1, %id0] [5, 4] [1, 1] : tensor<5x4xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = linalg.add ins(%0#0, %0#1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%dest: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @no_fuse_iteration_space_mismatch

//       CHECK:  %[[GENERIC:.+]]:2 = pcf.generic scope(#pcf.dummy_scope)
//       CHECK:  %[[ADD:.+]] = linalg.add ins(%[[GENERIC]]#0, %[[GENERIC]]#1
//       CHECK:  return %[[ADD]]

// -----

func.func @no_fuse_no_insertion_point(%arg0: tensor<?x?xi32>, %d0: index, %d1: index, %dest: tensor<?x?xi32>, %b: i1) -> tensor<?x?xi32> {
  %0:2 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0, %ref1)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>, !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xi32>, tensor<?x?xi32>{%d0, %d1}) {
    scf.if %b {
      %cst_5 = arith.constant dense<5> : tensor<5x4xi32>
      pcf.write_slice %cst_5 into %ref[%id0, %id1] [5, 4] [1, 1] : tensor<5x4xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    } else {
      %cst_7 = arith.constant dense<7> : tensor<5x4xi32>
      pcf.write_slice %cst_7 into %ref1[%id0, %id1] [5, 4] [1, 1] : tensor<5x4xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    }
    pcf.return
  }
  %1 = linalg.add ins(%0#0, %0#1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%dest: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @no_fuse_no_insertion_point

//       CHECK:  %[[GENERIC:.+]]:2 = pcf.generic scope(#pcf.dummy_scope)
//       CHECK:  %[[ADD:.+]] = linalg.add ins(%[[GENERIC]]#0, %[[GENERIC]]#1
//       CHECK:  return %[[ADD]]

// -----

// Extract slice consumer tests

func.func @fuse_extract_slice_into_generic(%arg0: tensor<?x?xi32>) -> tensor<4x5xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = tensor.extract_slice %0[0, 0] [4, 5] [1, 1] : tensor<?x?xi32> to tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

// CHECK-LABEL: @fuse_extract_slice_into_generic

//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<4x5xi32>
//       CHECK:  %[[INIT:.+]] = tensor.extract_slice %{{.+}}[0, 0] [4, 5]
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%{{.+}} = %[[INIT]])
//       CHECK:    -> (tensor<4x5xi32>)
//       CHECK:    pcf.write_slice %{{.+}} into %{{.+}}
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

func.func @fuse_extract_slice_into_loop(%arg0: tensor<?x?xi32>, %n0: index, %n1: index) -> tensor<4x5xi32> {
  %0 = pcf.loop scope(#pcf.dummy_scope) count(%n0, %n1)
    execute(%ref = %arg0)[%id0: index, %id1: index]
            : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
           -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = tensor.extract_slice %0[0, 0] [4, 5] [1, 1] : tensor<?x?xi32> to tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

// CHECK-LABEL: @fuse_extract_slice_into_loop

//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<4x5xi32>
//       CHECK:  %[[INIT:.+]] = tensor.extract_slice %{{.+}}[0, 0] [4, 5]
//       CHECK:  %[[LOOP:.+]] = pcf.loop scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%{{.+}} = %[[INIT]])
//       CHECK:    -> (tensor<4x5xi32>)
//       CHECK:    pcf.write_slice %{{.+}} into %{{.+}}
//       CHECK:    pcf.return
//       CHECK:  return %[[LOOP]]

// -----

func.func @fuse_extract_slice_multiple_write_slices(%arg0: tensor<?xi32>) -> tensor<5xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?xi32>) {
    %cst_5 = arith.constant dense<5> : tensor<3xi32>
    %cst_7 = arith.constant dense<7> : tensor<2xi32>
    pcf.write_slice %cst_5 into %ref[%id0] [3] [1] : tensor<3xi32> into !pcf.sref<?xi32, sync(#pcf.dummy_scope)>
    pcf.write_slice %cst_7 into %ref[%id1] [2] [1] : tensor<2xi32> into !pcf.sref<?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = tensor.extract_slice %0[0] [5] [1] : tensor<?xi32> to tensor<5xi32>
  return %1 : tensor<5xi32>
}

// CHECK-LABEL: @fuse_extract_slice_multiple_write_slices

//   CHECK-DAG:  %[[CST5:.+]] = arith.constant dense<5> : tensor<3xi32>
//   CHECK-DAG:  %[[CST7:.+]] = arith.constant dense<7> : tensor<2xi32>
//       CHECK:  %[[INIT:.+]] = tensor.extract_slice %{{.+}}[0] [5]
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%{{.+}} = %[[INIT]])
//       CHECK:    -> (tensor<5xi32>)
//       CHECK:    pcf.write_slice %{{.+}} into %{{.+}}[%{{.+}}]
//       CHECK:    pcf.write_slice %{{.+}} into %{{.+}}[%{{.+}}]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

func.func @fuse_extract_slice_with_tied_init(%arg0: tensor<8x10xi32>) -> tensor<4x5xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x10xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<8x10xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<8x10xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = tensor.extract_slice %0[0, 0] [4, 5] [1, 1] : tensor<8x10xi32> to tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

// CHECK-LABEL: @fuse_extract_slice_with_tied_init
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x10xi32>

//       CHECK:  %[[INIT:.+]] = tensor.extract_slice %[[ARG0]][0, 0] [4, 5]
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])
//       CHECK:    -> (tensor<4x5xi32>)
//       CHECK:    pcf.write_slice %{{.+}} into %[[REF]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Negative test: non-zero offset extract_slice

func.func @no_fuse_nonzero_offset_extract_slice(%arg0: tensor<?x?xi32>) -> tensor<4x5xi32> {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  %1 = tensor.extract_slice %0[1, 2] [4, 5] [1, 1] : tensor<?x?xi32> to tensor<4x5xi32>
  return %1 : tensor<4x5xi32>
}

// CHECK-LABEL: @no_fuse_nonzero_offset_extract_slice

//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//       CHECK:  %[[SLICE:.+]] = tensor.extract_slice %[[GENERIC]][1, 2]
//       CHECK:  return %[[SLICE]]

// -----

// Negative test: multiple uses of result (two extract_slices at different offsets)

func.func @no_fuse_multiple_uses(%arg0: tensor<?x?xi32>) -> (tensor<4x5xi32>, tensor<4x5xi32>) {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>)
        -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#pcf.dummy_scope)>
    pcf.return
  }
  // Both extract_slices use the result, so neither can fuse (would need different result types)
  %1 = tensor.extract_slice %0[0, 0] [4, 5] [1, 1] : tensor<?x?xi32> to tensor<4x5xi32>
  %2 = tensor.extract_slice %0[4, 5] [4, 5] [1, 1] : tensor<?x?xi32> to tensor<4x5xi32>
  return %1, %2 : tensor<4x5xi32>, tensor<4x5xi32>
}

// CHECK-LABEL: @no_fuse_multiple_uses

//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.dummy_scope)
//       CHECK:  %[[SLICE1:.+]] = tensor.extract_slice %[[GENERIC]][0, 0]
//       CHECK:  %[[SLICE2:.+]] = tensor.extract_slice %[[GENERIC]][4, 5]
//       CHECK:  return %[[SLICE1]], %[[SLICE2]]
