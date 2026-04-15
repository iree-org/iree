// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-linalg-ext-to-loops))" %s | FileCheck %s

func.func @sort_1d(%arg0: memref<128xi32>) {
  iree_linalg_ext.sort dimension(0)
    outs(%arg0 : memref<128xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %0 = arith.cmpi sgt, %arg2, %arg3 : i32
    iree_linalg_ext.yield %0 : i1
  }
  return
}
// CHECK-LABEL: func.func @sort_1d
// CHECK-SAME:    %[[BUF:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C127:.+]] = arith.constant 127 : index
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK:           scf.for %[[ARG2:.+]] = %[[C0]] to %[[C127]] step %[[C1]]
// CHECK:             %[[T1:.+]] = arith.addi %[[ARG2]], %[[C1]] : index
// CHECK:             %[[V1:.+]] = memref.load %[[BUF]][%[[ARG2]]]
// CHECK:             %[[V2:.+]] = memref.load %[[BUF]][%[[T1]]]
// CHECK:             %[[COND:.+]] = arith.cmpi sgt, %[[V1]], %[[V2]] : i32
// CHECK:             scf.if %[[COND]] {
// CHECK:             } else {
// CHECK:               %[[T2:.+]] = arith.addi %[[ARG2]], %[[C1]] : index
// CHECK:               memref.store %[[V2]], %[[BUF]][%[[ARG2]]]
// CHECK:               memref.store %[[V1]], %[[BUF]][%[[T2]]]
// CHECK:             }

// -----

func.func @sort_2d(%arg0: memref<16x32xi32>) {
  iree_linalg_ext.sort dimension(0)
    outs(%arg0 : memref<16x32xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %0 = arith.cmpi sgt, %arg2, %arg3 : i32
    iree_linalg_ext.yield %0 : i1
  }
  return
}
// CHECK-LABEL: func.func @sort_2d
// CHECK-SAME:    %[[BUF:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C15:.+]] = arith.constant 15 : index
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C16]] step %[[C1]]
// CHECK:           scf.for %[[ARG2:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK:             scf.for %[[ARG3:.+]] = %[[C0]] to %[[C15]] step %[[C1]]
// CHECK:               %[[T1:.+]] = arith.addi %[[ARG3]], %[[C1]] : index
// CHECK:               %[[V1:.+]] = memref.load %[[BUF]][%[[ARG3]], %[[ARG2]]]
// CHECK:               %[[V2:.+]] = memref.load %[[BUF]][%[[T1]], %[[ARG2]]]
// CHECK:               %[[COND:.+]] = arith.cmpi sgt, %[[V1]], %[[V2]] : i32
// CHECK:               scf.if %[[COND]] {
// CHECK:               } else {
// CHECK:                 %[[T2:.+]] = arith.addi %[[ARG3]], %[[C1]] : index
// CHECK:                 memref.store %[[V2]], %[[BUF]][%[[ARG3]], %[[ARG2]]]
// CHECK:                 memref.store %[[V1]], %[[BUF]][%[[T2]], %[[ARG2]]]
// CHECK:               }

// -----

func.func @sort_multi(%arg0: memref<128xf32>, %arg1: memref<128xi32>) {
  iree_linalg_ext.sort
    dimension(0)
    outs(%arg0, %arg1 : memref<128xf32>, memref<128xi32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i32, %arg5: i32):
    %0 = arith.cmpf ogt, %arg2, %arg3 : f32
    iree_linalg_ext.yield %0 : i1
  }
  return
}
// CHECK-LABEL: func.func @sort_multi
// CHECK-SAME:    %[[BUF1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[BUF2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C127:.+]] = arith.constant 127 : index
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK:           scf.for %[[ARG2:.+]] = %[[C0]] to %[[C127]] step %[[C1]]
// CHECK:             %[[T1:.+]] = arith.addi %[[ARG2]], %[[C1]] : index
// CHECK:             %[[V1:.+]] = memref.load %[[BUF1]][%[[ARG2]]]
// CHECK:             %[[V2:.+]] = memref.load %[[BUF1]][%[[T1]]]
// CHECK:             %[[V3:.+]] = memref.load %[[BUF2]][%[[ARG2]]]
// CHECK:             %[[V4:.+]] = memref.load %[[BUF2]][%[[T1]]]
// CHECK:             %[[COND:.+]] = arith.cmpf ogt, %[[V1]], %[[V2]] : f32
// CHECK:             scf.if %[[COND]] {
// CHECK:             } else {
// CHECK:               %[[T2:.+]] = arith.addi %[[ARG2]], %[[C1]] : index
// CHECK:               memref.store %[[V2]], %[[BUF1]][%[[ARG2]]]
// CHECK:               memref.store %[[V1]], %[[BUF1]][%[[T2]]]
// CHECK:               memref.store %[[V4]], %[[BUF2]][%[[ARG2]]]
// CHECK:               memref.store %[[V3]], %[[BUF2]][%[[T2]]]
// CHECK:             }

// -----

func.func @scatter_update_scalar_1D(
    %original: memref<8xi32>, %indices: memref<3x1xi32>,
    %updates: memref<3xi32>) {
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices : memref<3xi32>, memref<3x1xi32>)
    outs(%original : memref<8xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK-LABEL: func.func @scatter_update_scalar_1D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:           %[[T1:.+]] = memref.load %[[UPDATES]][%[[I]]] : memref<3xi32>
// CHECK:           %[[T2:.+]] =  memref.load %[[INDICES]][%[[I]], %[[C0]]] : memref<3x1xi32>
// CHECK:           %[[IDX:.+]] = arith.index_cast %[[T2]] : i32 to index
// CHECK:           memref.store %[[T1]], %[[ORIGINAL]][%[[IDX]]]

// -----

func.func @scatter_update_scalar_1D_masked(
    %original: memref<8xi32>, %indices: memref<3x1xi32>,
    %mask: memref<3xi1>, %updates: memref<3xi32>) {
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices, %mask : memref<3xi32>, memref<3x1xi32>, memref<3xi1>)
    outs(%original : memref<8xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK-LABEL: func.func @scatter_update_scalar_1D_masked
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[MASK:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:           %[[T1:.+]] = memref.load %[[UPDATES]][%[[I]]] : memref<3xi32>
// CHECK:           %[[T2:.+]] = memref.load %[[INDICES]][%[[I]], %[[C0]]] : memref<3x1xi32>
// CHECK:           %[[IDX:.+]] = arith.index_cast %[[T2]] : i32 to index
// CHECK:           %[[MASK_VAL:.+]] = memref.load %[[MASK]][%[[I]]] : memref<3xi1>
// CHECK:           scf.if %[[MASK_VAL]] {
// CHECK:             memref.store %[[T1]], %[[ORIGINAL]][%[[IDX]]]

// -----

func.func @scatter_update_scalar_1D_masked_i8(
    %original: memref<8xi32>, %indices: memref<3x1xi32>,
    %mask: memref<3xi8>, %updates: memref<3xi32>) {
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices, %mask : memref<3xi32>, memref<3x1xi32>, memref<3xi8>)
    outs(%original : memref<8xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK-LABEL: func.func @scatter_update_scalar_1D_masked_i8
// CHECK:         scf.for
// CHECK:           %[[MASK_VAL:.+]] = memref.load %{{.+}}[%{{.+}}] : memref<3xi8>
// CHECK:           %[[MASK_I1:.+]] = arith.trunci %[[MASK_VAL]] : i8 to i1
// CHECK:           scf.if %[[MASK_I1]] {

// -----

func.func @scatter_batch_2D(
    %original: memref<8xi32>, %indices: memref<1x3x1xi32>,
    %updates: memref<1x3xi32>) {
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices : memref<1x3xi32>, memref<1x3x1xi32>)
    outs(%original : memref<8xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK-LABEL: func.func @scatter_batch_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         scf.for %[[I0:.+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK:           scf.for %[[I1:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:             %[[T1:.+]] = memref.load %[[UPDATES]][%[[I0]], %[[I1]]] : memref<1x3xi32>
// CHECK:             %[[T2:.+]] =  memref.load %[[INDICES]][%[[I0]], %[[I1]], %[[C0]]] : memref<1x3x1xi32>
// CHECK:             %[[IDX:.+]] = arith.index_cast %[[T2]] : i32 to index
// CHECK:             memref.store %[[T1]], %[[ORIGINAL]][%[[IDX]]]

// -----

func.func @scatter_add_scalar_2D(
    %original: memref<4x3xi32>, %indices: memref<3x2xi32>,
    %updates: memref<3xi32>) {
  iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
    ins(%updates, %indices : memref<3xi32>, memref<3x2xi32>)
    outs(%original : memref<4x3xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg1, %arg0 : i32
    iree_linalg_ext.yield %0 : i32
  }
  return
}
// CHECK-LABEL: func.func @scatter_add_scalar_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:           %[[T1:.+]] = memref.load %[[UPDATES]][%[[I]]] : memref<3xi32>
// CHECK:           %[[T2:.+]] = memref.load %[[INDICES]][%[[I]], %[[C0]]] : memref<3x2xi32>
// CHECK:           %[[IDX1:.+]] = arith.index_cast %[[T2]] : i32 to index
// CHECK:           %[[T3:.+]] = memref.load %[[INDICES]][%[[I]], %[[C1]]] : memref<3x2xi32>
// CHECK:           %[[IDX2:.+]] = arith.index_cast %[[T3]] : i32 to index
// CHECK:           %[[ORI:.+]] = memref.load %[[ORIGINAL]][%[[IDX1]], %[[IDX2]]] : memref<4x3xi32>
// CHECK:           %[[ADD:.+]] = arith.addi %[[ORI]], %[[T1]] : i32
// CHECK:           memref.store %[[ADD]], %[[ORIGINAL]][%[[IDX1]], %[[IDX2]]]

// -----

func.func @scatter_update_slice_2D(
    %original: memref<4x3xi32>, %indices: memref<2x1xi32>,
    %updates: memref<2x3xi32>) {
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices : memref<2x3xi32>, memref<2x1xi32>)
    outs(%original : memref<4x3xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK:       func.func @scatter_update_slice_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:           scf.for %[[J:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:             %[[UPDATE:.+]] = memref.load %[[UPDATES]][%[[I]], %[[J]]]
// CHECK:             %[[INDEX:.+]] = memref.load %[[INDICES]][%[[I]], %[[C0]]]
// CHECK:             %[[LOC:.+]] = arith.index_cast %[[INDEX]] : i32 to index
// CHECK:             memref.store %[[UPDATE]], %[[ORIGINAL]][%[[LOC]], %[[J]]]
// CHECK:           }
// CHECK:         }

// -----

func.func @scatter_add_scalar_1D(
    %original: memref<8xi32>, %indices: memref<3x1xi32>,
    %updates: memref<3xi32>) {
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices : memref<3xi32>, memref<3x1xi32>)
    outs(%original : memref<8xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg1, %arg0 : i32
    iree_linalg_ext.yield %0 : i32
  }
  return
}
// CHECK-LABEL: func.func @scatter_add_scalar_1D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:           %[[T1:.+]] = memref.load %[[UPDATES]][%[[I]]] : memref<3xi32>
// CHECK:           %[[T2:.+]] =  memref.load %[[INDICES]][%[[I]], %[[C0]]] : memref<3x1xi32>
// CHECK:           %[[IDX:.+]] = arith.index_cast %[[T2]] : i32 to index
// CHECK:           %[[ORI:.+]] = memref.load %[[ORIGINAL]][%[[IDX]]] : memref<8xi32>
// CHECK:           %[[ADD:.+]] = arith.addi %[[ORI]], %[[T1]] : i32
// CHECK:           memref.store %[[ADD]], %[[ORIGINAL]][%[[IDX]]]

// -----

func.func @scatter_add_slice_2D(
    %original: memref<4x3xi32>, %indices: memref<2x1xi32>,
    %updates: memref<2x3xi32>) {
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices : memref<2x3xi32>, memref<2x1xi32>)
    outs(%original : memref<4x3xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg1, %arg0 : i32
    iree_linalg_ext.yield %0 : i32
  }
  return
}
// CHECK:       func.func @scatter_add_slice_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:           scf.for %[[J:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:             %[[UPDATEVAL:.+]] = memref.load %[[UPDATES]][%[[I]], %[[J]]]
// CHECK:             %[[INDEXVAL:.+]] = memref.load %[[INDICES]][%[[I]], %[[C0]]]
// CHECK:             %[[INDEX:.+]] = arith.index_cast %[[INDEXVAL]] : i32 to index
// CHECK:             %[[ORIGINALVAL:.+]] = memref.load %[[ORIGINAL]][%[[INDEX]], %[[J]]]
// CHECK:             %[[STOREVAL:.+]] = arith.addi %[[ORIGINALVAL]], %[[UPDATEVAL]]
// CHECK:             memref.store %[[STOREVAL]], %[[ORIGINAL]][%[[INDEX]], %[[J]]]

// -----

func.func @scatter_update_scalar_dynamic_1D(
    %original: memref<?xi32>, %indices: memref<?x1xi32>,
    %updates: memref<?xi32>) {
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices : memref<?xi32>, memref<?x1xi32>)
    outs(%original : memref<?xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK-LABEL: func.func @scatter_update_scalar_dynamic_1D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[UB:.+]] = memref.dim %[[UPDATES]], %[[C0]] : memref<?xi32>
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[UB]] step %[[C1]] {
// CHECK:           %[[T1:.+]] = memref.load %[[UPDATES]][%[[I]]] : memref<?xi32>
// CHECK:           %[[T2:.+]] =  memref.load %[[INDICES]][%[[I]], %[[C0]]] : memref<?x1xi32>
// CHECK:           %[[IDX:.+]] = arith.index_cast %[[T2]] : i32 to index
// CHECK:           memref.store %[[T1]], %[[ORIGINAL]][%[[IDX]]]

// -----

func.func @scatter_add_scalar_dynamic_2D(
    %original: memref<?x?xi32>, %indices: memref<?x2xi32>,
    %updates: memref<?xi32>) {
  iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
    ins(%updates, %indices : memref<?xi32>, memref<?x2xi32>)
    outs(%original : memref<?x?xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg1, %arg0 : i32
    iree_linalg_ext.yield %0 : i32
  }
  return
}
// CHECK-LABEL: func.func @scatter_add_scalar_dynamic_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[UB:.+]] = memref.dim %[[UPDATES]], %[[C0]] : memref<?xi32>
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[UB]] step %[[C1]] {
// CHECK:           %[[T1:.+]] = memref.load %[[UPDATES]][%[[I]]] : memref<?xi32>
// CHECK:           %[[T2:.+]] = memref.load %[[INDICES]][%[[I]], %[[C0]]] : memref<?x2xi32>
// CHECK:           %[[IDX1:.+]] = arith.index_cast %[[T2]] : i32 to index
// CHECK:           %[[T3:.+]] = memref.load %[[INDICES]][%[[I]], %[[C1]]] : memref<?x2xi32>
// CHECK:           %[[IDX2:.+]] = arith.index_cast %[[T3]] : i32 to index
// CHECK:           %[[ORI:.+]] = memref.load %[[ORIGINAL]][%[[IDX1]], %[[IDX2]]] : memref<?x?xi32>
// CHECK:           %[[ADD:.+]] = arith.addi %[[ORI]], %[[T1]] : i32
// CHECK:           memref.store %[[ADD]], %[[ORIGINAL]][%[[IDX1]], %[[IDX2]]]

// -----

func.func @scatter_update_slice_dynamic_2D(
    %original: memref<?x?xi32>, %indices: memref<?x1xi32>,
    %updates: memref<?x?xi32>) {
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices : memref<?x?xi32>, memref<?x1xi32>)
    outs(%original : memref<?x?xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK:       func.func @scatter_update_slice_dynamic_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[UB1:.+]] = memref.dim %[[UPDATES]], %[[C0]] : memref<?x?xi32>
// CHECK-DAG:     %[[UB2:.+]] = memref.dim %[[UPDATES]], %[[C1]] : memref<?x?xi32>
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[UB1]] step %[[C1]] {
// CHECK:           scf.for %[[J:.+]] = %[[C0]] to %[[UB2]] step %[[C1]] {
// CHECK:             %[[UPDATEVAL:.+]] = memref.load %[[UPDATES]][%[[I]], %[[J]]]
// CHECK:             %[[INDEXVAL:.+]] = memref.load %[[INDICES]][%[[I]], %[[C0]]]
// CHECK:             %[[INDEX:.+]] = arith.index_cast %[[INDEXVAL]] : i32 to index
// CHECK:             memref.store %[[UPDATEVAL]], %[[ORIGINAL]][%[[INDEX]], %[[J]]]

// -----

func.func @fft_1D(%real: memref<16xf32>, %imag: memref<16xf32>) {
  %stage = arith.constant 1 : index
  iree_linalg_ext.fft
    ins(%stage: index)
    outs(%real, %imag: memref<16xf32>, memref<16xf32>)
  return
}
// CHECK:   #[[MAP1:.+]] = affine_map<(d0) -> (d0)>
// CHECK:       func.func @fft_1D
// CHECK-SAME:    %[[REAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[IMAG:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[COEFF:.+]] = arith.constant -3.14159274 : f32
// CHECK:         scf.for %[[K:.+]] = %[[C0]] to %[[C16]] step %[[C2]]
// CHECK:           %[[L_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[K]]] [%[[C1]]] [1]
// CHECK:           %[[L_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[K]]] [%[[C1]]] [1]
// CHECK:           %[[R_OFFSET:.+]] = arith.addi %[[K]], %[[C1]] : index
// CHECK:           %[[R_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[R_OFFSET]]] [%[[C1]]] [1]
// CHECK:           %[[R_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[R_OFFSET]]] [%[[C1]]] [1]
// CHECK:           linalg.generic
// CHECK-SAME:        indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:        iterator_types = ["parallel"]
// CHECK-SAME:        outs(%[[L_REAL_SLICE]], %[[L_IMAG_SLICE]], %[[R_REAL_SLICE]], %[[R_IMAG_SLICE]]
// CHECK:           ^bb0(%[[L_REAL:.+]]: f32, %[[L_IMAG:.+]]: f32, %[[R_REAL:.+]]: f32, %[[R_IMAG:.+]]: f32)
//
//                    Compute exp coeff.
// CHECK:             %[[J_IDX:.+]] = linalg.index 0 : index
// CHECK:             %[[J_I32:.+]] = arith.index_cast %[[J_IDX]] : index to i32
// CHECK:             %[[J_F32:.+]] = arith.sitofp %[[J_I32]] : i32 to f32
// CHECK:             %[[EXP_COEF:.+]] = arith.mulf %[[J_F32]], %[[COEFF]] : f32
// CHECK:             %[[W_REAL:.+]] = math.cos %[[EXP_COEF]]
// CHECK:             %[[W_IMAG:.+]] = math.sin %[[EXP_COEF]]
//
//                    Compute "t = w * a[k + j + mh]" by expanding
//                      (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
// CHECK-DAG:         %[[XU:.+]] = arith.mulf %[[W_REAL]], %[[R_REAL]]
// CHECK-DAG:         %[[YV:.+]] = arith.mulf %[[W_IMAG]], %[[R_IMAG]]
// CHECK-DAG:         %[[XV:.+]] = arith.mulf %[[W_REAL]], %[[R_IMAG]]
// CHECK-DAG:         %[[YU:.+]] = arith.mulf %[[W_IMAG]], %[[R_REAL]]
// CHECK:             %[[T_REAL:.+]] = arith.subf %[[XU]], %[[YV]]
// CHECK:             %[[T_IMAG:.+]] = arith.addf %[[XV]], %[[YU]]
//
//                    Compute the results.
//                      u = a[k + j];
//                      a[k + j] = u + t;
//                      a[k + j + mh] = u - t;
// CHECK:             %[[RES1:.+]] = arith.addf %[[L_REAL]], %[[T_REAL]]
// CHECK:             %[[RES2:.+]] = arith.addf %[[L_IMAG]], %[[T_IMAG]]
// CHECK:             %[[RES3:.+]] = arith.subf %[[L_REAL]], %[[T_REAL]]
// CHECK:             %[[RES4:.+]] = arith.subf %[[L_IMAG]], %[[T_IMAG]]
// CHECK:             linalg.yield %[[RES1]], %[[RES2]], %[[RES3]], %[[RES4]]

// -----

func.func @fft_2D(%real: memref<?x16xf32>, %imag: memref<?x16xf32>) {
  %stage = arith.constant 2 : index
  iree_linalg_ext.fft
    ins(%stage: index)
    outs(%real, %imag: memref<?x16xf32>, memref<?x16xf32>)
  return
}
// CHECK:   #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:       func.func @fft_2D(
// CHECK-SAME:    %[[REAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[IMAG:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[D0:.+]] = memref.dim %[[REAL]], %[[C0]] : memref<?x16xf32>
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[D0]] step %[[C1]]
// CHECK:           scf.for %[[K:.+]] = %[[C0]] to %[[C16]] step %[[C4]]
// CHECK:             %[[L_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[I]], %[[K]]] [1, %[[C2]]] [1, 1]
// CHECK:             %[[L_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[I]], %[[K]]] [1, %[[C2]]] [1, 1]
// CHECK:             %[[R_OFFSET:.+]] = arith.addi %[[K]], %[[C2]] : index
// CHECK:             %[[R_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[I]], %[[R_OFFSET]]] [1, %[[C2]]] [1, 1]
// CHECK:             %[[R_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[I]], %[[R_OFFSET]]] [1, %[[C2]]] [1, 1]
// CHECK:             linalg.generic
// CHECK-SAME:          indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:          iterator_types = ["parallel", "parallel"]
// CHECK-SAME:          outs(%[[L_REAL_SLICE]], %[[L_IMAG_SLICE]], %[[R_REAL_SLICE]], %[[R_IMAG_SLICE]]
//
//                    The computation is basically the same, and they are
//                    checked above. Here only checks the different part.
// CHECK:             %{{.+}} = linalg.index 1 : index

// -----

func.func @fft_2D_coef_buf(%real: memref<?x16xf32>, %imag: memref<?x16xf32>,
                      %coef_real: memref<1xf32>, %coef_imag: memref<1xf32>) {
  %stage = arith.constant 1 : index
  iree_linalg_ext.fft
    ins(%stage, %coef_real, %coef_imag: index, memref<1xf32>, memref<1xf32>)
    outs(%real, %imag: memref<?x16xf32>, memref<?x16xf32>)
  return
}
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:       func.func @fft_2D_coef_buf
// CHECK-SAME:    %[[REAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[IMAG:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[COEF_REAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[COEF_IMAG:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[D0:.+]] = memref.dim %[[REAL]], %[[C0]] : memref<?x16xf32>
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[D0]] step %[[C1]]
// CHECK:           scf.for %[[K:.+]] = %[[C0]] to %[[C16]] step %[[C2]]
// CHECK:             %[[L_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[I]], %[[K]]] [1, %[[C1]]] [1, 1]
// CHECK:             %[[L_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[I]], %[[K]]] [1, %[[C1]]] [1, 1]
// CHECK:             %[[R_OFFSET:.+]] = arith.addi %[[K]], %[[C1]] : index
// CHECK:             %[[R_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[I]], %[[R_OFFSET]]] [1, %[[C1]]] [1, 1]
// CHECK:             %[[R_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[I]], %[[R_OFFSET]]] [1, %[[C1]]] [1, 1]
// CHECK:             linalg.generic
// CHECK-SAME:          indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP2]], #[[MAP2]], #[[MAP2]], #[[MAP2]]]
// CHECK-SAME:          iterator_types = ["parallel", "parallel"]
// CHECK-SAME:          ins(%[[COEF_REAL]], %[[COEF_IMAG]]
// CHECK-SAME:          outs(%[[L_REAL_SLICE]], %[[L_IMAG_SLICE]], %[[R_REAL_SLICE]], %[[R_IMAG_SLICE]]
// CHECK:             ^bb0(%[[W_REAL:.+]]: f32, %[[W_IMAG:.+]]: f32, %[[L_REAL:.+]]: f32, %[[L_IMAG:.+]]: f32, %[[R_REAL:.+]]: f32, %[[R_IMAG:.+]]: f32)
//                      Compute "t = w * a[k + j + mh]" by expanding
//                        (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
// CHECK-DAG:           %[[XU:.+]] = arith.mulf %[[W_REAL]], %[[R_REAL]]
// CHECK-DAG:           %[[YV:.+]] = arith.mulf %[[W_IMAG]], %[[R_IMAG]]
// CHECK-DAG:           %[[XV:.+]] = arith.mulf %[[W_REAL]], %[[R_IMAG]]
// CHECK-DAG:           %[[YU:.+]] = arith.mulf %[[W_IMAG]], %[[R_REAL]]
// CHECK:               %[[T_REAL:.+]] = arith.subf %[[XU]], %[[YV]]
// CHECK:               %[[T_IMAG:.+]] = arith.addf %[[XV]], %[[YU]]
//
//                      Compute the results.
//                        u = a[k + j];
//                        a[k + j] = u + t;
//                        a[k + j + mh] = u - t;
// CHECK:               %[[RES1:.+]] = arith.addf %[[L_REAL]], %[[T_REAL]]
// CHECK:               %[[RES2:.+]] = arith.addf %[[L_IMAG]], %[[T_IMAG]]
// CHECK:               %[[RES3:.+]] = arith.subf %[[L_REAL]], %[[T_REAL]]
// CHECK:               %[[RES4:.+]] = arith.subf %[[L_IMAG]], %[[T_IMAG]]
// CHECK:               linalg.yield %[[RES1]], %[[RES2]], %[[RES3]], %[[RES4]]

// -----

func.func @scan_1d_inclusive(%0: memref<128xi32>, %1: memref<128xi32>) {
  %c0 = memref.alloc() : memref<i32>
  iree_linalg_ext.scan dimension(0) inclusive(true)
    ins(%0 : memref<128xi32>) outs(%1, %c0 : memref<128xi32>, memref<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  }
  return
}
// CHECK-LABEL: func.func @scan_1d_inclusive
// CHECK-SAME:    %[[BUFI:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[BUFO:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[ACC:.+]] = memref.alloc() : memref<i32>
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK:           %[[COND:.+]] = arith.cmpi eq, %[[ARG1]], %[[C0]] : index
// CHECK:           scf.if %[[COND]] {
// CHECK:             %[[V1:.+]] = memref.load %[[BUFI]][%[[ARG1]]]
// CHECK:             memref.store %[[V1]], %[[BUFO]][%[[ARG1]]]
// CHECK:           } else {
// CHECK:             %[[T1:.+]] = arith.subi %[[ARG1]], %[[C1]] : index
// CHECK:             %[[V2:.+]] = memref.load %[[BUFO]][%[[T1]]]
// CHECK:             %[[V3:.+]] = memref.load %[[BUFI]][%[[ARG1]]]
// CHECK:             %[[V4:.+]] = arith.addi %[[V2]], %[[V3]] : i32
// CHECK:             memref.store %[[V4]], %[[BUFO]][%[[ARG1]]]
// CHECK:             memref.store %[[V4]], %[[ACC]][]
// CHECK:           }

// -----

func.func @scan_1d_exclusive(%0: memref<128xi32>, %1: memref<128xi32>) {
  %c0 = memref.alloc() : memref<i32>
  iree_linalg_ext.scan dimension(0) inclusive(false)
    ins(%0 : memref<128xi32>) outs(%1, %c0 : memref<128xi32>, memref<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  }
  return
}
// CHECK-LABEL: func.func @scan_1d_exclusive
// CHECK-SAME:    %[[BUFI:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[BUFO:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[ACC:.+]] = memref.alloc() : memref<i32>
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK:           %[[COND:.+]] = arith.cmpi eq, %[[ARG1]], %[[C0]] : index
// CHECK:           scf.if %[[COND]] {
// CHECK:             %[[V0:.+]] = memref.load %[[ACC]][] : memref<i32>
// CHECK:             memref.store %[[V0]], %[[BUFO]][%[[ARG1]]]
// CHECK:           } else {
// CHECK:             %[[T1:.+]] = arith.subi %[[ARG1]], %[[C1]] : index
// CHECK:             %[[V2:.+]] = memref.load %[[BUFO]][%[[T1]]]
// CHECK:             %[[V3:.+]] = memref.load %[[BUFI]][%[[T1]]]
// CHECK:             %[[V4:.+]] = arith.addi %[[V2]], %[[V3]] : i32
// CHECK:             memref.store %[[V4]], %[[BUFO]][%[[ARG1]]]
// CHECK:             memref.store %[[V4]], %[[ACC]][]
// CHECK:           }

// -----

func.func @scan_2d(%0: memref<16x32xi32>, %1: memref<16x32xi32>) {
  %t0 = memref.alloc() : memref<32xi32>
  iree_linalg_ext.scan dimension(0) inclusive(true)
    ins(%0 : memref<16x32xi32>) outs(%1, %t0 : memref<16x32xi32>, memref<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  }
  return
}
// CHECK-LABEL: func.func @scan_2d
// CHECK-SAME:    %[[BUFI:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[BUFO:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[ACC:.+]] = memref.alloc() : memref<32xi32>
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C16]] step %[[C1]]
// CHECK:           scf.for %[[ARG2:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK:             %[[COND:.+]] = arith.cmpi eq, %[[ARG1]], %[[C0]] : index
// CHECK:             scf.if %[[COND]] {
// CHECK:               %[[V1:.+]] = memref.load %[[BUFI]][%[[ARG1]], %[[ARG2]]]
// CHECK:               memref.store %[[V1]], %[[BUFO]][%[[ARG1]], %[[ARG2]]]
// CHECK:             } else {
// CHECK:               %[[T1:.+]] = arith.subi %[[ARG1]], %[[C1]] : index
// CHECK:               %[[V2:.+]] = memref.load %[[BUFO]][%[[T1]], %[[ARG2]]]
// CHECK:               %[[V3:.+]] = memref.load %[[BUFI]][%[[ARG1]], %[[ARG2]]]
// CHECK:               %[[V4:.+]] = arith.addi %[[V2]], %[[V3]] : i32
// CHECK:               memref.store %[[V4]], %[[BUFO]][%[[ARG1]], %[[ARG2]]]
// CHECK:               memref.store %[[V4]], %[[ACC]][%[[ARG2]]]
// CHECK:             }

// -----

func.func @topk_memref(%input_values: memref<2x10xf32>, %input_indices: memref<2x10xi32>, %out_values: memref<2x3xf32>, %out_indices: memref<2x3xi32>) {
  iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : memref<2x10xf32> , memref<2x10xi32>)
        outs(%out_values, %out_indices : memref<2x3xf32>, memref<2x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        }
  return
}

// CHECK-LABEL: func.func @topk_memref
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         scf.for %[[ARG4:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:           scf.for %[[ARG5:.+]] = %[[C0]] to %[[C10]] step %[[C1]]
// CHECK:             %[[D0:.+]] = memref.load %[[ARG0]][%[[ARG4]], %[[ARG5]]]
// CHECK:             %[[D1:.+]] = memref.load %[[ARG1]][%[[ARG4]], %[[ARG5]]]
// CHECK:             %[[D2:.+]]:2 = scf.for %[[ARG6:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG7:.+]] = %[[D0]], %[[ARG8:.+]] = %[[D1]])
// CHECK:               %[[D3:.+]] = memref.load %[[ARG2]][%[[ARG4]], %[[ARG6]]]
// CHECK:               %[[D4:.+]] = memref.load %[[ARG3]][%[[ARG4]], %[[ARG6]]]
// CHECK:               %[[D5:.+]] = arith.cmpf ogt, %[[ARG7]], %[[D3]] : f32
// CHECK:               %[[D6:.+]] = arith.cmpf ogt, %[[D3]], %[[ARG7]] : f32
// CHECK:               %[[D7:.+]] = arith.cmpi eq, %[[D5]], %[[D6]] : i1
// CHECK:               %[[D8:.+]] = arith.cmpi slt, %[[ARG8]], %[[D4]] : i32
// CHECK:               %[[D9:.+]] = arith.andi %[[D7]], %[[D8]] : i1
// CHECK:               %[[D10:.+]] = arith.ori %[[D5]], %[[D9]] : i1
// CHECK:               %[[D11:.+]] = arith.select %[[D5]], %[[ARG7]], %[[D3]] : f32
// CHECK:               %[[D12:.+]] = arith.select %[[D10]], %[[ARG8]], %[[D4]] : i32
// CHECK:               memref.store %[[D11]], %[[ARG2]][%[[ARG4]], %[[ARG6]]]
// CHECK:               memref.store %[[D12]], %[[ARG3]][%[[ARG4]], %[[ARG6]]]
// CHECK:               %[[D13:.+]] = arith.select %[[D5]], %[[D3]], %[[ARG7]] : f32
// CHECK:               %[[D14:.+]] = arith.select %[[D10]], %[[D4]], %[[ARG8]] : i32
// CHECK:               scf.yield %[[D13]], %[[D14]] : f32, i32

// -----

func.func @topk_memref_dynamic(%input_values: memref<?x?xf32>, %input_indices: memref<?x?xi32>, %out_values: memref<?x3xf32>, %out_indices: memref<?x3xi32>) {
  iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : memref<?x?xf32> , memref<?x?xi32>)
        outs(%out_values, %out_indices : memref<?x3xf32>, memref<?x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        }
  return
}

// CHECK-LABEL: func.func @topk_memref_dynamic
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[D0:.+]] = memref.dim %[[ARG0:.+]], %[[C0]]
// CHECK:         %[[D1:.+]] = memref.dim %[[ARG0:.+]], %[[C1]]
// CHECK:         scf.for %[[ARG4:.+]] = %[[C0]] to %[[D0]] step %[[C1]]
// CHECK:           scf.for %[[ARG5:.+]] = %[[C0]] to %[[D1]] step %[[C1]]
// CHECK:             %[[D2:.+]] = memref.load %[[ARG0]][%[[ARG4]], %[[ARG5]]]
// CHECK:             %[[D3:.+]] = memref.load %[[ARG1]][%[[ARG4]], %[[ARG5]]]
// CHECK:             %[[D4:.+]]:2 = scf.for %[[ARG6:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG7:.+]] = %[[D2]], %[[ARG8:.+]] = %[[D3]])
// CHECK:               %[[D5:.+]] = memref.load %[[ARG2]][%[[ARG4]], %[[ARG6]]]
// CHECK:               %[[D6:.+]] = memref.load %[[ARG3]][%[[ARG4]], %[[ARG6]]]
// CHECK:               %[[D7:.+]] = arith.cmpf ogt, %[[ARG7]], %[[D5]] : f32
// CHECK:               %[[D8:.+]] = arith.cmpf ogt, %[[D5]], %[[ARG7]] : f32
// CHECK:               %[[D9:.+]] = arith.cmpi eq, %[[D7]], %[[D8]] : i1
// CHECK:               %[[D10:.+]] = arith.cmpi slt, %[[ARG8]], %[[D6]] : i32
// CHECK:               %[[D11:.+]] = arith.andi %[[D9]], %[[D10]] : i1
// CHECK:               %[[D12:.+]] = arith.ori %[[D7]], %[[D11]] : i1
// CHECK:               %[[D13:.+]] = arith.select %[[D7]], %[[ARG7]], %[[D5]] : f32
// CHECK:               %[[D14:.+]] = arith.select %[[D12]], %[[ARG8]], %[[D6]] : i32
// CHECK:               memref.store %[[D13]], %[[ARG2]][%[[ARG4]], %[[ARG6]]]
// CHECK:               memref.store %[[D14]], %[[ARG3]][%[[ARG4]], %[[ARG6]]]
// CHECK:               %[[D15:.+]] = arith.select %[[D7]], %[[D5]], %[[ARG7]] : f32
// CHECK:               %[[D16:.+]] = arith.select %[[D12]], %[[D6]], %[[ARG8]] : i32
// CHECK:               scf.yield %[[D15]], %[[D16]] : f32, i32

// -----

func.func @topk_memref_optional(%input_values: memref<2x10xf32>, %out_values: memref<2x3xf32>, %out_indices: memref<2x3xi32>) {
  iree_linalg_ext.topk
        dimension(1)
        ins(%input_values : memref<2x10xf32>)
        outs(%out_values, %out_indices : memref<2x3xf32>, memref<2x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        }
  return
}

// CHECK-LABEL: func.func @topk_memref
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         scf.for %[[ARG3:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:           scf.for %[[ARG4:.+]] = %[[C0]] to %[[C10]] step %[[C1]]
// CHECK:             %[[D0:.+]] = memref.load %[[ARG0]][%[[ARG3]], %[[ARG4]]]
// CHECK:             %[[D1:.+]] = arith.index_cast %[[ARG4]] : index to i32
// CHECK:             %[[D2:.+]]:2 = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[D0]], %[[ARG7:.+]] = %[[D1]])
// CHECK:               %[[D3:.+]] = memref.load %[[ARG1]][%[[ARG3]], %[[ARG5]]]
// CHECK:               %[[D4:.+]] = memref.load %[[ARG2]][%[[ARG3]], %[[ARG5]]]
// CHECK:               %[[D5:.+]] = arith.cmpf ogt, %[[ARG6]], %[[D3]] : f32
// CHECK:               %[[D6:.+]] = arith.cmpf ogt, %[[D3]], %[[ARG6]] : f32
// CHECK:               %[[D7:.+]] = arith.cmpi eq, %[[D5]], %[[D6]] : i1
// CHECK:               %[[D8:.+]] = arith.cmpi slt, %[[ARG7]], %[[D4]] : i32
// CHECK:               %[[D9:.+]] = arith.andi %[[D7]], %[[D8]] : i1
// CHECK:               %[[D10:.+]] = arith.ori %[[D5]], %[[D9]] : i1
// CHECK:               %[[D11:.+]] = arith.select %[[D5]], %[[ARG6]], %[[D3]] : f32
// CHECK:               %[[D12:.+]] = arith.select %[[D10]], %[[ARG7]], %[[D4]] : i32
// CHECK:               memref.store %[[D11]], %[[ARG1]][%[[ARG3]], %[[ARG5]]]
// CHECK:               memref.store %[[D12]], %[[ARG2]][%[[ARG3]], %[[ARG5]]]
// CHECK:               %[[D13:.+]] = arith.select %[[D5]], %[[D3]], %[[ARG6]] : f32
// CHECK:               %[[D14:.+]] = arith.select %[[D10]], %[[D4]], %[[ARG7]] : i32
// CHECK:               scf.yield %[[D13]], %[[D14]] : f32, i32

// -----

func.func @arg_compare_memref(
    %input_values: memref<2x10xf32>,
    %out_values: memref<2xf32>,
    %out_indices: memref<2xi32>
) {
  iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input_values : memref<2x10xf32>)
    outs(%out_values, %out_indices : memref<2xf32>, memref<2xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  }
  return
}

// CHECK-LABEL: func.func @arg_compare_memref
// CHECK-SAME: %[[INPUT:.+]]: memref<2x10xf32>
// CHECK-SAME: %[[OUTVAL:.+]]: memref<2xf32>
// CHECK-SAME: %[[OUTIDX:.+]]: memref<2xi32>

// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C10:.+]] = arith.constant 10 : index

// CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:   scf.for %[[J:.+]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:     %[[V0:.+]] = memref.load %[[OUTVAL]][%[[I]]] : memref<2xf32>
// CHECK:     %[[I0:.+]] = memref.load %[[OUTIDX]][%[[I]]] : memref<2xi32>
// CHECK:     %[[V1:.+]] = memref.load %[[INPUT]][%[[I]], %[[J]]] : memref<2x10xf32>
// CHECK:     %[[CMP:.+]] = arith.cmpf ogt, %[[V1]], %[[V0]] : f32
// CHECK:     %[[VAL_SEL:.+]] = arith.select %[[CMP]], %[[V1]], %[[V0]] : f32
// CHECK:     %[[IDX_CAST:.+]] = arith.index_cast %[[J]] : index to i32
// CHECK:     %[[IDX_SEL:.+]] = arith.select %[[CMP]], %[[IDX_CAST]], %[[I0]] : i32
// CHECK:     memref.store %[[VAL_SEL]], %[[OUTVAL]][%[[I]]] : memref<2xf32>
// CHECK:     memref.store %[[IDX_SEL]], %[[OUTIDX]][%[[I]]] : memref<2xi32>

// -----

func.func @arg_compare_memref_dynamic(
    %input_values: memref<?x?xf32>,
    %out_values: memref<?xf32>,
    %out_indices: memref<?xi32>
) {
  iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input_values : memref<?x?xf32>)
    outs(%out_values, %out_indices : memref<?xf32>, memref<?xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  }
  return
}

// CHECK-LABEL: func.func @arg_compare_memref_dynamic
// CHECK-SAME: %[[INPUT:.+]]: memref<?x?xf32>
// CHECK-SAME: %[[OUTVAL:.+]]: memref<?xf32>
// CHECK-SAME: %[[OUTIDX:.+]]: memref<?xi32>

// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[DIM0:.+]] = memref.dim %[[INPUT]], %[[C0]] : memref<?x?xf32>
// CHECK-DAG: %[[DIM1:.+]] = memref.dim %[[INPUT]], %[[C1]] : memref<?x?xf32>

// CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[DIM0]] step %[[C1]] {
// CHECK:   scf.for %[[J:.+]] = %[[C0]] to %[[DIM1]] step %[[C1]] {
// CHECK:     %[[V0:.+]] = memref.load %[[OUTVAL]][%[[I]]] : memref<?xf32>
// CHECK:     %[[I0:.+]] = memref.load %[[OUTIDX]][%[[I]]] : memref<?xi32>
// CHECK:     %[[V1:.+]] = memref.load %[[INPUT]][%[[I]], %[[J]]] : memref<?x?xf32>
// CHECK:     %[[CMP:.+]] = arith.cmpf ogt, %[[V1]], %[[V0]] : f32
// CHECK:     %[[VAL_SEL:.+]] = arith.select %[[CMP]], %[[V1]], %[[V0]] : f32
// CHECK:     %[[IDX_CAST:.+]] = arith.index_cast %[[J]] : index to i32
// CHECK:     %[[IDX_SEL:.+]] = arith.select %[[CMP]], %[[IDX_CAST]], %[[I0]] : i32
// CHECK:     memref.store %[[VAL_SEL]], %[[OUTVAL]][%[[I]]] : memref<?xf32>
// CHECK:     memref.store %[[IDX_SEL]], %[[OUTIDX]][%[[I]]] : memref<?xi32>

// -----

func.func @arg_compare_memref_with_base(
    %input_values: memref<2x10xf32>,
    %index_base: index,
    %out_values: memref<2xf32>,
    %out_indices: memref<2xi32>
) {
  iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input_values : memref<2x10xf32>)
    outs(%out_values, %out_indices : memref<2xf32>, memref<2xi32>)
    index_base(%index_base : index) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  }
  return
}

// CHECK-LABEL: func.func @arg_compare_memref_with_base
// CHECK-SAME: %[[INPUT:.+]]: memref<2x10xf32>
// CHECK-SAME: %[[INDEX_BASE:.+]]: index
// CHECK-SAME: %[[OUT_VAL:.+]]: memref<2xf32>
// CHECK-SAME: %[[OUT_IDX:.+]]: memref<2xi32>

// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C10:.+]] = arith.constant 10 : index

// CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:   scf.for %[[J:.+]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:     %[[INIT_VAL:.+]] = memref.load %[[OUT_VAL]][%[[I]]] : memref<2xf32>
// CHECK:     %[[INIT_IDX:.+]] = memref.load %[[OUT_IDX]][%[[I]]] : memref<2xi32>
// CHECK:     %[[CAND_VAL:.+]] = memref.load %[[INPUT]][%[[I]], %[[J]]] : memref<2x10xf32>
// CHECK:     %[[CMP:.+]] = arith.cmpf ogt, %[[CAND_VAL]], %[[INIT_VAL]] : f32
// CHECK:     %[[SELECT_VAL:.+]] = arith.select %[[CMP]], %[[CAND_VAL]], %[[INIT_VAL]] : f32
// CHECK:     %[[OFFSET_IDX:.+]] = arith.addi %[[INDEX_BASE]], %[[J]] : index
// CHECK:     %[[OFFSET_I32:.+]] = arith.index_cast %[[OFFSET_IDX]] : index to i32
// CHECK:     %[[SELECT_IDX:.+]] = arith.select %[[CMP]], %[[OFFSET_I32]], %[[INIT_IDX]] : i32
// CHECK:     memref.store %[[SELECT_VAL]], %[[OUT_VAL]][%[[I]]] : memref<2xf32>
// CHECK:     memref.store %[[SELECT_IDX]], %[[OUT_IDX]][%[[I]]] : memref<2xi32>

// -----

func.func @arg_compare_reduce_dim0(
    %input_values: memref<2x10xf32>,
    %out_values: memref<10xf32>,
    %out_indices: memref<10xi32>
) {
  iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input_values : memref<2x10xf32>)
    outs(%out_values, %out_indices : memref<10xf32>, memref<10xi32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  }
  return
}

// CHECK-LABEL: func.func @arg_compare_reduce_dim0
// CHECK-SAME: %[[INPUT:.+]]: memref<2x10xf32>
// CHECK-SAME: %[[OUTVAL:.+]]: memref<10xf32>
// CHECK-SAME: %[[OUTIDX:.+]]: memref<10xi32>

// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C10:.+]] = arith.constant 10 : index

// CHECK: scf.for %[[J:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:   scf.for %[[I:.+]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:     %[[V0:.+]] = memref.load %[[OUTVAL]][%[[I]]] : memref<10xf32>
// CHECK:     %[[I0:.+]] = memref.load %[[OUTIDX]][%[[I]]] : memref<10xi32>
// CHECK:     %[[V1:.+]] = memref.load %[[INPUT]][%[[J]], %[[I]]] : memref<2x10xf32>
// CHECK:     %[[CMP:.+]] = arith.cmpf ogt, %[[V1]], %[[V0]] : f32
// CHECK:     %[[VAL_SEL:.+]] = arith.select %[[CMP]], %[[V1]], %[[V0]] : f32
// CHECK:     %[[IDX_CAST:.+]] = arith.index_cast %[[J]] : index to i32
// CHECK:     %[[IDX_SEL:.+]] = arith.select %[[CMP]], %[[IDX_CAST]], %[[I0]] : i32
// CHECK:     memref.store %[[VAL_SEL]], %[[OUTVAL]][%[[I]]] : memref<10xf32>
// CHECK:     memref.store %[[IDX_SEL]], %[[OUTIDX]][%[[I]]] : memref<10xi32>

// -----

func.func @arg_compare_explicit_index_memref(
    %input_values: memref<2x10xf32>,
    %input_indices: memref<2x10xi32>,
    %out_values: memref<2xf32>,
    %out_indices: memref<2xi32>
) {
  iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input_values, %input_indices : memref<2x10xf32>, memref<2x10xi32>)
    outs(%out_values, %out_indices : memref<2xf32>, memref<2xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  }
  return
}

// CHECK-LABEL: func.func @arg_compare_explicit_index_memref
// CHECK-SAME: %[[INPUT_VAL:.+]]: memref<2x10xf32>
// CHECK-SAME: %[[INPUT_IDX:.+]]: memref<2x10xi32>
// CHECK-SAME: %[[OUTVAL:.+]]: memref<2xf32>
// CHECK-SAME: %[[OUTIDX:.+]]: memref<2xi32>

// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C10:.+]] = arith.constant 10 : index

// CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:   scf.for %[[J:.+]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:     %[[V0:.+]] = memref.load %[[OUTVAL]][%[[I]]] : memref<2xf32>
// CHECK:     %[[I0:.+]] = memref.load %[[OUTIDX]][%[[I]]] : memref<2xi32>
// CHECK:     %[[V1:.+]] = memref.load %[[INPUT_VAL]][%[[I]], %[[J]]] : memref<2x10xf32>
// CHECK:     %[[CMP:.+]] = arith.cmpf ogt, %[[V1]], %[[V0]] : f32
// CHECK:     %[[VAL_SEL:.+]] = arith.select %[[CMP]], %[[V1]], %[[V0]] : f32
// CHECK:     %[[I1:.+]] = memref.load %[[INPUT_IDX]][%[[I]], %[[J]]] : memref<2x10xi32>
// CHECK:     %[[IDX_SEL:.+]] = arith.select %[[CMP]], %[[I1]], %[[I0]] : i32
// CHECK:     memref.store %[[VAL_SEL]], %[[OUTVAL]][%[[I]]] : memref<2xf32>
// CHECK:     memref.store %[[IDX_SEL]], %[[OUTIDX]][%[[I]]] : memref<2xi32>

// -----

func.func @arg_compare_explicit_index_memref_dynamic(
    %input_values: memref<?x?xf32>,
    %input_indices: memref<?x?xi32>,
    %out_values: memref<?xf32>,
    %out_indices: memref<?xi32>
) {
  iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input_values, %input_indices : memref<?x?xf32>, memref<?x?xi32>)
    outs(%out_values, %out_indices : memref<?xf32>, memref<?xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  }
  return
}

// CHECK-LABEL: func.func @arg_compare_explicit_index_memref_dynamic
// CHECK-SAME: %[[INPUT_VAL:.+]]: memref<?x?xf32>
// CHECK-SAME: %[[INPUT_IDX:.+]]: memref<?x?xi32>
// CHECK-SAME: %[[OUTVAL:.+]]: memref<?xf32>
// CHECK-SAME: %[[OUTIDX:.+]]: memref<?xi32>

// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[DIM0:.+]] = memref.dim %[[INPUT_VAL]], %[[C0]] : memref<?x?xf32>
// CHECK-DAG: %[[DIM1:.+]] = memref.dim %[[INPUT_VAL]], %[[C1]] : memref<?x?xf32>

// CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[DIM0]] step %[[C1]] {
// CHECK:   scf.for %[[J:.+]] = %[[C0]] to %[[DIM1]] step %[[C1]] {
// CHECK:     %[[V0:.+]] = memref.load %[[OUTVAL]][%[[I]]] : memref<?xf32>
// CHECK:     %[[I0:.+]] = memref.load %[[OUTIDX]][%[[I]]] : memref<?xi32>
// CHECK:     %[[V1:.+]] = memref.load %[[INPUT_VAL]][%[[I]], %[[J]]] : memref<?x?xf32>
// CHECK:     %[[CMP:.+]] = arith.cmpf ogt, %[[V1]], %[[V0]] : f32
// CHECK:     %[[VAL_SEL:.+]] = arith.select %[[CMP]], %[[V1]], %[[V0]] : f32
// CHECK:     %[[I1:.+]] = memref.load %[[INPUT_IDX]][%[[I]], %[[J]]] : memref<?x?xi32>
// CHECK:     %[[IDX_SEL:.+]] = arith.select %[[CMP]], %[[I1]], %[[I0]] : i32
// CHECK:     memref.store %[[VAL_SEL]], %[[OUTVAL]][%[[I]]] : memref<?xf32>
// CHECK:     memref.store %[[IDX_SEL]], %[[OUTIDX]][%[[I]]] : memref<?xi32>

// -----

func.func @gather_1d_indices(%arg0 : memref<10x10xi32>, %arg1 : memref<1xi32>, %arg2 : memref<1x10xi32>) {
  iree_linalg_ext.gather
    dimension_map = [0]
    ins(%arg0, %arg1: memref<10x10xi32>, memref<1xi32>)
    outs(%arg2: memref<1x10xi32>)
  return
}
// CHECK-LABEL: func @gather_1d_indices
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK:           scf.for %[[J:.+]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:             %[[IDX:.+]] = memref.load %[[ARG1]][%[[I]]] : memref<1xi32>
// CHECK:             %[[CAST:.+]] = arith.index_cast %[[IDX]] : i32 to index
// CHECK:             %[[LOAD:.+]] = memref.load %[[ARG0]][%[[CAST]], %[[J]]] : memref<10x10xi32>

// -----

func.func @gather_1d_indices_masked(%arg0 : memref<10x10xi32>, %arg1 : memref<1xi32>, %arg2 : memref<1xi1>, %arg3 : memref<1x10xi32>) {
  iree_linalg_ext.gather
    dimension_map = [0]
    ins(%arg0, %arg1, %arg2 : memref<10x10xi32>, memref<1xi32>, memref<1xi1>)
    outs(%arg3: memref<1x10xi32>)
  return
}
// CHECK-LABEL: func @gather_1d_indices_masked
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK:           scf.for %[[J:.+]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:             %[[IDX:.+]] = memref.load %[[ARG1]][%[[I]]] : memref<1xi32>
// CHECK:             %[[CAST:.+]] = arith.index_cast %[[IDX]] : i32 to index
// CHECK:             %[[MASK_VAL:.+]] = memref.load %[[ARG2]][%[[I]]] : memref<1xi1>
// CHECK:             scf.if %[[MASK_VAL]] {
// CHECK:               %[[LOAD:.+]] = memref.load %[[ARG0]][%[[CAST]], %[[J]]] : memref<10x10xi32>
// CHECK:               memref.store %[[LOAD]], %[[ARG3]][%[[I]], %[[J]]] : memref<1x10xi32>

// -----

func.func @gather_1d_indices_masked_i8(%arg0 : memref<10x10xi32>, %arg1 : memref<1xi32>, %arg2 : memref<1xi8>, %arg3 : memref<1x10xi32>) {
  iree_linalg_ext.gather
    dimension_map = [0]
    ins(%arg0, %arg1, %arg2 : memref<10x10xi32>, memref<1xi32>, memref<1xi8>)
    outs(%arg3: memref<1x10xi32>)
  return
}
// CHECK-LABEL: func @gather_1d_indices_masked_i8
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             %[[MASK_VAL:.+]] = memref.load %{{.+}}[%{{.+}}] : memref<1xi8>
// CHECK:             %[[MASK_I1:.+]] = arith.trunci %[[MASK_VAL]] : i8 to i1
// CHECK:             scf.if %[[MASK_I1]] {

// -----

func.func @gather_2d_indices(%arg0 : memref<2x2xi32>, %arg1 : memref<2x2xi32>, %arg2 : memref<2xi32>) {
  iree_linalg_ext.gather
    dimension_map = [0, 1]
    ins(%arg0, %arg1: memref<2x2xi32>, memref<2x2xi32>)
    outs(%arg2: memref<2xi32>)
  return
}
// CHECK-LABEL: func @gather_2d_indices
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:           %[[IDX0:.+]] = memref.load %[[ARG1]][%[[I]], %[[C0]]] : memref<2x2xi32>
// CHECK:           %[[CAST0:.+]] = arith.index_cast %[[IDX0]] : i32 to index
// CHECK:           %[[IDX1:.+]] = memref.load %[[ARG1]][%[[I]], %[[C1]]] : memref<2x2xi32>
// CHECK:           %[[CAST1:.+]] = arith.index_cast %[[IDX1]] : i32 to index
// CHECK:           %[[LOAD0:.+]] = memref.load %[[ARG0]][%[[CAST0]], %[[CAST1]]] : memref<2x2xi32>
// CHECK:           memref.store %[[LOAD0]], %[[ARG2]][%[[I]]] : memref<2xi32>

// -----

func.func @gather_perm_dim_map(%arg0 : memref<2x2xi32>, %arg1 : memref<2x2xi32>, %arg2 : memref<2xi32>) {
  iree_linalg_ext.gather
    dimension_map = [1, 0]
    ins(%arg0, %arg1: memref<2x2xi32>, memref<2x2xi32>)
    outs(%arg2: memref<2xi32>)
  return
}
// CHECK-LABEL: func @gather_perm_dim_map
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:           %[[IDX0:.+]] = memref.load %[[ARG1]][%[[I]], %[[C0]]] : memref<2x2xi32>
// CHECK:           %[[CAST0:.+]] = arith.index_cast %[[IDX0]] : i32 to index
// CHECK:           %[[IDX1:.+]] = memref.load %[[ARG1]][%[[I]], %[[C1]]] : memref<2x2xi32>
// CHECK:           %[[CAST1:.+]] = arith.index_cast %[[IDX1]] : i32 to index
// CHECK:           %[[LOAD0:.+]] = memref.load %[[ARG0]][%[[CAST1]], %[[CAST0]]] : memref<2x2xi32>
// CHECK:           memref.store %[[LOAD0]], %[[ARG2]][%[[I]]] : memref<2xi32>

// -----

func.func @gather_inline_region(%arg0 : memref<2x2xi32>, %arg1 : memref<2x2xi32>, %arg2 : memref<2xi32>) {
  %cst = arith.constant 3 : i32
  iree_linalg_ext.gather
    dimension_map = [0, 1]
    ins(%arg0, %arg1: memref<2x2xi32>, memref<2x2xi32>)
    outs(%arg2: memref<2xi32>)
  return
}
// CHECK-LABEL: func @gather_inline_region
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:           %[[IDX0:.+]] = memref.load %[[ARG1]][%[[I]], %[[C0]]] : memref<2x2xi32>
// CHECK:           %[[CAST0:.+]] = arith.index_cast %[[IDX0]] : i32 to index
// CHECK:           %[[IDX1:.+]] = memref.load %[[ARG1]][%[[I]], %[[C1]]] : memref<2x2xi32>
// CHECK:           %[[CAST1:.+]] = arith.index_cast %[[IDX1]] : i32 to index
// CHECK:           %[[LOAD0:.+]] = memref.load %[[ARG0]][%[[CAST0]], %[[CAST1]]] : memref<2x2xi32>
// CHECK:           memref.store %[[LOAD0]], %[[ARG2]][%[[I]]] : memref<2xi32>

// -----

func.func @map_store_memref(
  %input: memref<?xf32>, %output: memref<?x?xf32>, %bound: index
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = memref.dim %output, %c0 : memref<?x?xf32>
  %dim1 = memref.dim %output, %c1 : memref<?x?xf32>
  iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.cmpi uge, %idx0, %bound : index
      %out_idx:2 = affine.delinearize_index %idx0 into (%dim0, %dim1) : index, index
      iree_linalg_ext.yield %out_idx#0, %out_idx#1, %mask : index, index, i1
  } : memref<?xf32> into memref<?x?xf32>
  return
}
//      CHECK: func @map_store_memref
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[BOUND:[a-zA-Z0-9]+]]
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[IN_D0:.+]] = memref.dim %[[INPUT]], %[[C0]]
//  CHECK-DAG:   %[[OUT_D0:.+]] = memref.dim %[[OUTPUT]], %[[C0]]
//  CHECK-DAG:   %[[OUT_D1:.+]] = memref.dim %[[OUTPUT]], %[[C1]]
//      CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[IN_D0]] step %[[C1]]
//  CHECK-DAG:     %[[MASK:.+]] = arith.cmpi uge, %[[IV]], %[[BOUND]] : index
//  CHECK-DAG:     %[[OUT_IDX:.+]]:2 = affine.delinearize_index %[[IV]] into (%[[OUT_D0]], %[[OUT_D1]]) : index, index
//      CHECK:     scf.if %[[MASK]] {
// CHECK-NEXT:       %[[INPUT_ELEM:.+]] = memref.load %[[INPUT]][%[[IV]]]
// CHECK-NEXT:       memref.store %[[INPUT_ELEM]], %[[OUTPUT]]
// CHECK-SAME:         [%[[OUT_IDX]]#0, %[[OUT_IDX]]#1] : memref<?x?xf32>

// -----

func.func @map_load_memref(
  %source: memref<?x?xf32>, %output: memref<?xf32>
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = memref.dim %source, %c0 : memref<?x?xf32>
  %dim1 = memref.dim %source, %c1 : memref<?x?xf32>
  iree_linalg_ext.map_load %source into %output {
    ^bb0(%idx0: index):
      %src_idx:2 = affine.delinearize_index %idx0 into (%dim0, %dim1) : index, index
      %pad = arith.constant 0.0 : f32
      iree_linalg_ext.yield %src_idx#0, %src_idx#1, %pad : index, index, f32
  } : memref<?x?xf32> into memref<?xf32>
  return
}
//      CHECK: func @map_load_memref
// CHECK-SAME:    %[[SOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]
//  CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.{{0+}}e+00 : f32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[SRC_D0:.+]] = memref.dim %[[SOURCE]], %[[C0]]
//  CHECK-DAG:   %[[SRC_D1:.+]] = memref.dim %[[SOURCE]], %[[C1]]
//  CHECK-DAG:   %[[OUT_D0:.+]] = memref.dim %[[OUTPUT]], %[[C0]]
//      CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[OUT_D0]] step %[[C1]]
//      CHECK:     %[[SRC_IDX:.+]]:2 = affine.delinearize_index %[[IV]] into (%[[SRC_D0]], %[[SRC_D1]]) : index, index
//  CHECK-DAG:     %[[BOUND_D0:.+]] = memref.dim %[[SOURCE]], %[[C0]]
//  CHECK-DAG:     %[[GE_ZERO_0:.+]] = arith.cmpi sge, %[[SRC_IDX]]#0, %[[C0]] : index
//  CHECK-DAG:     %[[LT_DIM_0:.+]] = arith.cmpi slt, %[[SRC_IDX]]#0, %[[BOUND_D0]] : index
//  CHECK-DAG:     %[[IN_BOUNDS_0:.+]] = arith.andi %[[GE_ZERO_0]], %[[LT_DIM_0]]
//  CHECK-DAG:     %[[BOUND_D1:.+]] = memref.dim %[[SOURCE]], %[[C1]]
//  CHECK-DAG:     %[[GE_ZERO_1:.+]] = arith.cmpi sge, %[[SRC_IDX]]#1, %[[C0]] : index
//  CHECK-DAG:     %[[LT_DIM_1:.+]] = arith.cmpi slt, %[[SRC_IDX]]#1, %[[BOUND_D1]] : index
//  CHECK-DAG:     %[[IN_BOUNDS_1:.+]] = arith.andi %[[GE_ZERO_1]], %[[LT_DIM_1]]
//  CHECK-DAG:     %[[IN_BOUNDS:.+]] = arith.andi %[[IN_BOUNDS_0]], %[[IN_BOUNDS_1]]
//      CHECK:     %[[IF_RESULT:.+]] = scf.if %[[IN_BOUNDS]] -> (f32) {
//      CHECK:       %[[SOURCE_ELEM:.+]] = memref.load %[[SOURCE]]
// CHECK-SAME:         [%[[SRC_IDX]]#0, %[[SRC_IDX]]#1] : memref<?x?xf32>
//      CHECK:       scf.yield %[[SOURCE_ELEM]] : f32
//      CHECK:     } else {
//      CHECK:       scf.yield %[[PAD]] : f32
//      CHECK:     }
//      CHECK:     memref.store %[[IF_RESULT]], %[[OUTPUT]][%[[IV]]]
