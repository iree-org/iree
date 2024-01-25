// RUN: iree-dialects-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-linalg-ext-to-loops))" %s | FileCheck %s

func.func @sort_1d(%arg0: memref<128xi32>) {
  iree_linalg_ext.sort dimension(0)
    outs(%arg0 : memref<128xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
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
  ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
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
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i32, %arg5: i32):  // no predecessors
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
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
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

func.func @scatter_add_scalar_2D(
    %original: memref<4x3xi32>, %indices: memref<3x2xi32>,
    %updates: memref<3xi32>) {
  iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
    ins(%updates, %indices : memref<3xi32>, memref<3x2xi32>)
    outs(%original : memref<4x3xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
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
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
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
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
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
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
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
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
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
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
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
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
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

func.func @scatter_partial_slices(%arg0: memref<2x64x12xf32>, %arg1: memref<2x3xi32>, %arg2: memref<2x1x12xf32>) {
  iree_linalg_ext.scatter
    dimension_map = [0, 1, 2]
    unique_indices(true)
    ins(%arg2, %arg1 : memref<2x1x12xf32>, memref<2x3xi32>)
    outs(%arg0 : memref<2x64x12xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    iree_linalg_ext.yield %arg4 : f32
  }
  return
}

// CHECK-LABEL: @scatter_partial_slices
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG: %[[C0:.+]] = arith.constant
// CHECK-DAG: %[[C1:.+]] = arith.constant
// CHECK-DAG: %[[C2:.+]] = arith.constant
// CHECK-DAG: %[[C12:.+]] = arith.constant
// CHECK:     scf.for %[[ARG3:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:   scf.for %[[ARG4:.+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK-NEXT:     scf.for %[[ARG5:.+]] = %[[C0]] to %[[C12]] step %[[C1]] {
// CHECK-NEXT:       %[[LOAD0:.+]] = memref.load %[[ARG1]][%[[ARG3]], %[[C0]]] : memref<2x3xi32>
// CHECK-NEXT:       %[[CAST0:.+]] = arith.index_cast %[[LOAD0]] : i32 to index
// CHECK-NEXT:       %[[LOAD1:.+]] = memref.load %[[ARG1]][%[[ARG3]], %[[C1]]] : memref<2x3xi32>
// CHECK-NEXT:       %[[CAST1:.+]] = arith.index_cast %[[LOAD1]] : i32 to index
// CHECK-NEXT:       %[[ADD1:.+]] = arith.addi %[[CAST1]], %[[ARG4]] : index
// CHECK-NEXT:       %[[LOAD2:.+]] = memref.load %[[ARG1]][%[[ARG3]], %[[C2]]] : memref<2x3xi32>
// CHECK-NEXT:       %[[CAST2:.+]] = arith.index_cast %[[LOAD2]] : i32 to index
// CHECK-NEXT:       %[[ADD2:.+]] = arith.addi %[[CAST2]], %[[ARG5]] : index
// CHECK-NEXT:       %[[LOAD3:.+]] = memref.load %[[ARG0]][%[[CAST0]], %[[ADD1]], %[[ADD2]]] : memref<2x64x12xf32>
// CHECK-NEXT:       memref.store %[[LOAD3]], %[[ARG0]][%[[CAST0]], %[[ADD1]], %[[ADD2]]] : memref<2x64x12xf32>

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
//                    The computation is bascially the same, and they are
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

func.func @reverse_dim_0(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>) {
  iree_linalg_ext.reverse
    dimensions(dense<0> : tensor<1xi64>)
    ins(%arg0 : memref<?x?xi32>)
    outs(%arg1 : memref<?x?xi32>)
  return
}
// CHECK-LABEL: func.func @reverse_dim_0
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[D0:.+]] = memref.dim %arg0, %c0 : memref<?x?xi32>
// CHECK-DAG:     %[[D1:.+]] = memref.dim %arg0, %c1 : memref<?x?xi32>
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[D0]] step %[[C1]]
// CHECK:           scf.for %[[J:.+]] = %[[C0]] to %[[D1]] step %[[C1]]
// CHECK:             %[[T0:.+]] = memref.dim %[[IN]], %[[C0]]
// CHECK:             %[[T1:.+]] = arith.subi %[[T0]], %[[C1]] : index
// CHECK:             %[[T2:.+]] = arith.subi %[[T1]], %[[I]] : index
// CHECK:             %[[V0:.+]] = memref.load %[[IN]][%[[I]], %[[J]]]
// CHECK:             memref.store %[[V0]], %[[OUT]][%[[T2]], %[[J]]] : memref<?x?xi32>

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
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
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
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
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
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
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

func.func @NC_to_NCnc(%arg0: memref<128x256xf32>, %arg1: memref<4x8x32x32xf32>) {
  iree_linalg_ext.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1 : (memref<128x256xf32> memref<4x8x32x32xf32>)
  return
}
// CHECK:       #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-LABEL: func.func @NC_to_NCnc(
// CHECK-DAG:     %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[ubN:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[ubC:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[block:.*]] = arith.constant 32 : index
// CHECK:         scf.for %[[N:.*]] = %[[lb]] to %[[ubN]] step %[[step]] {
// CHECK:           scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[step]] {
// CHECK:             scf.for %[[n:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK:               scf.for %[[c:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK-DAG:             %[[applyMapI:.*]] = affine.apply #[[MAP]](%[[N]], %[[n]])
// CHECK-DAG:             %[[applyMapJ:.*]] = affine.apply #[[MAP]](%[[C]], %[[c]])
// CHECK:                 %[[scalar:.*]] = memref.load %arg0[%[[applyMapI]], %[[applyMapJ]]] : memref<128x256xf32>
// CHECK:                 memref.store %[[scalar]], %arg1[%[[N]], %[[C]], %[[n]], %[[c]]] : memref<4x8x32x32xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @NC_to_NCnc_pad_static(%arg0: memref<13x15xf32>, %arg1: memref<2x8x8x2xf32>, %arg2: f32) {
  iree_linalg_ext.pack %arg0 padding_value(%arg2 : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : (memref<13x15xf32> memref<2x8x8x2xf32>)
  return
}
// CHECK-DAG:   #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK-DAG:   #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-LABEL: func.func @NC_to_NCnc_pad_static(
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[C13:.*]] = arith.constant 13 : index
// CHECK-DAG:     %[[C15:.*]] = arith.constant 15 : index
// CHECK:           scf.for %[[N:.*]] = %[[C0]] to %[[C2]] step %[[step]] {
// CHECK:             scf.for %[[C:.*]] = %[[C0]] to %[[C8]] step %[[step]] {
// CHECK:               scf.for %[[n:.*]] = %[[C0]] to %[[C8]] step %[[step]] {
// CHECK:                 scf.for %[[c:.*]] = %[[C0]] to %[[C2]] step %[[step]] {
// CHECK-DAG:               %[[applyMapI:.*]] = affine.apply #[[MAP0]](%[[N]], %[[n]])
// CHECK-DAG:               %[[applyMapJ:.*]] = affine.apply #[[MAP1]](%[[C]], %[[c]])
// CHECK:                   %[[isIInBound:.*]] = arith.cmpi slt, %[[applyMapI]], %[[C13]] : index
// CHECK:                   %[[isJInBound:.*]] = arith.cmpi slt, %[[applyMapJ]], %[[C15]] : index
// CHECK:                   %[[isAllInBounds:.*]] = arith.andi %[[isIInBound]], %[[isJInBound]] : i1
// CHECK:                   %[[scalar:.*]] = scf.if %[[isAllInBounds]] -> (f32) {
// CHECK:                     %[[load:.*]] = memref.load %arg0[%[[applyMapI]], %[[applyMapJ]]] : memref<13x15xf32>
// CHECK:                     scf.yield %[[load]]
// CHECK:                   } else {
// CHECK:                     scf.yield %arg2
// CHECK:                   }
// CHECK:                   memref.store %[[scalar]], %arg1[%[[N]], %[[C]], %[[n]], %[[c]]] : memref<2x8x8x2xf32>

// -----

func.func @KC_to_KCck(%arg0: memref<128x256xf32>, %arg1: memref<4x8x32x32xf32>) {
  iree_linalg_ext.pack %arg0 inner_dims_pos = [1, 0] inner_tiles = [32, 32] into %arg1 : (memref<128x256xf32> memref<4x8x32x32xf32>)
  return
}
// CHECK:       #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-LABEL: func.func @KC_to_KCck(
// CHECK-DAG:     %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[ubK:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[ubC:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[block:.*]] = arith.constant 32 : index
// CHECK:         scf.for %[[K:.*]] = %[[lb]] to %[[ubK]] step %[[step]] {
// CHECK:           scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[step]] {
// CHECK:             scf.for %[[c:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK:               scf.for %[[k:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK-DAG:             %[[applyMapC:.*]] = affine.apply #[[MAP]](%[[C]], %[[c]])
// CHECK-DAG:             %[[applyMapK:.*]] = affine.apply #[[MAP]](%[[K]], %[[k]])
// CHECK:                 %[[scalar:.*]] = memref.load %arg0[%[[applyMapK]], %[[applyMapC]]] : memref<128x256xf32>
// CHECK:                 memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[c]], %[[k]]] : memref<4x8x32x32xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

// This should be a simple expand shape.
func.func @KC_to_KCc(%arg0: memref<128x256xf32>, %arg1: memref<128x8x32xf32>) {
  iree_linalg_ext.pack %arg0 inner_dims_pos = [1] inner_tiles = [32] into %arg1 : (memref<128x256xf32> memref<128x8x32xf32>)
  return
}
// CHECK:       #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-LABEL: func.func @KC_to_KCc(
// CHECK-DAG:     %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[ubK:.*]] = arith.constant 128 : index
// CHECK-DAG:     %[[ubC:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[block:.*]] = arith.constant 32 : index
// CHECK:         scf.for %[[K:.*]] = %[[lb]] to %[[ubK]] step %[[step]] {
// CHECK:           scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[step]] {
// CHECK:             scf.for %[[c:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK:               %[[applyMapC:.*]] = affine.apply #[[MAP]](%[[C]], %[[c]])
// CHECK:               %[[scalar:.*]] = memref.load %arg0[%[[K]], %[[applyMapC]]] : memref<128x256xf32>
// CHECK:               memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[c]]] : memref<128x8x32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @KC_to_KCk(%arg0: memref<128x256xf32>, %arg1: memref<4x256x32xf32>) {
  iree_linalg_ext.pack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %arg1 : (memref<128x256xf32> memref<4x256x32xf32>)
  return
}

// CHECK:       #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-LABEL: func.func @KC_to_KCk(
// CHECK-DAG:     %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[ubC:.*]] = arith.constant 256 : index
// CHECK-DAG:     %[[ubK:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[block:.*]] = arith.constant 32 : index
// CHECK:         scf.for %[[K:.*]] = %[[lb]] to %[[ubK]] step %[[step]] {
// CHECK:           scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[step]] {
// CHECK:             scf.for %[[k:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK:               %[[applyMapK:.*]] = affine.apply #[[MAP]](%[[K]], %[[k]])
// CHECK:               %[[scalar:.*]] = memref.load %arg0[%[[applyMapK]], %[[C]]] : memref<128x256xf32>
// CHECK:               memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[k]]] : memref<4x256x32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @KCRS_to_KCRSck(%arg0: memref<128x64x1x1xf32>, %arg1: memref<4x8x1x1x8x32xf32>) {
  iree_linalg_ext.pack %arg0 inner_dims_pos = [1, 0] inner_tiles = [8, 32] into %arg1 : (memref<128x64x1x1xf32> memref<4x8x1x1x8x32xf32>)
  return
}

// CHECK-DAG:   #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-DAG:   #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK-LABEL: func.func @KCRS_to_KCRSck(
// CHECK-DAG:     %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[one:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[ubK:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[ubC:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[blockK:.*]] = arith.constant 32 : index
// CHECK:         scf.for %[[K:.*]] = %[[lb]] to %[[ubK]] step %[[one]] {
// CHECK:           scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[one]] {
// CHECK:             scf.for %[[R:.*]] = %[[lb]] to %[[one]] step %[[one]] {
// CHECK:               scf.for %[[S:.*]] = %[[lb]] to %[[one]] step %[[one]] {
// CHECK:                 scf.for %[[c:.*]] = %[[lb]] to %[[ubC]] step %[[one]] {
// CHECK:                   scf.for %[[k:.*]] = %[[lb]] to %[[blockK]] step %[[one]] {
// CHECK-DAG:                 %[[affineMapK:.*]] = affine.apply #[[MAP0]](%[[K]], %[[k]])
// CHECK-DAG:                 %[[affineMapC:.*]] = affine.apply #[[MAP1]](%[[C]], %[[c]])
// CHECK:                     %[[scalar:.*]] = memref.load %arg0[%[[affineMapK]], %[[affineMapC]], %[[R]], %[[S]]] : memref<128x64x1x1xf32>
// CHECK:                     memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[R]], %[[S]], %[[c]], %[[k]]] : memref<4x8x1x1x8x32xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @KCRS_to_KCRSsr(%arg0: memref<1x1x128x64xf32>, %arg1: memref<1x1x4x8x8x32xf32>) {
  iree_linalg_ext.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : (memref<1x1x128x64xf32> memref<1x1x4x8x8x32xf32>)
  return
}

// CHECK-DAG:   #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-DAG:   #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK-LABEL: func.func @KCRS_to_KCRSsr(
// CHECK-DAG:     %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[one:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[ubR:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[ubS:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[blockR:.*]] = arith.constant 32 : index
// CHECK:         scf.for %[[K:.*]] = %[[lb]] to %[[one]] step %[[one]] {
// CHECK:           scf.for %[[C:.*]] = %[[lb]] to %[[one]] step %[[one]] {
// CHECK:             scf.for %[[R:.*]] = %[[lb]] to %[[ubR]] step %[[one]] {
// CHECK:               scf.for %[[S:.*]] = %[[lb]] to %[[ubS]] step %[[one]] {
// CHECK:                 scf.for %[[s:.*]] = %[[lb]] to %[[ubS]] step %[[one]] {
// CHECK:                   scf.for %[[r:.*]] = %[[lb]] to %[[blockR]] step %[[one]] {
// CHECK-DAG:                 %[[affineMapR:.*]] = affine.apply #[[MAP0]](%[[R]], %[[r]])
// CHECK-DAG:                 %[[affineMapS:.*]] = affine.apply #[[MAP1]](%[[S]], %[[s]])
// CHECK:                     %[[scalar:.*]] = memref.load %arg0[%[[K]], %[[C]], %[[affineMapR]], %[[affineMapS]]] : memref<1x1x128x64xf32>
// CHECK:                     memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[R]], %[[S]], %[[s]], %[[r]]] : memref<1x1x4x8x8x32xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

// Test to check that we properly handle shuffled `inner_dims_pos` and `tiles.
// In this example, the dimension at position `0` (aka `128`) is tiled with a factor of `32`.
// While the dimension at position `2` (aka `2`) is tiled with a factor of `2`.
func.func @shuffled_dim_pos_and_tiles(%arg0: memref<128x256x2x1000xf32>, %arg1: memref<4x256x1x1000x2x32xf32>) {
  iree_linalg_ext.pack %arg0 inner_dims_pos = [2, 0] inner_tiles = [2, 32] into %arg1 : (memref<128x256x2x1000xf32> memref<4x256x1x1000x2x32xf32>)
  return
}

// CHECK-DAG:   #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-DAG:   #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-LABEL: func.func @shuffled_dim_pos_and_tiles(
// CHECK-DAG:     %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[ubDimZero:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[ubDimOne:.*]] = arith.constant 256 : index
// CHECK-DAG:     %[[ubDimThree:.*]] = arith.constant 1000 : index
// CHECK-DAG:     %[[ubDimFour:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[ubDimFive:.*]] = arith.constant 32 : index
// CHECK:         scf.for %[[i:.*]] = %[[lb]] to %[[ubDimZero]] step %[[step]] {
// CHECK:           scf.for %[[j:.*]] = %[[lb]] to %[[ubDimOne]] step %[[step]] {
// CHECK:             scf.for %[[k:.*]] = %[[lb]] to %[[step]] step %[[step]] {
// CHECK:               scf.for %[[l:.*]] = %[[lb]] to %[[ubDimThree]] step %[[step]] {
// CHECK:                 scf.for %[[m:.*]] = %[[lb]] to %[[ubDimFour]] step %[[step]] {
// CHECK:                   scf.for %[[n:.*]] = %[[lb]] to %[[ubDimFive]] step %[[step]] {
// CHECK-DAG:                 %[[affineApplyZero:.*]] = affine.apply #[[MAP0]](%[[i]], %[[n]])
// CHECK-DAG:                 %[[affineApplyOne:.*]] = affine.apply #[[MAP1]](%[[k]], %[[m]])
// CHECK:                     %[[scalar:.*]] = memref.load %arg0[%[[affineApplyZero]], %[[j]], %[[affineApplyOne]], %[[l]]] : memref<128x256x2x1000xf32>
// CHECK:                     memref.store %[[scalar]], %arg1[%[[i]], %[[j]], %[[k]], %[[l]], %[[m]], %[[n]]] : memref<4x256x1x1000x2x32xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @KCRS_to_KCRSsr(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?x8x32xf32>) {
  iree_linalg_ext.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : (memref<?x?x?x?xf32> memref<?x?x?x?x8x32xf32>)
  return
}

// CHECK-DAG:   #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-DAG:   #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK:       func.func @KCRS_to_KCRSsr(
// CHECK-DAG:     %[[zero:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[one:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[two:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[three:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[eight:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[thirtyTwo:.*]] = arith.constant 32 : index
// CHECK-DAG:     %[[dimZero:.*]] = memref.dim %arg1, %[[zero]] : memref<?x?x?x?x8x32xf32>
// CHECK-DAG:     %[[dimOne:.*]] = memref.dim %arg1, %[[one]] : memref<?x?x?x?x8x32xf32>
// CHECK-DAG:     %[[dimTwo:.*]] = memref.dim %arg1, %[[two]] : memref<?x?x?x?x8x32xf32>
// CHECK-DAG:     %[[dimThree:.*]] = memref.dim %arg1, %[[three]] : memref<?x?x?x?x8x32xf32>
// CHECK:         scf.for %[[K:.*]] = %[[zero]] to %[[dimZero]] step %[[one]] {
// CHECK:           scf.for %[[C:.*]] = %[[zero]] to %[[dimOne]] step %[[one]] {
// CHECK:             scf.for %[[R:.*]] = %[[zero]] to %[[dimTwo]] step %[[one]] {
// CHECK:               scf.for %[[S:.*]] = %[[zero]] to %[[dimThree]] step %[[one]] {
// CHECK:                 scf.for %[[s:.*]] = %[[zero]] to %[[eight]] step %[[step]] {
// CHECK:                   scf.for %[[r:.*]] = %[[zero]] to %[[thirtyTwo]] step %[[step]] {
// CHECK-DAG:                 %[[affineMapR:.*]] = affine.apply #[[MAP0]](%[[R]], %[[r]])
// CHECK-DAG:                 %[[affineMapS:.*]] = affine.apply #[[MAP1]](%[[S]], %[[s]])
// CHECK:                     %[[scalar:.*]] = memref.load %arg0[%[[K]], %[[C]], %[[affineMapR]], %[[affineMapS]]] : memref<?x?x?x?xf32>
// CHECK:                     memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[R]], %[[S]], %[[s]], %[[r]]] : memref<?x?x?x?x8x32xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @KCRS_to_KCRSsr(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?x8x?xf32>, %block : index) {
  iree_linalg_ext.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, %block] into %arg1 : (memref<?x?x?x?xf32> memref<?x?x?x?x8x?xf32>)
  return
}

// CHECK-DAG:  #[[MAP0:.*]] = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
// CHECK-DAG:  #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK:      func.func @KCRS_to_KCRSsr
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[zero:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[one:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[two:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[three:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[eight:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[five:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[dimZero:.*]] = memref.dim %[[ARG1]], %[[zero]] : memref<?x?x?x?x8x?xf32>
// CHECK-DAG:     %[[dimOne:.*]] = memref.dim %[[ARG1]], %[[one]] : memref<?x?x?x?x8x?xf32>
// CHECK-DAG:     %[[dimTwo:.*]] = memref.dim %[[ARG1]], %[[two]] : memref<?x?x?x?x8x?xf32>
// CHECK-DAG:     %[[dimThree:.*]] = memref.dim %[[ARG1]], %[[three]] : memref<?x?x?x?x8x?xf32>
// CHECK:         scf.for %[[K:.*]] = %[[zero]] to %[[dimZero]] step %[[one]] {
// CHECK:           scf.for %[[C:.*]] = %[[zero]] to %[[dimOne]] step %[[one]] {
// CHECK:             scf.for %[[R:.*]] = %[[zero]] to %[[dimTwo]] step %[[one]] {
// CHECK:               scf.for %[[S:.*]] = %[[zero]] to %[[dimThree]] step %[[one]] {
// CHECK:                 %[[dimFive:.*]] = memref.dim %[[ARG1]], %[[five]] : memref<?x?x?x?x8x?xf32>
// CHECK:                 scf.for %[[s:.*]] = %[[zero]] to %[[eight]] step %[[step]] {
// CHECK:                   scf.for %[[r:.*]] = %[[zero]] to %[[dimFive]] step %[[step]] {
// CHECK-DAG:                 %[[affineMapR:.*]] = affine.apply #[[MAP0]](%[[R]], %[[r]])[%[[ARG2]]]
// CHECK-DAG:                 %[[affineMapS:.*]] = affine.apply #[[MAP1]](%[[S]], %[[s]])
// CHECK:                     %[[scalar:.*]] = memref.load %[[ARG0]][%[[K]], %[[C]], %[[affineMapR]], %[[affineMapS]]] : memref<?x?x?x?xf32>
// CHECK:                     memref.store %[[scalar]], %[[ARG1]][%[[K]], %[[C]], %[[R]], %[[S]], %[[s]], %[[r]]] : memref<?x?x?x?x8x?xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @NCnc_to_NC(%arg0: memref<128x256xf32>, %arg1: memref<4x8x32x32xf32>) {
  iree_linalg_ext.unpack %arg1 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg0 : (memref<4x8x32x32xf32> memref<128x256xf32>)
  return
}

// CHECK-DAG:   #[[MAP_FLOOR:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:   #[[MAP_MOD:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK:       func.func @NCnc_to_NC
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[UBI:.*]] = arith.constant 128 : index
// CHECK-DAG:     %[[UBJ:.*]] = arith.constant 256 : index
// CHECK:         scf.for %[[I:.*]] = %[[LB]] to %[[UBI]] step %[[STEP]] {
// CHECK:           scf.for %[[J:.*]] = %[[LB]] to %[[UBJ]] step %[[STEP]] {
// CHECK-DAG:         %[[FLOORI:.*]] = affine.apply #[[MAP_FLOOR]](%[[I]])
// CHECK-DAG:         %[[FLOORJ:.*]] = affine.apply #[[MAP_FLOOR]](%[[J]])
// CHECK-DAG:         %[[MODI:.*]] = affine.apply #[[MAP_MOD]](%[[I]])
// CHECK-DAG:         %[[MODJ:.*]] = affine.apply #[[MAP_MOD]](%[[J]])
// CHECK:             %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[FLOORI]], %[[FLOORJ]], %[[MODI]], %[[MODJ]]] : memref<4x8x32x32xf32>
// CHECK:             memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]]] : memref<128x256xf32>
// CHECK:           }
// CHECK:         }

// -----

func.func @KCck_to_KC(%arg0: memref<128x256xf32>, %arg1: memref<4x8x32x32xf32>) {
  iree_linalg_ext.unpack %arg1 inner_dims_pos = [1, 0] inner_tiles = [32, 32] into %arg0 : (memref<4x8x32x32xf32> memref<128x256xf32>)
  return
}

// CHECK-DAG:   #[[MAP_FLOOR:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:   #[[MAP_MOD:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK:       func.func @KCck_to_KC
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[UBI:.*]] = arith.constant 128 : index
// CHECK-DAG:     %[[UBJ:.*]] = arith.constant 256 : index
// CHECK:         scf.for %[[I:.*]] = %[[LB]] to %[[UBI]] step %[[STEP]] {
// CHECK:           scf.for %[[J:.*]] = %[[LB]] to %[[UBJ]] step %[[STEP]] {
// CHECK-DAG:         %[[FLOORI:.*]] = affine.apply #[[MAP_FLOOR]](%[[I]])
// CHECK-DAG:         %[[FLOORJ:.*]] = affine.apply #[[MAP_FLOOR]](%[[J]])
// CHECK-DAG:         %[[MODI:.*]] = affine.apply #[[MAP_MOD]](%[[I]])
// CHECK-DAG:         %[[MODJ:.*]] = affine.apply #[[MAP_MOD]](%[[J]])
// CHECK:             %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[FLOORI]], %[[FLOORJ]], %[[MODJ]], %[[MODI]]] : memref<4x8x32x32xf32>
// CHECK:             memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]]] : memref<128x256xf32>
// CHECK:           }
// CHECK:         }

// -----

// This should be a simple collapse shape.
func.func @KCc_to_KC(%arg0: memref<128x256xf32>, %arg1: memref<128x8x32xf32>) {
  iree_linalg_ext.unpack %arg1 inner_dims_pos = [1] inner_tiles = [32] into %arg0 : (memref<128x8x32xf32> memref<128x256xf32>)
  return
}

// CHECK-DAG:   #[[MAP_FLOOR:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:   #[[MAP_MOD:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK:       func.func @KCc_to_KC
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[UBI:.*]] = arith.constant 128 : index
// CHECK-DAG:     %[[UBJ:.*]] = arith.constant 256 : index
// CHECK:         scf.for %[[I:.*]] = %[[LB]] to %[[UBI]] step %[[STEP]] {
// CHECK:           scf.for %[[J:.*]] = %[[LB]] to %[[UBJ]] step %[[STEP]] {
// CHECK-DAG:         %[[FLOORJ:.*]] = affine.apply #[[MAP_FLOOR]](%[[J]])
// CHECK-DAG:         %[[MODJ:.*]] = affine.apply #[[MAP_MOD]](%[[J]])
// CHECK:             %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[I]], %[[FLOORJ]], %[[MODJ]]] : memref<128x8x32xf32>
// CHECK:             memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]]] : memref<128x256xf32>
// CHECK:           }
// CHECK:         }



// -----

func.func @KCRSsr_to_KCRS(%arg0: memref<1x1x128x64xf32>, %arg1: memref<1x1x4x8x8x32xf32>) {
  iree_linalg_ext.unpack %arg1 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg0 : (memref<1x1x4x8x8x32xf32> memref<1x1x128x64xf32>)
  return
}

// CHECK-DAG:   #[[MAP_FLOORK:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:   #[[MAP_MODK:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK-DAG:   #[[MAP_FLOORL:.*]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG:   #[[MAP_MODL:.*]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK:       func.func @KCRSsr_to_KCRS
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[UBK:.*]] = arith.constant 128 : index
// CHECK-DAG:     %[[UBL:.*]] = arith.constant 64 : index
// CHECK:         scf.for %[[I:.*]] = %[[LB]] to %[[STEP]] step %[[STEP]] {
// CHECK:           scf.for %[[J:.*]] = %[[LB]] to %[[STEP]] step %[[STEP]] {
// CHECK:             scf.for %[[K:.*]] = %[[LB]] to %[[UBK]] step %[[STEP]] {
// CHECK:               scf.for %[[L:.*]] = %[[LB]] to %[[UBL]] step %[[STEP]] {
// CHECK-DAG:             %[[FLOORK:.*]] = affine.apply #[[MAP_FLOORK]](%[[K]])
// CHECK-DAG:             %[[FLOORL:.*]] = affine.apply #[[MAP_FLOORL]](%[[L]])
// CHECK-DAG:             %[[MODK:.*]] = affine.apply #[[MAP_MODK]](%[[K]])
// CHECK-DAG:             %[[MODL:.*]] = affine.apply #[[MAP_MODL]](%[[L]])
// CHECK:                 %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[I]], %[[J]], %[[FLOORK]], %[[FLOORL]], %[[MODL]], %[[MODK]]] : memref<1x1x4x8x8x32xf32>
// CHECK:                 memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]], %[[K]], %[[L]]] : memref<1x1x128x64xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @shuffled_dim_pos_and_tiles(%arg0: memref<128x256x2x1000xf32>, %arg1: memref<4x256x1x1000x2x32xf32>) {
  iree_linalg_ext.unpack %arg1 inner_dims_pos = [2, 0] inner_tiles = [2, 32] into %arg0 : (memref<4x256x1x1000x2x32xf32> memref<128x256x2x1000xf32>)
  return
}

// CHECK-DAG:   #[[MAP_FLOORI:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:   #[[MAP_MODI:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK-DAG:   #[[MAP_FLOORK:.*]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK-DAG:   #[[MAP_MODK:.*]] = affine_map<(d0) -> (d0 mod 2)>
// CHECK:       func.func @shuffled_dim_pos_and_tiles
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[UBI:.*]] = arith.constant 128 : index
// CHECK-DAG:     %[[UBJ:.*]] = arith.constant 256 : index
// CHECK-DAG:     %[[UBK:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[UBL:.*]] = arith.constant 1000 : index
// CHECK:         scf.for %[[I:.*]] = %[[LB]] to %[[UBI]] step %[[STEP]] {
// CHECK:           scf.for %[[J:.*]] = %[[LB]] to %[[UBJ]] step %[[STEP]] {
// CHECK:             scf.for %[[K:.*]] = %[[LB]] to %[[UBK]] step %[[STEP]] {
// CHECK:               scf.for %[[L:.*]] = %[[LB]] to %[[UBL]] step %[[STEP]] {
// CHECK-DAG:             %[[FLOORI:.*]] = affine.apply #[[MAP_FLOORI]](%[[I]])
// CHECK-DAG:             %[[MODI:.*]] = affine.apply #[[MAP_MODI]](%[[I]])
// CHECK-DAG:             %[[FLOORK:.*]] = affine.apply #[[MAP_FLOORK]](%[[K]])
// CHECK-DAG:             %[[MODK:.*]] = affine.apply #[[MAP_MODK]](%[[K]])
// CHECK:                 %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[FLOORI]], %[[J]], %[[FLOORK]], %[[L]], %[[MODK]], %[[MODI]]] : memref<4x256x1x1000x2x32xf32>
// CHECK:                 memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]], %[[K]], %[[L]]] : memref<128x256x2x1000xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @KCRSsr_to_KCRS(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?x8x32xf32>) {
  iree_linalg_ext.unpack %arg1 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg0 : (memref<?x?x?x?x8x32xf32> memref<?x?x?x?xf32>)
  return
}

// CHECK-DAG:    #[[MAP_FLOORK:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:    #[[MAP_MODK:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK-DAG:    #[[MAP_FLOORL:.*]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG:    #[[MAP_MODL:.*]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK:       func.func @KCRSsr_to_KCRS
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[UBI:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?x?x?x?xf32>
// CHECK-DAG:     %[[UBJ:.*]] = memref.dim %[[ARG0]], %[[C1]] : memref<?x?x?x?xf32>
// CHECK-DAG:     %[[UBK:.*]] = memref.dim %[[ARG0]], %[[C2]] : memref<?x?x?x?xf32>
// CHECK-DAG:     %[[UBL:.*]] = memref.dim %[[ARG0]], %[[C3]] : memref<?x?x?x?xf32>
// CHECK:         scf.for %[[I:.*]] = %[[C0]] to %[[UBI]] step %[[C1]] {
// CHECK:           scf.for %[[J:.*]] = %[[C0]] to %[[UBJ]] step %[[C1]] {
// CHECK:             scf.for %[[K:.*]] = %[[C0]] to %[[UBK]] step %[[C1]] {
// CHECK:               scf.for %[[L:.*]] = %[[C0]] to %[[UBL]] step %[[C1]] {
// CHECK-DAG:             %[[FLOORK:.*]] = affine.apply #[[MAP_FLOORK]](%[[K]])
// CHECK-DAG:             %[[FLOORL:.*]] = affine.apply #[[MAP_FLOORL]](%[[L]])
// CHECK-DAG:             %[[MODK:.*]] = affine.apply #[[MAP_MODK]](%[[K]])
// CHECK-DAG:             %[[MODL:.*]] = affine.apply #[[MAP_MODL]](%[[L]])
// CHECK:                 %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[I]], %[[J]], %[[FLOORK]], %[[FLOORL]], %[[MODL]], %[[MODK]]] : memref<?x?x?x?x8x32xf32>
// CHECK:                 memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]], %[[K]], %[[L]]] : memref<?x?x?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @KCRSsr_to_KCRS(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?x8x?xf32>, %block : index) {
  iree_linalg_ext.unpack %arg1 inner_dims_pos = [3, 2] inner_tiles = [8, %block] into %arg0 : (memref<?x?x?x?x8x?xf32> memref<?x?x?x?xf32>)
  return
}

// CHECK-DAG:   #[[MAP_FLOORK:.*]] = affine_map<(d0)[s0] -> (d0 floordiv s0)>
// CHECK-DAG:   #[[MAP_MODK:.*]] = affine_map<(d0)[s0] -> (d0 mod s0)>
// CHECK-DAG:   #[[MAP_FLOORL:.*]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG:   #[[MAP_MODL:.*]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK:       func.func @KCRSsr_to_KCRS
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[UBI:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?x?x?x?xf32>
// CHECK-DAG:     %[[UBJ:.*]] = memref.dim %[[ARG0]], %[[C1]] : memref<?x?x?x?xf32>
// CHECK-DAG:     %[[UBK:.*]] = memref.dim %[[ARG0]], %[[C2]] : memref<?x?x?x?xf32>
// CHECK-DAG:     %[[UBL:.*]] = memref.dim %[[ARG0]], %[[C3]] : memref<?x?x?x?xf32>
// CHECK:         scf.for %[[I:.*]] = %[[C0]] to %[[UBI]] step %[[C1]] {
// CHECK:           scf.for %[[J:.*]] = %[[C0]] to %[[UBJ]] step %[[C1]] {
// CHECK:             scf.for %[[K:.*]] = %[[C0]] to %[[UBK]] step %[[C1]] {
// CHECK:               scf.for %[[L:.*]] = %[[C0]] to %[[UBL]] step %[[C1]] {
// CHECK-DAG:             %[[FLOORK:.*]] = affine.apply #[[MAP_FLOORK]](%[[K]])[%[[ARG2]]]
// CHECK-DAG:             %[[FLOORL:.*]] = affine.apply #[[MAP_FLOORL]](%[[L]])
// CHECK-DAG:             %[[MODK:.*]] = affine.apply #[[MAP_MODK]](%[[K]])[%[[ARG2]]]
// CHECK-DAG:             %[[MODL:.*]] = affine.apply #[[MAP_MODL]](%[[L]])
// CHECK:                 %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[I]], %[[J]], %[[FLOORK]], %[[FLOORL]], %[[MODL]], %[[MODK]]] : memref<?x?x?x?x8x?xf32>
// CHECK:                 memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]], %[[K]], %[[L]]] : memref<?x?x?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @unpack_undo_padding(%input: memref<2x8x8x2xf32>, %output: memref<13x15xf32>) {
  iree_linalg_ext.unpack %input inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (memref<2x8x8x2xf32> memref<13x15xf32>)
  return
}
// CHECK-DAG:  #[[MAP_FLOORI:.*]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG:  #[[MAP_MODI:.*]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK-DAG:  #[[MAP_FLOORJ:.*]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK-DAG:  #[[MAP_MODJ:.*]] = affine_map<(d0) -> (d0 mod 2)>
// CHECK:      func.func @unpack_undo_padding
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[OUTPUT:[a-zA-Z0-9]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C13:.+]] = arith.constant 13 : index
// CHECK-DAG:    %[[C15:.+]] = arith.constant 15 : index
// CHECK:        scf.for %[[I:.+]] = %[[C0]] to %[[C13]] step %[[C1]] {
// CHECK:          scf.for %[[J:.+]] = %[[C0]] to %[[C15]] step %[[C1]] {
// CHECK-DAG:        %[[OUTER_I:.+]] = affine.apply #[[MAP_FLOORI]](%[[I]])
// CHECK-DAG:        %[[INNER_I:.+]] = affine.apply #[[MAP_MODI]](%[[I]])
// CHECK-DAG:        %[[OUTER_J:.+]] = affine.apply #[[MAP_FLOORJ]](%[[J]])
// CHECK-DAG:        %[[INNER_J:.+]] = affine.apply #[[MAP_MODJ]](%[[J]])
// CHECK:            %[[VAL:.+]] = memref.load %[[INPUT]][%[[OUTER_I]], %[[OUTER_J]], %[[INNER_I]], %[[INNER_J]]]
// CHECK:            memref.store %[[VAL]], %[[OUTPUT]][%[[I]], %[[J]]]

// -----

func.func @KC_to_CKkc(%arg0: memref<128x256xf32>, %arg1: memref<32x4x32x8xf32>) {
  iree_linalg_ext.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : (memref<128x256xf32> memref<32x4x32x8xf32>)
  return
}

// CHECK:      #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK:      #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK:      func.func @KC_to_CKkc
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[OUTPUT:[a-zA-Z0-9]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:    %[[C8:.+]] = arith.constant 8 : index
// CHECK:        scf.for %[[C:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:          scf.for %[[K:.+]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:            scf.for %[[k:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:              scf.for %[[c:.+]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:        %[[MAPK:.+]] = affine.apply #[[MAP0]](%[[K]], %[[k]])
// CHECK:        %[[MAPC:.+]] = affine.apply #[[MAP1]](%[[C]], %[[c]])
// CHECK:        %[[VAL:.+]] = memref.load %[[ARG0]][%[[MAPK]], %[[MAPC]]] : memref<128x256xf32>
// CHECK:        memref.store %[[VAL]], %[[ARG1]][%[[C]], %[[K]], %[[k]], %[[c]]] : memref<32x4x32x8xf32>
// CHECK:              }
// CHECK:            }
// CHECK:          }
// CHECK:        }

// -----

func.func @CKkc_to_KC(%arg0: memref<128x256xf32>, %arg1: memref<32x4x32x8xf32>) {
  iree_linalg_ext.unpack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg0 : (memref<32x4x32x8xf32> memref<128x256xf32>)
  return
}

// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK:       func.func @CKkc_to_KC
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[C256:.+]] = arith.constant 256 : index
// CHECK:         scf.for %[[K:.+]] = %[[C0]] to %[[C128]] step %[[C1]] {
// CHECK:           scf.for %[[C:.+]] = %[[C0]] to %[[C256]] step %[[C1]] {
// CHECK-DAG:         %[[FLOORK:.+]] = affine.apply #[[MAP0]](%[[K]])
// CHECK-DAG:         %[[MODK:.+]] = affine.apply #[[MAP1]](%[[K]])
// CHECK-DAG:         %[[FLOORC:.+]] = affine.apply #[[MAP2]](%[[C]])
// CHECK-DAG:         %[[MODC:.+]] = affine.apply #[[MAP3]](%[[C]])
// CHECK:             %[[VAL:.+]] = memref.load %[[ARG1]][%[[FLOORC]], %[[FLOORK]], %[[MODK]], %[[MODC]]] : memref<32x4x32x8xf32>
// CHECK:             memref.store %[[VAL]], %[[ARG0]][%[[K]], %[[C]]] : memref<128x256xf32>
// CHECK:           }
// CHECK:         }

// -----

func.func @NPQK_to_NKPQk(%arg0: memref<1x56x56x64xf32>, %arg1: memref<1x2x56x56x32xf32>) {
  iree_linalg_ext.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %arg1 : (memref<1x56x56x64xf32> memref<1x2x56x56x32xf32>)
  return
}

// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK:       func.func @NPQK_to_NKPQk
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C56:.+]] = arith.constant 56 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK:         scf.for %[[N:.+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK:           scf.for %[[K:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:             scf.for %[[P:.+]] = %[[C0]] to %[[C56]] step %[[C1]] {
// CHECK:               scf.for %[[Q:.+]] = %[[C0]] to %[[C56]] step %[[C1]] {
// CHECK:                 scf.for %[[k:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:                   %[[APPLY:.+]] = affine.apply #[[MAP0]](%[[K]], %[[k]])
// CHECK:                   %[[VAL:.+]] = memref.load %[[INPUT]][%[[N]], %[[P]], %[[Q]], %[[APPLY]]] : memref<1x56x56x64xf32>
// CHECK:                   memref.store %[[VAL]], %[[OUTPUT]][%[[N]], %[[K]], %[[P]], %[[Q]], %[[k]]] : memref<1x2x56x56x32xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }

// -----

func.func @unpack(%arg0: memref<1x4x6x6x2xf32>, %arg1: memref<1x6x6x8xf32>) {
  iree_linalg_ext.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %arg1 : (memref<1x4x6x6x2xf32> memref<1x6x6x8xf32>)
  return
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0 mod 2)>
// CHECK: func.func @unpack(
// CHECK-SAME:  %[[INPUT:[a-zA-Z0-9]+]]
// CHECK-SAME:  %[[OUTPUT:[a-zA-Z0-9]+]]
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK:     scf.for %[[I:.+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK:       scf.for %[[J:.+]] = %[[C0]] to %[[C6]] step %[[C1]] {
// CHECK:         scf.for %[[K:.+]] = %[[C0]] to %[[C6]] step %[[C1]] {
// CHECK:           scf.for %[[L:.+]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:             %[[APPLY_TILE:.+]] = affine.apply #[[MAP]](%[[L]])
// CHECK:             %[[APPLY_LOOP:.+]] = affine.apply #[[MAP1]](%[[L]])
// CHECK:             %[[LOAD:.+]] = memref.load %[[INPUT]][%[[I]], %[[APPLY_TILE]], %[[J]], %[[K]], %[[APPLY_LOOP]]] : memref<1x4x6x6x2xf32>
// CHECK:             memref.store %[[LOAD]], %[[OUTPUT]][%[[I]], %[[J]], %[[K]], %[[L]]] : memref<1x6x6x8xf32>
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
