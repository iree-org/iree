// RUN: iree-dialects-opt -split-input-file -iree-linalg-ext-to-loops %s | FileCheck %s

func @sort_1d(%arg0: memref<128xi32>) {
  iree_linalg_ext.sort dimension(0)
    outs(%arg0 : memref<128xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
    %0 = arith.cmpi sgt, %arg2, %arg3 : i32
    iree_linalg_ext.yield %0 : i1
  }
  return
}
// CHECK-LABEL: func @sort_1d
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

func @sort_2d(%arg0: memref<16x32xi32>) {
  iree_linalg_ext.sort dimension(0)
    outs(%arg0 : memref<16x32xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
    %0 = arith.cmpi sgt, %arg2, %arg3 : i32
    iree_linalg_ext.yield %0 : i1
  }
  return
}
// CHECK-LABEL: func @sort_2d
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

func @sort_multi(%arg0: memref<128xf32>, %arg1: memref<128xi32>) {
  iree_linalg_ext.sort
    dimension(0)
    outs(%arg0, %arg1 : memref<128xf32>, memref<128xi32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i32, %arg5: i32):  // no predecessors
    %0 = arith.cmpf ogt, %arg2, %arg3 : f32
    iree_linalg_ext.yield %0 : i1
  }
  return
}
// CHECK-LABEL: func @sort_multi
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

func @scatter_update_scalar_1D(
    %original: memref<8xi32>, %indices: memref<3x1xi32>,
    %updates: memref<3xi32>) {
  iree_linalg_ext.scatter unique_indices(true)
    ins(%updates, %indices : memref<3xi32>, memref<3x1xi32>)
    outs(%original : memref<8xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK-LABEL: func @scatter_update_scalar_1D
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

func @scatter_add_scalar_2D(
    %original: memref<4x3xi32>, %indices: memref<3x2xi32>,
    %updates: memref<3xi32>) {
  iree_linalg_ext.scatter unique_indices(true)
    ins(%updates, %indices : memref<3xi32>, memref<3x2xi32>)
    outs(%original : memref<4x3xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = arith.addi %arg1, %arg0 : i32
    iree_linalg_ext.yield %0 : i32
  }
  return
}
// CHECK-LABEL: func @scatter_add_scalar_2D
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

func @scatter_update_slice_2D(
    %original: memref<4x3xi32>, %indices: memref<2x1xi32>,
    %updates: memref<2x3xi32>) {
  iree_linalg_ext.scatter unique_indices(true)
    ins(%updates, %indices : memref<2x3xi32>, memref<2x1xi32>)
    outs(%original : memref<4x3xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK:       func @scatter_update_slice_2D
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

func @scatter_add_scalar_1D(
    %original: memref<8xi32>, %indices: memref<3x1xi32>,
    %updates: memref<3xi32>) {
  iree_linalg_ext.scatter unique_indices(true)
    ins(%updates, %indices : memref<3xi32>, memref<3x1xi32>)
    outs(%original : memref<8xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = arith.addi %arg1, %arg0 : i32
    iree_linalg_ext.yield %0 : i32
  }
  return
}
// CHECK-LABEL: func @scatter_add_scalar_1D
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

func @scatter_add_slice_2D(
    %original: memref<4x3xi32>, %indices: memref<2x1xi32>,
    %updates: memref<2x3xi32>) {
  iree_linalg_ext.scatter unique_indices(true)
    ins(%updates, %indices : memref<2x3xi32>, memref<2x1xi32>)
    outs(%original : memref<4x3xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = arith.addi %arg1, %arg0 : i32
    iree_linalg_ext.yield %0 : i32
  }
  return
}
// CHECK:       func @scatter_add_slice_2D
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

func @scatter_update_scalar_dynamic_1D(
    %original: memref<?xi32>, %indices: memref<?x1xi32>,
    %updates: memref<?xi32>) {
  iree_linalg_ext.scatter unique_indices(true)
    ins(%updates, %indices : memref<?xi32>, memref<?x1xi32>)
    outs(%original : memref<?xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK-LABEL: func @scatter_update_scalar_dynamic_1D
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

func @scatter_add_scalar_dynamic_2D(
    %original: memref<?x?xi32>, %indices: memref<?x2xi32>,
    %updates: memref<?xi32>) {
  iree_linalg_ext.scatter unique_indices(true)
    ins(%updates, %indices : memref<?xi32>, memref<?x2xi32>)
    outs(%original : memref<?x?xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = arith.addi %arg1, %arg0 : i32
    iree_linalg_ext.yield %0 : i32
  }
  return
}
// CHECK-LABEL: func @scatter_add_scalar_dynamic_2D
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

func @scatter_update_slice_dynamic_2D(
    %original: memref<?x?xi32>, %indices: memref<?x1xi32>,
    %updates: memref<?x?xi32>) {
  iree_linalg_ext.scatter unique_indices(true)
    ins(%updates, %indices : memref<?x?xi32>, memref<?x1xi32>)
    outs(%original : memref<?x?xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK:       func @scatter_update_slice_dynamic_2D
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

func @scatter_partial_slices(%arg0: memref<2x64x12xf32>, %arg1: memref<2x3xi32>, %arg2: memref<2x1x12xf32>) {
  iree_linalg_ext.scatter
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

func @fft_1D(%real: memref<16xf32>, %imag: memref<16xf32>) {
  %stage = arith.constant 1 : index
  iree_linalg_ext.fft
    ins(%stage: index)
    outs(%real, %imag: memref<16xf32>, memref<16xf32>)
  return
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0)>
// CHECK:       func @fft_1D
// CHECK-SAME:    %[[REAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[IMAG:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[COEFF:.+]] = arith.constant -3.14159274 : f32
// CHECK:         scf.for %[[K:.+]] = %[[C0]] to %[[C16]] step %[[C2]]
// CHECK-DAG:       %[[HM:.+]] = arith.shrsi %[[C2]], %[[C1]] : index
// CHECK:           %[[L_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[K]]] [%[[HM]]] [1]
// CHECK:           %[[L_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[K]]] [%[[HM]]] [1]
// CHECK:           %[[R_OFFSET:.+]] = arith.addi %[[K]], %[[HM]] : index
// CHECK:           %[[R_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[R_OFFSET]]] [%[[HM]]] [1]
// CHECK:           %[[R_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[R_OFFSET]]] [%[[HM]]] [1]
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

func @fft_2D(%real: memref<?x16xf32>, %imag: memref<?x16xf32>) {
  %stage = arith.constant 2 : index
  iree_linalg_ext.fft
    ins(%stage: index)
    outs(%real, %imag: memref<?x16xf32>, memref<?x16xf32>)
  return
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 16 + s0 + d1)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:       func @fft_2D(
// CHECK-SAME:    %[[REAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[IMAG:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[D0:.+]] = memref.dim %[[REAL]], %[[C0]] : memref<?x16xf32>
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[D0]] step %[[C1]]
// CHECK:           scf.for %[[K:.+]] = %[[C0]] to %[[C16]] step %[[C4]]
// CHECK-DAG:         %[[HM:.+]] = arith.shrsi %[[C4]], %[[C1]] : index
// CHECK:             %[[L_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[I]], %[[K]]] [1, %[[HM]]] [1, 1]
// CHECK:             %[[L_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[I]], %[[K]]] [1, %[[HM]]] [1, 1]
// CHECK:             %[[R_OFFSET:.+]] = arith.addi %[[K]], %[[HM]] : index
// CHECK:             %[[R_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[I]], %[[R_OFFSET]]] [1, %[[HM]]] [1, 1]
// CHECK:             %[[R_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[I]], %[[R_OFFSET]]] [1, %[[HM]]] [1, 1]
// CHECK:             linalg.generic
// CHECK-SAME:          indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:          iterator_types = ["parallel", "parallel"]
// CHECK-SAME:          outs(%[[L_REAL_SLICE]], %[[L_IMAG_SLICE]], %[[R_REAL_SLICE]], %[[R_IMAG_SLICE]]
//
//                    The computation is bascially the same, and they are
//                    checked above. Here only checks the different part.
// CHECK:             %{{.+}} = linalg.index 1 : index

// -----

func @fft_2D_coef_buf(%real: memref<?x16xf32>, %imag: memref<?x16xf32>,
                      %coef_real: memref<1xf32>, %coef_imag: memref<1xf32>) {
  %stage = arith.constant 1 : index
  iree_linalg_ext.fft
    ins(%stage, %coef_real, %coef_imag: index, memref<1xf32>, memref<1xf32>)
    outs(%real, %imag: memref<?x16xf32>, memref<?x16xf32>)
  return
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 16 + s0 + d1)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:       func @fft_2D_coef_buf
// CHECK-SAME:    %[[REAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[IMAG:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[COEF_REAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[COEF_IMAG:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[D0:.+]] = memref.dim %[[REAL]], %[[C0]] : memref<?x16xf32>
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[D0]] step %[[C1]]
// CHECK:           scf.for %[[K:.+]] = %[[C0]] to %[[C16]] step %[[C2]]
// CHECK-DAG:         %[[HM:.+]] = arith.shrsi %[[C2]], %[[C1]] : index
// CHECK:             %[[L_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[I]], %[[K]]] [1, %[[HM]]] [1, 1]
// CHECK:             %[[L_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[I]], %[[K]]] [1, %[[HM]]] [1, 1]
// CHECK:             %[[R_OFFSET:.+]] = arith.addi %[[K]], %[[HM]] : index
// CHECK:             %[[R_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[I]], %[[R_OFFSET]]] [1, %[[HM]]] [1, 1]
// CHECK:             %[[R_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[I]], %[[R_OFFSET]]] [1, %[[HM]]] [1, 1]
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

func @reverse_dim_0(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>) {
  iree_linalg_ext.reverse
    dimensions(dense<0> : tensor<1xi64>)
    ins(%arg0 : memref<?x?xi32>)
    outs(%arg1 : memref<?x?xi32>)
  return
}
// CHECK-LABEL: func @reverse_dim_0
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

func @scan_1d_inclusive(%0: memref<128xi32>, %1: memref<128xi32>) {
  %c0 = memref.alloc() : memref<i32>
  iree_linalg_ext.scan dimension(0) inclusive(true)
    ins(%0 : memref<128xi32>) outs(%1, %c0 : memref<128xi32>, memref<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  }
  return
}
// CHECK-LABEL: func @scan_1d_inclusive
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

func @scan_1d_exclusive(%0: memref<128xi32>, %1: memref<128xi32>) {
  %c0 = memref.alloc() : memref<i32>
  iree_linalg_ext.scan dimension(0) inclusive(false)
    ins(%0 : memref<128xi32>) outs(%1, %c0 : memref<128xi32>, memref<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  }
  return
}
// CHECK-LABEL: func @scan_1d_exclusive
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

func @scan_2d(%0: memref<16x32xi32>, %1: memref<16x32xi32>) {
  %t0 = memref.alloc() : memref<32xi32>
  iree_linalg_ext.scan dimension(0) inclusive(true)
    ins(%0 : memref<16x32xi32>) outs(%1, %t0 : memref<16x32xi32>, memref<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  }
  return
}
// CHECK-LABEL: func @scan_2d
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
