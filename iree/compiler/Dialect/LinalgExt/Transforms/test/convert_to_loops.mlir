// RUN: iree-opt -split-input-file -iree-linalg-ext-to-loops %s | IreeFileCheck %s

func @sort_1d(%arg0: memref<128xi32>) {
  linalg_ext.sort dimension(0)
    outs(%arg0 : memref<128xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
    %0 = cmpi sgt, %arg2, %arg3 : i32
    linalg_ext.yield %0 : i1
  }
  return
}
// CHECK-LABEL: func @sort_1d
// CHECK-SAME:    %[[BUF:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C128:.+]] = constant 128 : index
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C127:.+]] = constant 127 : index
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK:           scf.for %[[ARG2:.+]] = %[[C0]] to %[[C127]] step %[[C1]]
// CHECK:             %[[T1:.+]] = addi %[[ARG2]], %[[C1]] : index
// CHECK:             %[[V1:.+]] = memref.load %[[BUF]][%[[ARG2]]]
// CHECK:             %[[V2:.+]] = memref.load %[[BUF]][%[[T1]]]
// CHECK:             %[[COND:.+]] = cmpi sgt, %[[V1]], %[[V2]] : i32
// CHECK:             scf.if %[[COND]] {
// CHECK:             } else {
// CHECK:               %[[T2:.+]] = addi %[[ARG2]], %[[C1]] : index
// CHECK:               memref.store %[[V2]], %[[BUF]][%[[ARG2]]]
// CHECK:               memref.store %[[V1]], %[[BUF]][%[[T2]]]
// CHECK:             }

// -----

func @sort_2d(%arg0: memref<16x32xi32>) {
  linalg_ext.sort dimension(0)
    outs(%arg0 : memref<16x32xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
    %0 = cmpi sgt, %arg2, %arg3 : i32
    linalg_ext.yield %0 : i1
  }
  return
}
// CHECK-LABEL: func @sort_2d
// CHECK-SAME:    %[[BUF:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C16:.+]] = constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = constant 32 : index
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C15:.+]] = constant 15 : index
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C16]] step %[[C1]]
// CHECK:           scf.for %[[ARG2:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK:             scf.for %[[ARG3:.+]] = %[[C0]] to %[[C15]] step %[[C1]]
// CHECK:               %[[T1:.+]] = addi %[[ARG3]], %[[C1]] : index
// CHECK:               %[[V1:.+]] = memref.load %[[BUF]][%[[ARG3]], %[[ARG2]]]
// CHECK:               %[[V2:.+]] = memref.load %[[BUF]][%[[T1]], %[[ARG2]]]
// CHECK:               %[[COND:.+]] = cmpi sgt, %[[V1]], %[[V2]] : i32
// CHECK:               scf.if %[[COND]] {
// CHECK:               } else {
// CHECK:                 %[[T2:.+]] = addi %[[ARG3]], %[[C1]] : index
// CHECK:                 memref.store %[[V2]], %[[BUF]][%[[ARG3]], %[[ARG2]]]
// CHECK:                 memref.store %[[V1]], %[[BUF]][%[[T2]], %[[ARG2]]]
// CHECK:               }

// -----

func @sort_multi(%arg0: memref<128xf32>, %arg1: memref<128xi32>) {
  linalg_ext.sort
    outs(%arg0, %arg1 : memref<128xf32>, memref<128xi32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i32, %arg5: i32):  // no predecessors
    %0 = cmpf ogt, %arg2, %arg3 : f32
    linalg_ext.yield %0 : i1
  }
  return
}
// CHECK-LABEL: func @sort_multi
// CHECK-SAME:    %[[BUF1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[BUF2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C128:.+]] = constant 128 : index
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C127:.+]] = constant 127 : index
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK:           scf.for %[[ARG2:.+]] = %[[C0]] to %[[C127]] step %[[C1]]
// CHECK:             %[[T1:.+]] = addi %[[ARG2]], %[[C1]] : index
// CHECK:             %[[V1:.+]] = memref.load %[[BUF1]][%[[ARG2]]]
// CHECK:             %[[V2:.+]] = memref.load %[[BUF1]][%[[T1]]]
// CHECK:             %[[V3:.+]] = memref.load %[[BUF2]][%[[ARG2]]]
// CHECK:             %[[V4:.+]] = memref.load %[[BUF2]][%[[T1]]]
// CHECK:             %[[COND:.+]] = cmpf ogt, %[[V1]], %[[V2]] : f32
// CHECK:             scf.if %[[COND]] {
// CHECK:             } else {
// CHECK:               %[[T2:.+]] = addi %[[ARG2]], %[[C1]] : index
// CHECK:               memref.store %[[V2]], %[[BUF1]][%[[ARG2]]]
// CHECK:               memref.store %[[V1]], %[[BUF1]][%[[T2]]]
// CHECK:               memref.store %[[V4]], %[[BUF2]][%[[ARG2]]]
// CHECK:               memref.store %[[V3]], %[[BUF2]][%[[T2]]]
// CHECK:             }

// -----

func @scatter_update_scalar_1D(
    %original: memref<8xi32>, %indices: memref<3x1xi32>,
    %updates: memref<3xi32>) {
  linalg_ext.scatter
    ins(%updates, %indices : memref<3xi32>, memref<3x1xi32>)
    outs(%original : memref<8xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK-LABEL: func @scatter_update_scalar_1D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = constant 3 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:           %[[T1:.+]] = memref.load %[[UPDATES]][%[[I]]] : memref<3xi32>
// CHECK:           %[[T2:.+]] =  memref.load %[[INDICES]][%[[I]], %[[C0]]] : memref<3x1xi32>
// CHECK:           %[[IDX:.+]] = index_cast %[[T2]] : i32 to index
// CHECK:           memref.store %[[T1]], %[[ORIGINAL]][%[[IDX]]]

// -----

func @scatter_add_scalar_2D(
    %original: memref<4x3xi32>, %indices: memref<3x2xi32>,
    %updates: memref<3xi32>) {
  linalg_ext.scatter
    ins(%updates, %indices : memref<3xi32>, memref<3x2xi32>)
    outs(%original : memref<4x3xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = addi %arg1, %arg0 : i32
    linalg_ext.yield %0 : i32
  }
  return
}
// CHECK-LABEL: func @scatter_add_scalar_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = constant 3 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:           %[[T1:.+]] = memref.load %[[UPDATES]][%[[I]]] : memref<3xi32>
// CHECK:           %[[T2:.+]] = memref.load %[[INDICES]][%[[I]], %[[C0]]] : memref<3x2xi32>
// CHECK:           %[[IDX1:.+]] = index_cast %[[T2]] : i32 to index
// CHECK:           %[[T3:.+]] = memref.load %[[INDICES]][%[[I]], %[[C1]]] : memref<3x2xi32>
// CHECK:           %[[IDX2:.+]] = index_cast %[[T3]] : i32 to index
// CHECK:           %[[ORI:.+]] = memref.load %[[ORIGINAL]][%[[IDX1]], %[[IDX2]]] : memref<4x3xi32>
// CHECK:           %[[ADD:.+]] = addi %[[ORI]], %[[T1]] : i32
// CHECK:           memref.store %[[ADD]], %[[ORIGINAL]][%[[IDX1]], %[[IDX2]]]

// -----

func @scatter_update_slice_2D(
    %original: memref<4x3xi32>, %indices: memref<2x1xi32>,
    %updates: memref<2x3xi32>) {
  linalg_ext.scatter
    ins(%updates, %indices : memref<2x3xi32>, memref<2x1xi32>)
    outs(%original : memref<4x3xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK:       func @scatter_update_slice_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = constant 2 : index
// CHECK-DAG:     %[[C3:.+]] = constant 3 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:           scf.for %[[J:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:             %[[UPDATE:.+]] = memref.load %[[UPDATES]][%[[I]], %[[J]]]
// CHECK:             %[[INDEX:.+]] = memref.load %[[INDICES]][%[[I]], %[[C0]]]
// CHECK:             %[[LOC:.+]] = index_cast %[[INDEX]] : i32 to index
// CHECK:             memref.store %[[UPDATE]], %[[ORIGINAL]][%[[LOC]], %[[J]]]
// CHECK:           }
// CHECK:         }

// -----

func @scatter_add_scalar_1D(
    %original: memref<8xi32>, %indices: memref<3x1xi32>,
    %updates: memref<3xi32>) {
  linalg_ext.scatter
    ins(%updates, %indices : memref<3xi32>, memref<3x1xi32>)
    outs(%original : memref<8xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = addi %arg1, %arg0 : i32
    linalg_ext.yield %0 : i32
  }
  return
}
// CHECK-LABEL: func @scatter_add_scalar_1D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = constant 3 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:           %[[T1:.+]] = memref.load %[[UPDATES]][%[[I]]] : memref<3xi32>
// CHECK:           %[[T2:.+]] =  memref.load %[[INDICES]][%[[I]], %[[C0]]] : memref<3x1xi32>
// CHECK:           %[[IDX:.+]] = index_cast %[[T2]] : i32 to index
// CHECK:           %[[ORI:.+]] = memref.load %[[ORIGINAL]][%[[IDX]]] : memref<8xi32>
// CHECK:           %[[ADD:.+]] = addi %[[ORI]], %[[T1]] : i32
// CHECK:           memref.store %[[ADD]], %[[ORIGINAL]][%[[IDX]]]

// -----

func @scatter_add_slice_2D(
    %original: memref<4x3xi32>, %indices: memref<2x1xi32>,
    %updates: memref<2x3xi32>) {
  linalg_ext.scatter
    ins(%updates, %indices : memref<2x3xi32>, memref<2x1xi32>)
    outs(%original : memref<4x3xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = addi %arg1, %arg0 : i32
    linalg_ext.yield %0 : i32
  }
  return
}
// CHECK:       func @scatter_add_slice_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = constant 2 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:           scf.for %[[J:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:             %[[UPDATEVAL:.+]] = memref.load %[[UPDATES]][%[[I]], %[[J]]]
// CHECK:             %[[INDEXVAL:.+]] = memref.load %[[INDICES]][%[[I]], %[[C0]]]
// CHECK:             %[[INDEX:.+]] = index_cast %[[INDEXVAL]] : i32 to index
// CHECK:             %[[ORIGINALVAL:.+]] = memref.load %[[ORIGINAL]][%[[INDEX]], %[[J]]]
// CHECK:             %[[STOREVAL:.+]] = addi %[[ORIGINALVAL]], %[[UPDATEVAL]]
// CHECK:             memref.store %[[STOREVAL]], %[[ORIGINAL]][%[[INDEX]], %[[J]]]

// -----

func @scatter_update_scalar_dynamic_1D(
    %original: memref<?xi32>, %indices: memref<?x1xi32>,
    %updates: memref<?xi32>) {
  linalg_ext.scatter
    ins(%updates, %indices : memref<?xi32>, memref<?x1xi32>)
    outs(%original : memref<?xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK-LABEL: func @scatter_update_scalar_dynamic_1D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[UB:.+]] = memref.dim %[[UPDATES]], %[[C0]] : memref<?xi32>
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[UB]] step %[[C1]] {
// CHECK:           %[[T1:.+]] = memref.load %[[UPDATES]][%[[I]]] : memref<?xi32>
// CHECK:           %[[T2:.+]] =  memref.load %[[INDICES]][%[[I]], %[[C0]]] : memref<?x1xi32>
// CHECK:           %[[IDX:.+]] = index_cast %[[T2]] : i32 to index
// CHECK:           memref.store %[[T1]], %[[ORIGINAL]][%[[IDX]]]

// -----

func @scatter_add_scalar_dynamic_2D(
    %original: memref<?x?xi32>, %indices: memref<?x2xi32>,
    %updates: memref<?xi32>) {
  linalg_ext.scatter
    ins(%updates, %indices : memref<?xi32>, memref<?x2xi32>)
    outs(%original : memref<?x?xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = addi %arg1, %arg0 : i32
    linalg_ext.yield %0 : i32
  }
  return
}
// CHECK-LABEL: func @scatter_add_scalar_dynamic_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[UB:.+]] = memref.dim %[[UPDATES]], %[[C0]] : memref<?xi32>
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[UB]] step %[[C1]] {
// CHECK:           %[[T1:.+]] = memref.load %[[UPDATES]][%[[I]]] : memref<?xi32>
// CHECK:           %[[T2:.+]] = memref.load %[[INDICES]][%[[I]], %[[C0]]] : memref<?x2xi32>
// CHECK:           %[[IDX1:.+]] = index_cast %[[T2]] : i32 to index
// CHECK:           %[[T3:.+]] = memref.load %[[INDICES]][%[[I]], %[[C1]]] : memref<?x2xi32>
// CHECK:           %[[IDX2:.+]] = index_cast %[[T3]] : i32 to index
// CHECK:           %[[ORI:.+]] = memref.load %[[ORIGINAL]][%[[IDX1]], %[[IDX2]]] : memref<?x?xi32>
// CHECK:           %[[ADD:.+]] = addi %[[ORI]], %[[T1]] : i32
// CHECK:           memref.store %[[ADD]], %[[ORIGINAL]][%[[IDX1]], %[[IDX2]]]

// -----

func @scatter_update_slice_dynamic_2D(
    %original: memref<?x?xi32>, %indices: memref<?x1xi32>,
    %updates: memref<?x?xi32>) {
  linalg_ext.scatter
    ins(%updates, %indices : memref<?x?xi32>, memref<?x1xi32>)
    outs(%original : memref<?x?xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    linalg_ext.yield %arg0 : i32
  }
  return
}
// CHECK:       func @scatter_update_slice_dynamic_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[UB1:.+]] = memref.dim %[[UPDATES]], %[[C0]] : memref<?x?xi32>
// CHECK-DAG:     %[[UB2:.+]] = memref.dim %[[UPDATES]], %[[C1]] : memref<?x?xi32>
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[UB1]] step %[[C1]] {
// CHECK:           scf.for %[[J:.+]] = %[[C0]] to %[[UB2]] step %[[C1]] {
// CHECK:             %[[UPDATEVAL:.+]] = memref.load %[[UPDATES]][%[[I]], %[[J]]]
// CHECK:             %[[INDEXVAL:.+]] = memref.load %[[INDICES]][%[[I]], %[[C0]]]
// CHECK:             %[[INDEX:.+]] = index_cast %[[INDEXVAL]] : i32 to index
// CHECK:             memref.store %[[UPDATEVAL]], %[[ORIGINAL]][%[[INDEX]], %[[J]]]

// -----

func @fft_1D(%real: memref<16xf32>, %imag: memref<16xf32>) {
  %stage = constant 1 : index
  linalg_ext.fft
    ins(%stage: index)
    outs(%real, %imag: memref<16xf32>, memref<16xf32>)
  return
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0)>
// CHECK:       func @fft_1D
// CHECK-SAME:    %[[REAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[IMAG:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C16:.+]] = constant 16 : index
// CHECK-DAG:     %[[SCALE:.+]] = constant -6.28318548 : f32
// CHECK-DAG:     %[[NODE_RNG:.+]] = shift_left %[[C1]], %[[C1]] : index
// CHECK:         scf.for %[[K:.+]] = %[[C0]] to %[[C16]] step %[[NODE_RNG]]
// CHECK-DAG:       %[[M:.+]] = shift_left %[[C1]], %[[C1]] : index
// CHECK-DAG:       %[[HM:.+]] = shift_right_signed %[[M]], %[[C1]] : index
// CHECK:           %[[L_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[K]]] [%[[HM]]] [1]
// CHECK:           %[[L_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[K]]] [%[[HM]]] [1]
// CHECK:           %[[R_OFFSET:.+]] = addi %[[K]], %[[HM]] : index
// CHECK:           %[[R_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[R_OFFSET]]] [%[[HM]]] [1]
// CHECK:           %[[R_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[R_OFFSET]]] [%[[HM]]] [1]
// CHECK:           %[[M_I32:.+]] = index_cast %[[M]] : index to i32
// CHECK:           %[[M_F32:.+]] = sitofp %[[M_I32]] : i32 to f32
// CHECK:           %[[COEFF:.+]] = divf %[[SCALE]], %[[M_F32]]
// CHECK:           linalg.generic
// CHECK-SAME:        indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:        iterator_types = ["parallel"]
// CHECK-SAME:        outs(%[[L_REAL_SLICE]], %[[L_IMAG_SLICE]], %[[R_REAL_SLICE]], %[[R_IMAG_SLICE]]
// CHECK:           ^bb0(%[[L_REAL:.+]]: f32, %[[L_IMAG:.+]]: f32, %[[R_REAL:.+]]: f32, %[[R_IMAG:.+]]: f32)
//
//                    Compute exp coeff.
// CHECK:             %[[J_IDX:.+]] = linalg.index 0 : index
// CHECK:             %[[J_I32:.+]] = index_cast %[[J_IDX]] : index to i32
// CHECK:             %[[J_F32:.+]] = sitofp %[[J_I32]] : i32 to f32
// CHECK:             %[[EXP_COEF:.+]] = mulf %[[COEFF]], %[[J_F32]] : f32
// CHECK:             %[[W_REAL:.+]] = math.cos %[[EXP_COEF]]
// CHECK:             %[[W_IMAG:.+]] = math.sin %[[EXP_COEF]]
//
//                    Compute "t = w * a[k + j + mh]" by expanding
//                      (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
// CHECK-DAG:         %[[XU:.+]] = mulf %[[W_REAL]], %[[R_REAL]]
// CHECK-DAG:         %[[YV:.+]] = mulf %[[W_IMAG]], %[[R_IMAG]]
// CHECK-DAG:         %[[XV:.+]] = mulf %[[W_REAL]], %[[R_IMAG]]
// CHECK-DAG:         %[[YU:.+]] = mulf %[[W_IMAG]], %[[R_REAL]]
// CHECK:             %[[T_REAL:.+]] = subf %[[XU]], %[[YV]]
// CHECK:             %[[T_IMAG:.+]] = addf %[[XV]], %[[YU]]
//
//                    Compute the results.
//                      u = a[k + j];
//                      a[k + j] = u + t;
//                      a[k + j + mh] = u - t;
// CHECK:             %[[RES1:.+]] = addf %[[L_REAL]], %[[T_REAL]]
// CHECK:             %[[RES2:.+]] = addf %[[L_IMAG]], %[[T_IMAG]]
// CHECK:             %[[RES3:.+]] = subf %[[L_REAL]], %[[T_REAL]]
// CHECK:             %[[RES4:.+]] = subf %[[L_IMAG]], %[[T_IMAG]]
// CHECK:             linalg.yield %[[RES1]], %[[RES2]], %[[RES3]], %[[RES4]]

// -----

func @fft_2D(%real: memref<?x16xf32>, %imag: memref<?x16xf32>) {
  %stage = constant 2 : index
  linalg_ext.fft
    ins(%stage: index)
    outs(%real, %imag: memref<?x16xf32>, memref<?x16xf32>)
  return
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 16 + s0 + d1)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:       func @fft_2D
// CHECK-SAME:    %[[REAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[IMAG:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = constant 2 : index
// CHECK-DAG:     %[[D0:.+]] = memref.dim %[[REAL]], %[[C0]] : memref<?x16xf32>
// CHECK-DAG:     %[[NODE_RNG:.+]] = shift_left %[[C1]], %[[C2]] : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[D0]] step %[[C1]]
// CHECK:           scf.for %[[K:.+]] = %[[C0]] to %[[C16]] step %[[NODE_RNG]]
// CHECK-DAG:         %[[M:.+]] = shift_left %[[C1]], %[[C2]] : index
// CHECK-DAG:         %[[HM:.+]] = shift_right_signed %[[M]], %[[C1]] : index
// CHECK:             %[[L_REAL_SLICE:.+]] = memref.subview %[[REAL]][%[[I]], %[[K]]] [1, %[[HM]]] [1, 1]
// CHECK:             %[[L_IMAG_SLICE:.+]] = memref.subview %[[IMAG]][%[[I]], %[[K]]] [1, %[[HM]]] [1, 1]
// CHECK:             %[[R_OFFSET:.+]] = addi %[[K]], %[[HM]] : index
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
