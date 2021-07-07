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
// CHECK-DAG:     %[[UB:.+]] = subi %[[C128]], %c1 : index
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK:           scf.for %[[ARG2:.+]] = %[[C0]] to %[[UB]] step %[[C1]]
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
// CHECK-DAG:     %[[UB:.+]] = subi %[[C16]], %c1 : index
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C16]] step %[[C1]]
// CHECK:           scf.for %[[ARG2:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK:             scf.for %[[ARG3:.+]] = %[[C0]] to %[[UB]] step %[[C1]]
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
// CHECK-DAG:     %[[UB:.+]] = subi %[[C128]], %c1 : index
// CHECK:         scf.for %[[ARG1:.+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK:           scf.for %[[ARG2:.+]] = %[[C0]] to %[[UB]] step %[[C1]]
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
