// RUN: iree-opt -split-input-file -iree-codegen-vectorize-linalg-ops -canonicalize -cse %s | IreeFileCheck %s

func @broadcast_add() {
  %0 = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<4xf32>
  %1 = iree.placeholder for "interface buffer" {binding = @io::@arg1} : memref<3x4xf32>
  %2 = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<3x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>,
                                   affine_map<(d0, d1) -> (d0, d1)>,
                                   affine_map<(d0, d1) -> (d0, d1)>],
                  iterator_types = ["parallel", "parallel"]}
    ins(%0, %1 : memref<4xf32>, memref<3x4xf32>)
   outs(%2 : memref<3x4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
    %3 = addf %arg0, %arg1 : f32
    linalg.yield %3 : f32
  }
  return
}
// CHECK-LABEL: func @broadcast_add
//   CHECK-DAG: %[[BUF0:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<1xvector<4xf32>>
//   CHECK-DAG: %[[BUF1:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg1} : memref<3x1xvector<4xf32>>
//   CHECK-DAG: %[[BUF2:.+]] = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<3x1xvector<4xf32>>
//       CHECK: linalg.generic
//  CHECK-SAME:   ins(%[[BUF0]], %[[BUF1]] :
//  CHECK-SAME:   outs(%[[BUF2]] :
//       CHECK: ^bb0(%[[ARG0:.+]]: vector<4xf32>, %[[ARG1:.+]]: vector<4xf32>, %[[ARG2:.+]]: vector<4xf32>)
//       CHECK:   %[[RES:.+]] = addf %[[ARG0]], %[[ARG1]] : vector<4xf32>
//       CHECK:   linalg.yield %[[RES]] : vector<4xf32>

// -----

func @log_plus_one() {
  %0 = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<4xf32>
  %c0 = constant 0 : index
  %cst = constant 1.000000e+00 : f32
  %1 = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
    ins(%1 : memref<4xf32>)
   outs(%0 : memref<4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
    %2 = addf %arg0, %cst : f32
    %3 = math.log %2 : f32
    linalg.yield %3 : f32
  }
  return
}
// CHECK-LABEL: func @log_plus_one
//   CHECK-DAG: %[[BUF0:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<1xvector<4xf32>>
//   CHECK-DAG: %[[BUF1:.+]] = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<1xvector<4xf32>>
//   CHECK-DAG: %[[CST:.+]] = constant dense<1.000000e+00> : vector<4xf32>
//       CHECK: linalg.generic
//  CHECK-SAME:   ins(%[[BUF0]] :
//  CHECK-SAME:   outs(%[[BUF1]] :
//       CHECK: ^bb0(%[[ARG0:.+]]: vector<4xf32>, %[[ARG1:.+]]: vector<4xf32>)
//       CHECK:   %[[T1:.+]] = addf %[[ARG0]], %[[CST]] : vector<4xf32>
//       CHECK:   %[[T2:.+]] = math.log %[[T1]] : vector<4xf32>
//       CHECK:   linalg.yield %[[T2]] : vector<4xf32>

// -----

func @cmp_and_select() {
  %0 = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<4xi32>
  %1 = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<4xi32>
  %2 = iree.placeholder for "interface buffer" {binding = @io::@arg1} : memref<4xi32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
    ins(%1, %2 : memref<4xi32>, memref<4xi32>)
   outs(%0 : memref<4xi32>) {
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  // no predecessors
    %3 = cmpi sgt, %arg0, %arg1 : i32
    %4 = select %3, %arg0, %arg1 : i32
    linalg.yield %4 : i32
  }
  return
}
// CHECK-LABEL: func @cmp_and_select
//   CHECK-DAG: %[[BUF0:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<1xvector<4xi32>>
//   CHECK-DAG: %[[BUF1:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg1} : memref<1xvector<4xi32>>
//   CHECK-DAG: %[[BUF2:.+]] = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<1xvector<4xi32>>
//       CHECK: linalg.generic
//  CHECK-SAME:   ins(%[[BUF0]], %[[BUF1]] :
//  CHECK-SAME:   outs(%[[BUF2]] :
//       CHECK: ^bb0(%[[ARG0:.+]]: vector<4xi32>, %[[ARG1:.+]]: vector<4xi32>, %[[ARG2:.+]]: vector<4xi32>)
//       CHECK:   %[[T1:.+]] = cmpi sgt, %[[ARG0]], %[[ARG1]] : vector<4xi32>
//       CHECK:   %[[T2:.+]] = select %[[T1]], %[[ARG0]], %[[ARG1]] : vector<4xi1>, vector<4xi32>
//       CHECK:   linalg.yield %[[T2]] : vector<4xi32>

// -----

func @cmp_convert_mul() {
  %1 = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<4xf32>
  %2 = iree.placeholder for "interface buffer" {binding = @io::@arg1} : memref<4xi32>
  %0 = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<4xi32>
  %cst = constant 0.000000e+00 : f32
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                   affine_map<(d0) -> (d0)>,
                                   affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
    ins(%1, %2 : memref<4xf32>, memref<4xi32>)
   outs(%0 : memref<4xi32>) {
  ^bb0(%arg0: f32, %arg1: i32, %arg2: i32):  // no predecessors
    %3 = cmpf oeq, %arg0, %cst : f32
    %4 = zexti %3 : i1 to i32
    %5 = muli %4, %arg1 : i32
    linalg.yield %5 : i32
  }
  return
}
// CHECK-LABEL: func @cmp_convert_mul
//   CHECK-DAG: %[[BUF0:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<1xvector<4xf32>>
//   CHECK-DAG: %[[BUF1:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg1} : memref<1xvector<4xi32>>
//   CHECK-DAG: %[[BUF2:.+]] = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<1xvector<4xi32>>
//       CHECK: linalg.generic
//  CHECK-SAME:   ins(%[[BUF0]], %[[BUF1]] :
//  CHECK-SAME:  outs(%[[BUF2]] :
//       CHECK: ^bb0(%[[ARG0:.+]]: vector<4xf32>, %[[ARG1:.+]]: vector<4xi32>, %[[ARG2:.+]]: vector<4xi32>)
//       CHECK:   %[[T1:.+]] = cmpf oeq, %[[ARG0]], %{{.+}} : vector<4xf32>
//       CHECK:   %[[T2:.+]] = zexti %[[T1]] : vector<4xi1> to vector<4xi32>
//       CHECK:   %[[T3:.+]] = muli %[[T2]], %[[ARG1]] : vector<4xi32>
//       CHECK:   linalg.yield %[[T3]] : vector<4xi32>

// -----

func @not_contiguous() {
  %0 = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<4x4xf32>
  %c0 = constant 0 : index
  %cst = constant 1.000000e+00 : f32
  %1 = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<4x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]}
    ins(%1 : memref<4x4xf32>)
   outs(%0 : memref<4x4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
    %2 = addf %arg0, %cst : f32
    linalg.yield %2 : f32
  }
  return
}
// CHECK-LABEL: func @not_contiguous
//   CHECK-DAG: iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<4x4xf32>
//   CHECK-DAG: iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<4x4xf32>

// -----

func @not_4s() {
  %0 = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<4x3xf32>
  %c0 = constant 0 : index
  %cst = constant 1.000000e+00 : f32
  %1 = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<4x3xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%1 : memref<4x3xf32>)
   outs(%0 : memref<4x3xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
    %2 = addf %arg0, %cst : f32
    linalg.yield %2 : f32
  }
  return
}
// CHECK-LABEL: func @not_4s
//   CHECK-DAG: iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<4x3xf32>
//   CHECK-DAG: iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<4x3xf32>

// -----

// CHECK-LABEL: func @cst
//       CHECK: iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<4xf32>
func @cst() {
  %cst = constant 1.001000e+00 : f32
  %0 = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%0 : memref<4xf32>) {
  ^bb0(%arg0: f32):  // no predecessors
    %1 = math.rsqrt %cst : f32
    linalg.yield %1 : f32
  }
  return
}
