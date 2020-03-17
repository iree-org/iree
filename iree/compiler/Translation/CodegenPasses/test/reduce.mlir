// RUN: iree-opt -split-input-file -iree-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

// CHECK: [[MAP0:#.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: [[MAP1:#.*]] = affine_map<(d0, d1) -> ()>
// CHECK: [[MAP2:#.*]] = affine_map<(d0, d1) -> (d0)>
module {
  //      CHECK: func @reduction_entry(
  // CHECK-SAME: [[ARG0:%.*]]: memref<5x4xf32>,
  // CHECK-SAME: [[ARG1:%.*]]: memref<f32>,
  // CHECK-SAME: [[ARG2:%.*]]: memref<5xf32>)
  //      CHECK: linalg.indexed_generic {args_in = 2 : i64, args_out = 1 : i64,
  // CHECK-SAME: indexing_maps
  // CHECK-SAME: [[MAP0]], [[MAP1]], [[MAP2]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME: [[ARG0]], [[ARG1]], [[ARG2]] {
  // CHECK-NEXT: ^{{.+}}({{%.*}}, [[IDX:%.*]]: index, [[SRC:%.*]]: f32, [[INIT:%.*]]: f32, [[DST:%.*]]: f32):
  //      CHECK:   [[OPERAND:%.*]] = select {{%.*}}, [[INIT]], [[DST]] : f32
  // CHECK-NEXT:   [[RES:%.*]] = addf [[SRC]], [[OPERAND]] : f32
  // CHECK-NEXT:   linalg.yield [[RES]] : f32
  // CHECK-NEXT: }: memref<5x4xf32>, memref<f32>, memref<5xf32>
  func @reduction_entry(%arg0: memref<5x4xf32>, %arg1: memref<f32>, %arg2: memref<5xf32>)
  attributes {iree.executable.export} {
    %0 = iree.load_input(%arg0 : memref<5x4xf32>) : tensor<5x4xf32>
    %1 = iree.load_input(%arg1 : memref<f32>) : tensor<f32>
    %2 = "xla_hlo.reduce"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = xla_hlo.add %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<5xf32>
    iree.store_output(%2 : tensor<5xf32>, %arg2 : memref<5xf32>)
    return
  }
}

// -----

module {
  //      CHECK:   [[COND:%.*]] = cmpf "olt", {{%.*}}, {{%.*}} : f32
  // CHECK-NEXT:   {{%.*}} = select [[COND]], {{%.*}}, {{%.*}} : f32
  func @reduction_entry(%arg0: memref<5x4xf32>, %arg1: memref<f32>, %arg2: memref<5xf32>)
  attributes {iree.executable.export} {
    %0 = iree.load_input(%arg0 : memref<5x4xf32>) : tensor<5x4xf32>
    %1 = iree.load_input(%arg1 : memref<f32>) : tensor<f32>
    %2 = "xla_hlo.reduce"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = xla_hlo.minimum %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<5xf32>
    iree.store_output(%2 : tensor<5xf32>, %arg2 : memref<5xf32>)
    return
  }
}

// -----

module {
  //      CHECK:   [[COND:%.*]] = cmpf "ogt", {{%.*}}, {{%.*}} : f32
  // CHECK-NEXT:   {{%.*}} = select [[COND]], {{%.*}}, {{%.*}} : f32
  func @reduction_entry(%arg0: memref<5x4xf32>, %arg1: memref<f32>, %arg2: memref<5xf32>)
  attributes {iree.executable.export} {
    %0 = iree.load_input(%arg0 : memref<5x4xf32>) : tensor<5x4xf32>
    %1 = iree.load_input(%arg1 : memref<f32>) : tensor<f32>
    %2 = "xla_hlo.reduce"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = xla_hlo.maximum %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<5xf32>
    iree.store_output(%2 : tensor<5xf32>, %arg2 : memref<5xf32>)
    return
  }
}

// -----

module {
  //      CHECK:   [[COND:%.*]] = cmpf "ogt", {{%.*}}, {{%.*}} : f32
  // CHECK-NEXT:   {{%.*}} = select [[COND]], {{%.*}}, {{%.*}} : f32
  func @reduction_entry(%arg0: memref<5x4xf32>, %arg1: memref<f32>, %arg2: memref<4xf32>)
  attributes {iree.executable.export} {
    %0 = iree.load_input(%arg0 : memref<5x4xf32>) : tensor<5x4xf32>
    %1 = iree.load_input(%arg1 : memref<f32>) : tensor<f32>
    %2 = "xla_hlo.reduce"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = xla_hlo.maximum %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<4xf32>
    iree.store_output(%2 : tensor<4xf32>, %arg2 : memref<4xf32>)
    return
  }
}

// -----

// CHECK: [[MAP0:#.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: [[MAP1:#.*]] = affine_map<(d0, d1) -> ()>
// CHECK: [[MAP2:#.*]] = affine_map<(d0, d1) -> (d0)>
module {
  //      CHECK: func @reduction_entry(
  // CHECK-SAME: [[ARG0:%.*]]: memref<5x4xf32>,
  // CHECK-SAME: [[ARG1:%.*]]: memref<f32>,
  // CHECK-SAME: [[ARG2:%.*]]: memref<4xf32>)
  //      CHECK: linalg.indexed_generic {args_in = 2 : i64, args_out = 1 : i64,
  // CHECK-SAME: indexing_maps
  // CHECK-SAME: [[MAP0]], [[MAP1]], [[MAP2]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME: [[ARG0]], [[ARG1]], [[ARG2]] {
  // CHECK-NEXT: ^{{.+}}({{%.*}}, [[IDX:%.*]]: index, [[SRC:%.*]]: f32, [[INIT:%.*]]: f32, [[DST:%.*]]: f32):
  //      CHECK:   [[OPERAND:%.*]] = select {{%.*}}, [[INIT]], [[DST]] : f32
  // CHECK-NEXT:   [[RES:%.*]] = addf [[SRC]], [[OPERAND]] : f32
  // CHECK-NEXT:   linalg.yield [[RES]] : f32
  // CHECK-NEXT: }: memref<5x4xf32>, memref<f32>, memref<4xf32>
  func @reduction_entry(%arg0: memref<5x4xf32>, %arg1: memref<f32>, %arg2: memref<4xf32>)
  attributes {iree.executable.export} {
    %0 = iree.load_input(%arg0 : memref<5x4xf32>) : tensor<5x4xf32>
    %1 = iree.load_input(%arg1 : memref<f32>) : tensor<f32>
    %2 = "xla_hlo.reduce"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = xla_hlo.add %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<4xf32>
    iree.store_output(%2 : tensor<4xf32>, %arg2 : memref<4xf32>)
    return
  }
}

// -----

module{
  func @reduce_init_const(%arg0: memref<1x10xf32>, %arg1: memref<1xf32>) attributes {iree.executable.export} {
    %0 = iree.load_input(%arg0 : memref<1x10xf32>) : tensor<1x10xf32>
    // CHECK: %[[CST:.*]] = constant 0xFF800000 : f32
    // CHECK: linalg.indexed_generic
    // CHECK-SAME: args_in = 1
    // CHECK-SAME: args_out = 1
    // CHECK: ^{{.*}}(%{{.*}}: index, %[[DIM:.*]]: index, %{{.*}}: f32, %[[OUTPUT:.*]]: f32):
    // CHECK: %[[C0:.*]] = constant 0 : index
    // CHECK: select {{%.*}}, %[[CST]], %[[OUTPUT]] : f32
    %cst = constant dense<0xFF800000> : tensor<f32>
    %1 = "xla_hlo.reduce"(%0, %cst) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>): // no predecessors
      %2 = xla_hlo.add %arg2, %arg3 {name = "maximum.21"} : tensor<f32>
      "xla_hlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    iree.store_output(%1 : tensor<1xf32>, %arg1 : memref<1xf32>)
    return
  }
}

// -----

// CHECK: [[MAP0:#.*]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK: [[MAP1:#.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK: [[MAP2:#.*]] = affine_map<(d0, d1, d2) -> (d0)>
module {
  //      CHECK: func @reduction_multi_dimensions(
  // CHECK-SAME: [[ARG0:%.*]]: memref<5x4x3xf32>,
  // CHECK-SAME: [[ARG1:%.*]]: memref<f32>,
  // CHECK-SAME: [[ARG2:%.*]]: memref<4xf32>)
  //      CHECK: linalg.indexed_generic {args_in = 2 : i64, args_out = 1 : i64,
  // CHECK-SAME: indexing_maps
  // CHECK-SAME: [[MAP0]], [[MAP1]], [[MAP2]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction"]}
  // CHECK-SAME: [[ARG0]], [[ARG1]], [[ARG2]] {
  // CHECK-NEXT: ^{{.+}}({{%.*}}, [[IDX:%.*]]: index, [[SRC:%.*]]: f32, [[INIT:%.*]]: f32, [[DST:%.*]]: f32):
  //      CHECK:   [[TRUE:%.*]] = constant true
  //      CHECK:   [[CMP1:%.*]] = cmpi
  //      CHECK:   [[COND1:%.*]] = and [[TRUE]], [[CMP1]]
  //      CHECK:   [[CMP2:%.*]] = cmpi
  //      CHECK:   [[COND2:%.*]] = and [[COND1]], [[CMP2]]
  // CHECK-NEXT:   [[OPERAND:%.*]] = select [[COND2]], [[INIT]], [[DST]] : f32
  // CHECK-NEXT:   [[RES:%.*]] = addf [[SRC]], [[OPERAND]] : f32
  // CHECK-NEXT:   linalg.yield [[RES]] : f32
  // CHECK-NEXT: }: memref<5x4x3xf32>, memref<f32>, memref<4xf32>
  func @reduction_multi_dimensions(%arg0: memref<5x4x3xf32>, %arg1: memref<f32>, %arg2: memref<4xf32>)
  attributes {iree.executable.export} {
    %0 = iree.load_input(%arg0 : memref<5x4x3xf32>) : tensor<5x4x3xf32>
    %1 = iree.load_input(%arg1 : memref<f32>) : tensor<f32>
    %2 = "xla_hlo.reduce"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = xla_hlo.add %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<5x4x3xf32>, tensor<f32>) -> tensor<4xf32>
    iree.store_output(%2 : tensor<4xf32>, %arg2 : memref<4xf32>)
    return
  }
}
