// RUN: iree-opt -split-input-file -iree-hlo-reduction-to-linalg %s | IreeFileCheck %s

// CHECK: [[MAP0:#.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: [[MAP1:#.*]] = affine_map<(d0, d1) -> (0)>
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
  // CHECK-SAME: %{{.*}}, %{{.*}}, %{{.*}} {
  // CHECK-NEXT: ^{{.+}}({{%.*}}, [[IDX:%.*]]: index, [[SRC:%.*]]: f32, [[INIT:%.*]]: f32, [[DST:%.*]]: f32):
  //      CHECK:   [[COND:%.*]] = cmpi
  // CHECK-NEXT:   [[OPERAND:%.*]] = select [[COND]], [[INIT]], [[DST]] : f32
  // CHECK-NEXT:   [[RES:%.*]] = addf [[OPERAND]], [[SRC]] : f32
  // CHECK-NEXT:   linalg.yield [[RES]] : f32
  // CHECK-NEXT: }: memref<5x4xf32>, memref<f32>, memref<4xf32>
  func @reduction_entry(memref<5x4xf32>, memref<f32>, memref<4xf32>)
  attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_apply, iree.executable.reduction.dimension = 1 : i32, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<[4, 5, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32}

  func @reduction_apply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}

// -----

module {
  //      CHECK:   [[COND:%.*]] = cmpf "olt", {{%.*}}, {{%.*}} : f32
  // CHECK-NEXT:   {{%.*}} = select [[COND]], {{%.*}}, {{%.*}} : f32
  func @reduction_entry(memref<5x4xf32>, memref<f32>, memref<4xf32>)
  attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_apply, iree.executable.reduction.dimension = 1 : i32}

  func @reduction_apply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = xla_hlo.min %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}

// -----

module {
  //      CHECK:   [[COND:%.*]] = cmpf "ogt", {{%.*}}, {{%.*}} : f32
  // CHECK-NEXT:   {{%.*}} = select [[COND]], {{%.*}}, {{%.*}} : f32
  func @reduction_entry(memref<5x4xf32>, memref<f32>, memref<4xf32>)
  attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_apply, iree.executable.reduction.dimension = 1 : i32}

  func @reduction_apply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = xla_hlo.max %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
