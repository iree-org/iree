// RUN: iree-opt --split-input-file --iree-stablehlo-preprocessing-unfuse-batch-norm \
// RUN:   --cse --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @batchNormInference_2D_inner_features(
// CHECK-SAME: %[[X:[^:]+]]
// CHECK-SAME: %[[SCALE:[^:]+]]
// CHECK-SAME: %[[OFFSET:[^:]+]]
// CHECK-SAME: %[[MEAN:[^:]+]]
// CHECK-SAME: %[[VARIANCE:[^:]+]]
func.func @batchNormInference_2D_inner_features(
    %x: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<4x256xf32>) {
  // CHECK-DAG: %[[EPS:.+]] = stablehlo.constant dense<1.001000e-05> : tensor<f32>
  // CHECK-DAG: %[[EPS_BCAST:.+]] = stablehlo.broadcast_in_dim %[[EPS]], dims = [] : (tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = stablehlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = stablehlo.sqrt %[[VARIANCE_EPS]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = stablehlo.broadcast_in_dim %[[STDDEV]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = stablehlo.broadcast_in_dim %[[SCALE]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[MEAN_BCAST:.+]] = stablehlo.broadcast_in_dim %[[MEAN]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = stablehlo.subtract %[[X]], %[[MEAN_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[X_SCALED:.+]] = stablehlo.multiply %[[X_CENTER]], %[[SCALE_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = stablehlo.divide %[[X_SCALED]], %[[STDDEV_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[RESULT:.+]] = stablehlo.add %[[X_NORMED]], %[[OFFSET_BCAST]] : tensor<4x256xf32>
  %0 = "stablehlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: return %[[RESULT]]
  func.return %0 : tensor<4x256xf32>
}

// -----
// CHECK-LABEL: @batchNormTraining_2D_inner_features(
// CHECK-SAME: %[[X:[^:]+]]:
// CHECK-SAME: %[[SCALE:[^:]+]]:
// CHECK-SAME: %[[OFFSET:[^:]+]]:
func.func @batchNormTraining_2D_inner_features(
    %x: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>)
    -> (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>) {
  // CHECK-DAG: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[EPS:.+]] = stablehlo.constant dense<1.001000e-05> : tensor<f32>
  // CHECK-DAG: %[[EPS_BCAST:.+]] = stablehlo.broadcast_in_dim %[[EPS]], dims = [] : (tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[NUM_REDUCE:.+]] = stablehlo.constant dense<4.000000e+00> : tensor<256xf32>
  // CHECK-DAG: %[[SumX:.+]] = stablehlo.reduce(%[[X]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0] : (tensor<4x256xf32>, tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[X2:.+]] = stablehlo.multiply %[[X]], %[[X]] : tensor<4x256xf32>
  // CHECK-DAG: %[[SumX2:.+]] = stablehlo.reduce(%[[X2]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0] : (tensor<4x256xf32>, tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[EX:.+]] = stablehlo.divide %[[SumX]], %[[NUM_REDUCE]] : tensor<256xf32>
  // CHECK-DAG: %[[EX2:.+]] = stablehlo.divide %[[SumX2]], %[[NUM_REDUCE]] : tensor<256xf32>
  // CHECK-DAG: %[[E2X:.+]] = stablehlo.multiply %[[EX]], %[[EX]] : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE:.+]] = stablehlo.subtract %[[EX2]], %[[E2X]] : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = stablehlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = stablehlo.sqrt %[[VARIANCE_EPS]] : tensor<256xf32>
  // CHECK-DAG: %[[EX_BCAST:.+]] = stablehlo.broadcast_in_dim %[[EX]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = stablehlo.subtract %[[X]], %[[EX_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = stablehlo.broadcast_in_dim %[[STDDEV]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = stablehlo.divide %[[X_CENTER]], %[[STDDEV_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = stablehlo.broadcast_in_dim %[[SCALE]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED_SCALED:.+]] = stablehlo.multiply %[[X_NORMED]], %[[SCALE_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED_SCALED_OFFSET:.+]] = stablehlo.add %[[X_NORMED_SCALED]], %[[OFFSET_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: return %[[X_NORMED_SCALED_OFFSET]], %[[EX]], %[[VARIANCE]] : tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>
  %0:3 = "stablehlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>
}

// -----
// CHECK-LABEL: @batchNormInference_4D_middle_features
// Just validate that one of the broadcasts happens correctly and rely on
// the verifier to enforce the rest.
// CHECK-SAME: %[[X:[^:]+]]
// CHECK-SAME: %[[SCALE:[^:]+]]
// CHECK-DAG: %[[SCALE_BCAST:.+]] = stablehlo.broadcast_in_dim %[[SCALE]], dims = [2] : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
func.func @batchNormInference_4D_middle_features(
    %x: tensor<3x4x256x6xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<3x4x256x6xf32>) {
  %0 = "stablehlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 2 : i64} :
      (tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<3x4x256x6xf32>
  func.return %0 : tensor<3x4x256x6xf32>
}

// -----
// CHECK-LABEL: @batchNormTraining_4D_middle_features
// CHECK-SAME: %[[X:[^:]+]]
// CHECK-SAME: %[[SCALE:[^:]+]]
// CHECK-SAME: %[[OFFSET:[^:]+]]
func.func @batchNormTraining_4D_middle_features(
    %x: tensor<3x4x256x6xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>)
    -> (tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>) {
  // CHECK-DAG: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[EPS:.+]] = stablehlo.constant dense<1.001000e-05> : tensor<f32>
  // CHECK-DAG: %[[EPS_BCAST:.+]] = stablehlo.broadcast_in_dim %[[EPS]], dims = [] : (tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[NUM_REDUCE:.+]] = stablehlo.constant dense<7.200000e+01> : tensor<256xf32>
  // CHECK-DAG: %[[SumX:.+]] = stablehlo.reduce(%[[X]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0, 1, 3] : (tensor<3x4x256x6xf32>, tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[X2:.+]] = stablehlo.multiply %[[X]], %[[X]] : tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[SumX2:.+]] = stablehlo.reduce(%[[X2]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0, 1, 3] : (tensor<3x4x256x6xf32>, tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[EX:.+]] = stablehlo.divide %[[SumX]], %[[NUM_REDUCE]] : tensor<256xf32>
  // CHECK-DAG: %[[EX2:.+]] = stablehlo.divide %[[SumX2]], %[[NUM_REDUCE]] : tensor<256xf32>
  // CHECK-DAG: %[[E2X:.+]] = stablehlo.multiply %[[EX]], %[[EX]] : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE:.+]] = stablehlo.subtract %[[EX2]], %[[E2X]] : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = stablehlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = stablehlo.sqrt %[[VARIANCE_EPS]] : tensor<256xf32>
  // CHECK-DAG: %[[EX_BCAST:.+]] = stablehlo.broadcast_in_dim %[[EX]], dims = [2] : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = stablehlo.subtract %[[X]], %[[EX_BCAST]] : tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = stablehlo.broadcast_in_dim %[[STDDEV]], dims = [2] : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = stablehlo.divide %[[X_CENTER]], %[[STDDEV_BCAST]] : tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = stablehlo.broadcast_in_dim %[[SCALE]], dims = [2] : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[X_NORMED_SCALED:.+]] = stablehlo.multiply %[[X_NORMED]], %[[SCALE_BCAST]] : tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [2] : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[X_NORMED_SCALED_OFFSET:.+]] = stablehlo.add %[[X_NORMED_SCALED]], %[[OFFSET_BCAST]] : tensor<3x4x256x6xf32>
  // CHECK-DAG: return %[[X_NORMED_SCALED_OFFSET]], %[[EX]], %[[VARIANCE]] : tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>
  %0:3 = "stablehlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 1.001000e-05 : f32, feature_index = 2 : i64} :
      (tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>
}

// -----
// CHECK-LABEL: @batchNormInference_f64
// Validate that epsilon is properly promoted to f64
// CHECK-DAG: %[[EPS:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
func.func @batchNormInference_f64(
    %x: tensor<4x256xf64>, %scale: tensor<256xf64>, %offset: tensor<256xf64>,
    %mean: tensor<256xf64>, %variance: tensor<256xf64>)
    -> (tensor<4x256xf64>) {
  %0 = "stablehlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.0 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf64>, tensor<256xf64>, tensor<256xf64>, tensor<256xf64>,
        tensor<256xf64>) -> tensor<4x256xf64>
  func.return %0 : tensor<4x256xf64>
}

// -----
// CHECK-LABEL: @batchNormTraining_f64
// Validate that epsilon is properly promoted to f64
// CHECK-DAG: %[[EPS:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
func.func @batchNormTraining_f64(
    %x: tensor<4x256xf64>, %scale: tensor<256xf64>, %offset: tensor<256xf64>)
    -> (tensor<4x256xf64>, tensor<256xf64>, tensor<256xf64>) {
  %0:3 = "stablehlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 1.0 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf64>, tensor<256xf64>, tensor<256xf64>) -> (tensor<4x256xf64>, tensor<256xf64>, tensor<256xf64>)
  func.return %0#0, %0#1, %0#2 : tensor<4x256xf64>, tensor<256xf64>, tensor<256xf64>
}

// -----
// CHECK-LABEL: @batchNormInference_f16
// Validate that epsilon is properly down to f16
// CHECK-DAG: %[[EPS:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f16>
func.func @batchNormInference_f16(
    %x: tensor<4x256xf16>, %scale: tensor<256xf16>, %offset: tensor<256xf16>,
    %mean: tensor<256xf16>, %variance: tensor<256xf16>)
    -> (tensor<4x256xf16>) {
  %0 = "stablehlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.0 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>, tensor<256xf16>,
        tensor<256xf16>) -> tensor<4x256xf16>
  func.return %0 : tensor<4x256xf16>
}

// -----
// CHECK-LABEL: @batchNormTraining_f16
// Validate that epsilon is properly down to f16
// CHECK-DAG: %[[EPS:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f16>
func.func @batchNormTraining_f16(
    %x: tensor<4x256xf16>, %scale: tensor<256xf16>, %offset: tensor<256xf16>)
    -> (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>) {
  %0:3 = "stablehlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 1.0 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>) -> (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>)
  func.return %0#0, %0#1, %0#2 : tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>
}

// -----
// CHECK-LABEL: @batchNormInference_f16_overflow
// Validate that epsilon is overflow
func.func @batchNormInference_f16_overflow(
    %x: tensor<4x256xf16>, %scale: tensor<256xf16>, %offset: tensor<256xf16>,
    %mean: tensor<256xf16>, %variance: tensor<256xf16>)
    -> (tensor<4x256xf16>) {
  // expected-warning @+1 {{Could not convert batch_norm epsilon to target fp type: opStatus = 24}}
  %0 = "stablehlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 0.00000001 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>, tensor<256xf16>,
        tensor<256xf16>) -> tensor<4x256xf16>
  func.return %0 : tensor<4x256xf16>
}

// -----
// Validate that epsilon is overflow
func.func @batchNormTraining_f16_overflow(
    %x: tensor<4x256xf16>, %scale: tensor<256xf16>, %offset: tensor<256xf16>)
    -> (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>) {
  // expected-warning @+1 {{Could not convert batch_norm epsilon to target fp type: opStatus = 24}}
  %0:3 = "stablehlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 0.00000001 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>) -> (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>)
  func.return %0#0, %0#1, %0#2 :tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>
}

// -----
// CHECK-LABEL: @batchNormInference_dynamic_shape
// Validate that dynamic shapes are handled properly.
// CHECK-SAME: %[[X:[^:]+]]
// CHECK-SAME: %[[SCALE:[^:]+]]
// CHECK-SAME: %[[OFFSET:[^:]+]]
// CHECK-SAME: %[[MEAN:[^:]+]]
// CHECK-SAME: %[[VARIANCE:[^:]+]]
func.func @batchNormInference_dynamic_shape(
    %x: tensor<?x?x?x?xf32>, %scale: tensor<?xf32>, %offset: tensor<?xf32>,
    %mean: tensor<?xf32>, %variance: tensor<?xf32>)
    -> tensor<?x?x?x?xf32> {
  // CHECK-DAG: %[[EPS:.+]] = stablehlo.constant dense<1.000000e-03> : tensor<f32>
  // CHECK-DAG: %[[VAR_SHAPE:.+]] = shape.shape_of %[[VARIANCE]] : tensor<?xf32> -> tensor<1xindex>
  // CHECK-DAG: %[[EPS_BCAST:.+]] =  stablehlo.dynamic_broadcast_in_dim %[[EPS]], %[[VAR_SHAPE]], dims = [] : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = stablehlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<?xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = stablehlo.sqrt %[[VARIANCE_EPS]] : tensor<?xf32>
  // CHECK-DAG: %[[X_SHAPE:.+]] = shape.shape_of %[[X]] : tensor<?x?x?x?xf32> -> tensor<4xindex>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[STDDEV]], %[[X_SHAPE]], dims = [1] : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[SCALE]], %[[X_SHAPE]], dims = [1] : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[OFFSET]], %[[X_SHAPE]], dims = [1] : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[MEAN_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[MEAN]], %[[X_SHAPE]], dims = [1] : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = stablehlo.subtract %[[X]], %[[MEAN_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_SCALED:.+]] = stablehlo.multiply %[[X_CENTER]], %[[SCALE_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = stablehlo.divide %[[X_SCALED]], %[[STDDEV_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[RESULT:.+]] = stablehlo.add %[[X_NORMED]], %[[OFFSET_BCAST]] : tensor<?x?x?x?xf32>
  %0 = "stablehlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 0.001 : f32, feature_index = 1 : i64} :
      (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
        tensor<?xf32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// -----
// CHECK-LABEL: @batchNormTraining_dynamic_shape
// Validate that dynamic shapes are handled properly.
// CHECK-SAME: %[[X:[^:]+]]
// CHECK-SAME: %[[SCALE:[^:]+]]
// CHECK-SAME: %[[OFFSET:[^:]+]]
func.func @batchNormTraining_dynamic_shape(
    %x: tensor<?x?x?x?xf32>, %scale: tensor<?xf32>, %offset: tensor<?xf32>)
    -> (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>) {
  // CHECK-DAG: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[EPS:.+]] = stablehlo.constant dense<1.001000e-05> : tensor<f32>
  // CHECK-DAG: %[[SCALE_SHAPE:.+]] = shape.shape_of %[[SCALE]] : tensor<?xf32> -> tensor<1xindex>
  // CHECK-DAG: %[[EPS_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[EPS]], %[[SCALE_SHAPE]], dims = [] : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK-DAG: %[[X_SHAPE:.+]] = shape.shape_of %[[X]] : tensor<?x?x?x?xf32> -> tensor<4xindex>
  // CHECK-DAG: %[[X_SIZE:.+]] = shape.num_elements %[[X_SHAPE]] : tensor<4xindex> -> index
  // CHECK-DAG: %[[SCALE_SIZE:.+]] = shape.num_elements %[[SCALE_SHAPE]] : tensor<1xindex> -> index
  // CHECK-DAG: %[[REDUCE_SIZE:.+]] = shape.div %[[X_SIZE]], %[[SCALE_SIZE]] : index, index -> index
  // CHECK-DAG: %[[INDEX_CAST:.+]] = arith.index_cast %[[REDUCE_SIZE]] : index to i64
  // CHECK-DAG: %[[REDUCE_SIZE_TENSOR:.+]] = tensor.from_elements %[[INDEX_CAST]] : tensor<1xi64>
  // CHECK-DAG: %[[REDUCE_SIZE_TENSOR_FP:.+]] = stablehlo.convert %[[REDUCE_SIZE_TENSOR]] : (tensor<1xi64>) -> tensor<1xf32>
  // CHECK-DAG: %[[REDUCE_SIZE_RESHAPE:.+]] = stablehlo.reshape %[[REDUCE_SIZE_TENSOR_FP]] : (tensor<1xf32>) -> tensor<f32>
  // CHECK-DAG: %[[REDUCE_SIZE_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[REDUCE_SIZE_RESHAPE]], %[[SCALE_SHAPE]], dims = [] : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK-DAG: %[[X_SUM:.+]] = stablehlo.reduce(%[[X]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0, 1, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK-DAG: %[[X2:.+]] = stablehlo.multiply %[[X]], %[[X]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X2_SUM:.+]] = stablehlo.reduce(%[[X2]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0, 1, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK-DAG: %[[EX:.+]] = stablehlo.divide %[[X_SUM]], %[[REDUCE_SIZE_BCAST]] : tensor<?xf32>
  // CHECK-DAG: %[[EX2:.+]] = stablehlo.divide %[[X2_SUM]], %[[REDUCE_SIZE_BCAST]] : tensor<?xf32>
  // CHECK-DAG: %[[EX_2:.+]] = stablehlo.multiply %[[EX]], %[[EX]] : tensor<?xf32>
  // CHECK-DAG: %[[VARX:.+]] = stablehlo.subtract %[[EX2]], %[[EX_2]] : tensor<?xf32>
  // CHECK-DAG: %[[VARX_EPS:.+]] = stablehlo.add %[[VARX]], %[[EPS_BCAST]] : tensor<?xf32>
  // CHECK-DAG: %[[STDX:.+]] = stablehlo.sqrt %[[VARX_EPS]] : tensor<?xf32>
  // CHECK-DAG: %[[EX_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[EX]], %[[X_SHAPE]], dims = [2] : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_SUB_EX:.+]] = stablehlo.subtract %[[X]], %[[EX_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[STDX_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[STDX]], %[[X_SHAPE]], dims = [2] : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_CENTOR:.+]] = stablehlo.divide %[[X_SUB_EX]], %[[STDX_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[SCALE]], %[[X_SHAPE]], dims = [2] : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_SCALED:.+]] = stablehlo.multiply %[[X_CENTOR]], %[[SCALE_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[OFFSET]], %[[X_SHAPE]], dims = [2] : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[RESULT:.+]] = stablehlo.add %[[X_SCALED]], %[[OFFSET_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: return %[[RESULT]], %[[EX]], %[[VARX]] : tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>
  %0:3 = "stablehlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 1.001000e-05 : f32, feature_index = 2 : i64} :
      (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>
}
