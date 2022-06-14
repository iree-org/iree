// RUN: iree-tf-opt --iree-tf-convert-to-mhlo --split-input-file %s | FileCheck %s

// CHECK-LABEL: @sigmoid
func.func @sigmoid(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: [[SCALAR:%.+]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK-DAG: [[SHAPE_OF:%.+]] = shape.shape_of %arg0 : tensor<2xf32> -> tensor<1xindex>
  // CHECK-DAG: [[SHAPE_VAL:%.+]] = shape.to_extent_tensor [[SHAPE_OF]] : tensor<1xindex> -> tensor<1xindex>
  // CHECK-DAG: [[HALF:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[SCALAR]], [[SHAPE_VAL]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<1xindex>) -> tensor<2xf32>
  // CHECK-DAG: [[R1:%.+]] =  mhlo.multiply %arg0, [[HALF]] : tensor<2xf32>
  // CHECK-DAG: [[R2:%.+]] =  mhlo.tanh [[R1]] : tensor<2xf32>
  // CHECK-DAG: [[R3:%.+]] =  mhlo.multiply [[R2]], [[HALF]] : tensor<2xf32>
  // CHECK-DAG: [[R4:%.+]] =  mhlo.add [[R3]], [[HALF]] : tensor<2xf32>
  %0 = "tf.Sigmoid"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: @sigmoid_complex
func.func @sigmoid_complex(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK: [[R0:%.+]] = mhlo.constant dense<(5.000000e-01,0.000000e+00)> : tensor<complex<f32>>
  // CHECK-NOT: tf.Sigmoid
  %0 = "tf.Sigmoid"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  return %0 : tensor<2xcomplex<f32>>
}

// CHECK-LABEL: @sigmoid_unranked
func.func @sigmoid_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-DAG: [[SCALAR:%.+]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK-DAG: [[SHAPE_OF:%.+]] = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  // CHECK-DAG: [[SHAPE_VAL:%.+]] = shape.to_extent_tensor [[SHAPE_OF]] : tensor<?xindex> -> tensor<?xindex>
  // CHECK-DAG: [[HALF:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[SCALAR]], [[SHAPE_VAL]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<?xindex>) -> tensor<*xf32>
  // CHECK-DAG: [[R1:%.+]] =  mhlo.multiply %arg0, [[HALF]] : tensor<*xf32>
  // CHECK-DAG: [[R2:%.+]] =  mhlo.tanh [[R1]] : tensor<*xf32>
  // CHECK-DAG: [[R3:%.+]] =  mhlo.multiply [[R2]], [[HALF]] : tensor<*xf32>
  // CHECK-DAG: [[R4:%.+]] =  mhlo.add [[R3]], [[HALF]] : tensor<*xf32>
  %0 = "tf.Sigmoid"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
