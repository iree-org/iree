// RUN: iree-opt %s -iree-codegen-linalg-vectorization-pass -iree-codegen-llvm-vector-unroll-shape=1,1,1 -split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @tensor_dispatch_0
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK:   %[[I0:.*]] = flow.dispatch.tensor.load {{.*}} : !flow.dispatch.tensor<readonly:2x3xf32> -> tensor<2x3xf32>
//       CHECK:   %[[I1:.*]] = flow.dispatch.tensor.load {{.*}} : !flow.dispatch.tensor<readonly:3x4xf32> -> tensor<3x1xf32>
//       CHECK:   %[[I2:.*]] = flow.dispatch.tensor.load {{.*}} : !flow.dispatch.tensor<readonly:2x4xf32> -> tensor<2x1xf32>
//       CHECK:   %[[V0:.*]] = vector.transfer_read %[[I0]][%[[C0]], %[[C0]]], {{.*}} : tensor<2x3xf32>, vector<1x1xf32>
//       CHECK:   %[[V1:.*]] = vector.transfer_read %[[I0]][%[[C0]], %[[C1]]], {{.*}} : tensor<2x3xf32>, vector<1x1xf32>
//       CHECK:   %[[V2:.*]] = vector.transfer_read %[[I0]][%[[C0]], %[[C2]]], {{.*}} : tensor<2x3xf32>, vector<1x1xf32>
//       CHECK:   %[[V3:.*]] = vector.transfer_read %[[I0]][%[[C1]], %[[C0]]], {{.*}} : tensor<2x3xf32>, vector<1x1xf32>
//       CHECK:   %[[V4:.*]] = vector.transfer_read %[[I0]][%[[C1]], %[[C1]]], {{.*}} : tensor<2x3xf32>, vector<1x1xf32>
//       CHECK:   %[[V5:.*]] = vector.transfer_read %[[I0]][%[[C1]], %[[C2]]], {{.*}} : tensor<2x3xf32>, vector<1x1xf32>
//       CHECK:   %[[V6:.*]] = vector.transfer_read %[[I1]][%[[C0]], %[[C0]]], {{.*}} : tensor<3x1xf32>, vector<1x1xf32>
//       CHECK:   %[[V7:.*]] = vector.transfer_read %[[I1]][%[[C1]], %[[C0]]], {{.*}} : tensor<3x1xf32>, vector<1x1xf32>
//       CHECK:   %[[V8:.*]] = vector.transfer_read %[[I1]][%[[C2]], %[[C0]]], {{.*}} : tensor<3x1xf32>, vector<1x1xf32>
//       CHECK:   %[[VI0:.*]] = vector.insert_strided_slice %[[V6]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<1x1xf32> into vector<3x1xf32>
//       CHECK:   %[[VI1:.*]] = vector.insert_strided_slice %[[V7]], %[[VI0]] {offsets = [1, 0], strides = [1, 1]} : vector<1x1xf32> into vector<3x1xf32>
//       CHECK:   %[[VI2:.*]] = vector.insert_strided_slice %[[V8]], %[[VI1]] {offsets = [2, 0], strides = [1, 1]} : vector<1x1xf32> into vector<3x1xf32>
//       CHECK:   %[[T:.*]] = vector.transpose %[[VI2]], [1, 0] : vector<3x1xf32> to vector<1x3xf32>
//       CHECK:   %[[V9:.*]] = vector.transfer_read %[[I2]][%[[C0]], %[[C0]]], {{.*}} : tensor<2x1xf32>, vector<1x1xf32>
//       CHECK:   %[[VA:.*]] = vector.transfer_read %[[I2]][%[[C1]], %[[C0]]], {{.*}} : tensor<2x1xf32>, vector<1x1xf32>
//       CHECK:   %[[VE0:.*]] = vector.extract_strided_slice %[[T]] {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x3xf32> to vector<1x1xf32>
//       CHECK:   %[[VE1:.*]] = vector.extract_strided_slice %[[T]] {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x3xf32> to vector<1x1xf32>
//       CHECK:   %[[VE2:.*]] = vector.extract_strided_slice %[[T]] {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x3xf32> to vector<1x1xf32>
//       CHECK:   %[[D0:.*]] = vector.contract {{.*}} %[[V0]], %[[VE0]], %[[V9]] : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//       CHECK:   %[[D1:.*]] = vector.contract {{.*}} %[[V1]], %[[VE1]], %[[D0]] : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//       CHECK:   %[[D2:.*]] = vector.contract {{.*}} %[[V2]], %[[VE2]], %[[D1]] : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//       CHECK:   %[[D3:.*]] = vector.contract {{.*}} %[[V3]], %[[VE0]], %[[VA]] : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//       CHECK:   %[[D4:.*]] = vector.contract {{.*}} %[[V4]], %[[VE1]], %[[D3]] : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//       CHECK:   %[[D5:.*]] = vector.contract {{.*}} %[[V5]], %[[VE2]], %[[D4]] : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//       CHECK:   %[[W0:.*]] = vector.transfer_write %[[D2]], %[[I2]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<1x1xf32>, tensor<2x1xf32>
//       CHECK:   %[[W1:.*]] = vector.transfer_write %[[D5]], %[[W0]][%[[C1]], %[[C0]]] {in_bounds = [true, true]} : vector<1x1xf32>, tensor<2x1xf32>
//       CHECK:   flow.dispatch.tensor.store %[[W1]]

func @tensor_dispatch_0() {
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c1 = constant 1 : index
  %c2 = constant 1 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:2x3xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:3x4xf32>
  %2 = hal.interface.binding.subspan @io::@arg2[%c0] : !flow.dispatch.tensor<readonly:2x4xf32>
  %3 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:2x4xf32>
  %4 = flow.dispatch.tensor.load %0, offsets = [%c0, %c0], sizes = [%c2, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:2x3xf32> -> tensor<2x3xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [%c0, %c0], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:3x4xf32> -> tensor<3x1xf32>
  %6 = flow.dispatch.tensor.load %2, offsets = [%c0, %c0], sizes = [%c2, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:2x4xf32> -> tensor<2x1xf32>
  %7 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%4, %5 : tensor<2x3xf32>, tensor<3x1xf32>) outs(%6 : tensor<2x1xf32>) -> tensor<2x1xf32>
  flow.dispatch.tensor.store %7, %3, offsets = [%c0, %c0], sizes = [%c2, %c1], strides = [%c1, %c1] : tensor<2x1xf32> -> !flow.dispatch.tensor<writeonly:2x4xf32>
  return
}
