// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors -canonicalize %s | IreeFileCheck %s


// CHECK-DAG:  %[[C0:.+]] = constant 0 : index
// CHECK-DAG:  %[[C1:.+]] = constant 1 : index
// CHECK-DAG:  %[[C2:.+]] = constant 2 : index
// CHECK-DAG:  %[[C3:.+]] = constant 3 : index
// CHECK-DAG:  %[[IN0:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xi32>
// CHECK-DAG:  %[[IN1:.+]] = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2x3xi32>
// CHECK:      %[[INIT:.+]] = linalg.init_tensor [2, 5] : tensor<2x5xi32>
// CHECK:      %[[FILL:.+]] = linalg.fill(%[[INIT]], %{{.*}}) : tensor<2x5xi32>, i32 -> tensor<2x5xi32>
// CHECK:      %[[OUT0:.+]] = subtensor_insert %[[IN0]] into
// CHECK-SAME:   %[[FILL]][%[[C0]], %[[C0]]] [%[[C2]], %[[C2]]] [%[[C1]], %[[C1]]] : tensor<2x2xi32> into tensor<2x5xi32>
// CHECK:      %[[OUT1:.+]] = subtensor_insert %[[IN1]] into
// CHECK-SAME:   %[[OUT0]][%[[C0]], %[[C2]]] [%[[C2]], %[[C3]]] [%[[C1]], %[[C1]]] : tensor<2x3xi32> into tensor<2x5xi32>
module {
  func @concatenate() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xi32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2x3xi32>
    %2 = "mhlo.concatenate"(%0, %1) {
      dimension = 1
    } : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x5xi32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<2x5xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

// CHECK-DAG:  %[[C0:.+]] = constant 0 : index
// CHECK-DAG:  %[[C1:.+]] = constant 1 : index
// CHECK-DAG:  %[[C2:.+]] = constant 2 : index
// CHECK-DAG:  %[[C3:.+]] = constant 3 : index
// CHECK-DAG:  %[[CST:.+]] = constant dense<42> : tensor<3x2xi32>
// CHECK-DAG:  %[[IN:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xi32>
// CHECK:      %[[INIT:.+]] = linalg.init_tensor [5, 2] : tensor<5x2xi32>
// CHECK:      %[[FILL:.+]] = linalg.fill(%[[INIT]], %{{.*}}) : tensor<5x2xi32>, i32 -> tensor<5x2xi32>
// CHECK:      %[[OUT0:.+]] = subtensor_insert %[[IN]] into
// CHECK-SAME:   %[[FILL]][%[[C0]], %[[C0]]] [%[[C2]], %[[C2]]] [%[[C1]], %[[C1]]] : tensor<2x2xi32> into tensor<5x2xi32>
// CHECK:      %[[OUT1:.+]] = subtensor_insert %[[CST]] into
// CHECK-SAME:   %[[OUT0]][%[[C2]], %[[C0]]] [%[[C3]], %[[C2]]] [%[[C1]], %[[C1]]] : tensor<3x2xi32> into tensor<5x2xi32>
module {
  func @concatenate() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xi32>
    %1 = constant dense<42> : tensor<3x2xi32>
    %2 = "mhlo.concatenate"(%0, %1) {
      dimension = 0
    } : (tensor<2x2xi32>, tensor<3x2xi32>) -> tensor<5x2xi32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<5x2xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}
