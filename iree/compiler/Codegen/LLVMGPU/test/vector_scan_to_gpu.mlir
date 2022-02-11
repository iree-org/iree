// RUN: iree-opt -iree-llvmgpu-scan-to-gpu %s | FileCheck %s

func @scan1d(%arg0: memref<32xi32>, %arg1: memref<i32>, %arg2: memref<32xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = vector.transfer_read %arg0[%c0], %c0_i32 {in_bounds = [true]} : memref<32xi32>, vector<32xi32>
  %1 = vector.transfer_read %arg1[], %c0_i32 : memref<i32>, vector<i32>
  %dest, %acc = vector.scan <add>, %0, %1 {inclusive = true, reduction_dim = 0 : i64} : vector<32xi32>, vector<i32>
  vector.transfer_write %dest, %arg2[%c0] {in_bounds = [true]} : vector<32xi32>, memref<32xi32>
  vector.transfer_write %acc, %arg1[] : vector<i32>, memref<i32>
  return
}

// CHECK:      #map = affine_map<()[s0] -> (s0 mod 32)>
// CHECK:      func @scan1d
// CHECK-SAME:   %[[ARG0:.+]]: memref<32xi32>,
// CHECK-SAME:   %[[ARG1:.+]]: memref<i32>,
// CHECK-SAME:   %[[ARG2:.+]]: memref<32xi32>
// CHECK:        %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK:        %[[C32_I32:.+]] = arith.constant 32 : i32
// CHECK:        %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK:        %[[C2_I32:.+]] = arith.constant 2 : i32
// CHECK:        %[[C4_I32:.+]] = arith.constant 4 : i32
// CHECK:        %[[C8_I32:.+]] = arith.constant 8 : i32
// CHECK:        %[[C16_I32:.+]] = arith.constant 16 : i32
// CHECK:        %[[C31_I32:.+]] = arith.constant 31 : i32
// CHECK:        %[[A0:.+]] = gpu.thread_id  x
// CHECK:        %[[A1:.+]] = affine.apply #map()[%[[A0]]]
// CHECK:        %[[A2:.+]] = vector.transfer_read %[[ARG0]][%[[A1]]], %[[C0_I32]] {in_bounds = [true]} : memref<32xi32>, vector<1xi32>
// CHECK:        %[[A3:.+]] = vector.transfer_read %[[ARG1]][], %[[C0_I32]] : memref<i32>, vector<i32>
// CHECK:        %[[A4:.+]] = vector.extract %[[A2]][0] : vector<1xi32>
// CHECK:        %[[A5:.+]] = vector.extractelement %[[A3]][] : vector<i32>
// CHECK:        %[[RESULT:.+]], %[[VALID:.+]] = gpu.shuffle  up %[[A4]], %[[C1_I32]], %[[C32_I32]] : i32
// CHECK:        %[[A6:.+]] = select %[[VALID]], %[[RESULT]], %[[C0_I32]] : i32
// CHECK:        %[[A7:.+]] = arith.addi %[[A4]], %[[A6]] : i32
// CHECK:        %[[RESULT_0:.+]], %[[VALID_1:.+]] = gpu.shuffle  up %[[A7]], %[[C2_I32]], %[[C32_I32]] : i32
// CHECK:        %[[A8:.+]] = select %[[VALID_1]], %[[RESULT_0]], %[[C0_I32]] : i32
// CHECK:        %[[A9:.+]] = arith.addi %[[A7]], %[[A8]] : i32
// CHECK:        %[[RESULT_2:.+]], %[[VALID_3:.+]] = gpu.shuffle  up %[[A9]], %[[C4_I32]], %[[C32_I32]] : i32
// CHECK:        %[[A10:.+]] = select %[[VALID_3]], %[[RESULT_2]], %[[C0_I32]] : i32
// CHECK:        %[[A11:.+]] = arith.addi %[[A9]], %[[A10]] : i32
// CHECK:        %[[RESULT_4:.+]], %[[VALID_5:.+]] = gpu.shuffle  up %[[A11]], %[[C8_I32]], %[[C32_I32]] : i32
// CHECK:        %[[A12:.+]] = select %[[VALID_5]], %[[RESULT_4]], %[[C0_I32]] : i32
// CHECK:        %[[A13:.+]] = arith.addi %[[A11]], %[[A12]] : i32
// CHECK:        %[[RESULT_6:.+]], %[[VALID_7:.+]] = gpu.shuffle  up %[[A13]], %[[C16_I32]], %[[C32_I32]] : i32
// CHECK:        %[[A14:.+]] = select %[[VALID_7]], %[[RESULT_6]], %[[C0_I32]] : i32
// CHECK:        %[[A15:.+]] = arith.addi %[[A13]], %[[A14]] : i32
// CHECK:        %[[A16:.+]] = arith.addi %[[A5]], %[[A15]] : i32
// CHECK:        %[[A17:.+]] = vector.broadcast %[[A16]] : i32 to vector<1xi32>
// CHECK:        %[[RESULT_8:.+]], %[[VALID_9:.+]] = gpu.shuffle  idx %[[A16]], %[[C31_I32]], %[[C32_I32]] : i32
// CHECK:        %[[A18:.+]] = vector.broadcast %[[RESULT_8]] : i32 to vector<i32>
// CHECK:        %[[A19:.+]] = affine.apply #map()[%[[A0]]]
// CHECK:        vector.transfer_write %[[A17]], %[[ARG2]][%[[A19]]] {in_bounds = [true]} : vector<1xi32>, memref<32xi32>
// CHECK:        vector.transfer_write %[[A18]], %[[ARG1]][] : vector<i32>, memref<i32>
// CHECK:        return
// CHECK:      }
// CHECK:    }
