// RUN: iree-opt -iree-test-llvmgpu-scalarize-math-op %s | IreeFileCheck %s

// CHECK-LABEL: func @scalarize
func @scalarize(
  %arg0: vector<3x1x2xf32>,%arg1: vector<2xf32>, %arg2: vector<2xf32>)
  -> (vector<3x1x2xf32>, vector<2xf32>) {
// CHECK: %[[E_0_0:.+]] = vector.extract %{{.*}}[0, 0, 0] : vector<3x1x2xf32>
// CHECK: %[[S_0_0:.+]] = math.sqrt %[[E_0_0]] : f32
// CHECK: vector.insert %[[S_0_0]], %{{.*}} [0, 0, 0] : f32 into vector<3x1x2xf32>
// CHECK: %[[E_1_0:.+]] = vector.extract %{{.*}}[1, 0, 0] : vector<3x1x2xf32>
// CHECK: %[[S_1_0:.+]] = math.sqrt %[[E_1_0]] : f32
// CHECK: vector.insert %[[S_1_0]], %{{.*}} [1, 0, 0] : f32 into vector<3x1x2xf32>
// CHECK: %[[E_2_0:.+]] = vector.extract %{{.*}}[2, 0, 0] : vector<3x1x2xf32>
// CHECK: %[[S_2_0:.+]] = math.sqrt %[[E_2_0]] : f32
// CHECK: vector.insert %[[S_2_0]], %{{.*}} [2, 0, 0] : f32 into vector<3x1x2xf32>
// CHECK: %[[E_0_1:.+]] = vector.extract %{{.*}}[0, 0, 1] : vector<3x1x2xf32>
// CHECK: %[[S_0_1:.+]] = math.sqrt %[[E_0_1]] : f32
// CHECK: vector.insert %[[S_0_1]], %{{.*}} [0, 0, 1] : f32 into vector<3x1x2xf32>
// CHECK: %[[E_1_1:.+]] = vector.extract %{{.*}}[1, 0, 1] : vector<3x1x2xf32>
// CHECK: %[[S_1_1:.+]] = math.sqrt %[[E_1_1]] : f32
// CHECK: vector.insert %[[S_1_1]], %{{.*}} [1, 0, 1] : f32 into vector<3x1x2xf32>
// CHECK: %[[E_2_1:.+]] = vector.extract %{{.*}}[2, 0, 1] : vector<3x1x2xf32>
// CHECK: %[[S_2_1:.+]] = math.sqrt %[[E_2_1]] : f32
// CHECK: vector.insert %[[S_2_1]], %{{.*}} [2, 0, 1] : f32 into vector<3x1x2xf32>
  %0 = math.sqrt %arg0 : vector<3x1x2xf32>
// CHECK: %[[E0:.+]] = vector.extract %{{.*}}[0] : vector<2xf32>
// CHECK: %[[E1:.+]] = vector.extract %{{.*}}[0] : vector<2xf32>
// CHECK: %[[P0:.+]] = math.powf %[[E0]], %[[E1]] : f32
// CHECK: vector.insert %[[P0]], %{{.*}} [0] : f32 into vector<2xf32>
// CHECK: %[[E2:.+]] = vector.extract %{{.*}}[1] : vector<2xf32>
// CHECK: %[[E3:.+]] = vector.extract %{{.*}}[1] : vector<2xf32>
// CHECK: %[[P1:.+]] = math.powf %[[E2]], %[[E3]] : f32
// CHECK: vector.insert %[[P1]], %{{.*}} [1] : f32 into vector<2xf32>
  %1 = math.powf %arg1, %arg2 : vector<2xf32>
  return %0, %1 : vector<3x1x2xf32>, vector<2xf32>
}
