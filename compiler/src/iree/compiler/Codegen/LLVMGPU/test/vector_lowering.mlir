// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-lowering))" --split-input-file %s | FileCheck %s

module {
  func.func @broadcast_read_lowering(%arg0: memref<4096x32xf16>) -> vector<1x8xf16> {
    %cst_1 = arith.constant 0.000000e+00 : f16
    %0 = gpu.thread_id  x
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %broadcast_read = vector.transfer_read %arg0[%workgroup_id_x, %0], %cst_1 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, 0)>} : memref<4096x32xf16>, vector<1x8xf16>
    return %broadcast_read : vector<1x8xf16>
  }
}
// CHECK-LABEL: func.func @broadcast_read_lowering
//  CHECK-SAME: (%[[ARG0:.+]]: memref<4096x32xf16>)
//  CHECK: %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<1x8xf16>
//  CHECK: %[[ELEM:.+]] = memref.load %[[ARG0]]{{.*}} : memref<4096x32xf16>
//  CHECK: %[[SPLAT:.+]] = vector.splat %[[ELEM]] : vector<8xf16>
//  CHECK: %[[INSERT:.+]] = vector.insert %[[SPLAT]], %[[INIT]] [0] : vector<8xf16> into vector<1x8xf16>
//  CHECK: return %[[INSERT]]
