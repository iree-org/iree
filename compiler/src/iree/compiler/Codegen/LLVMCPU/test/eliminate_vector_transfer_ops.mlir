// RUN: iree-opt %s --iree-llvmcpu-eliminate-vector-transfer-pass --cse -canonicalize --split-input-file | FileCheck %s

func.func @vec_write_read(%a: vector<1x7x16xf32>, %b: tensor<1x1x7x16xf32>) -> vector<1x1x7x16xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %1 = vector.transfer_write %a, %b[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x7x16xf32>, tensor<1x1x7x16xf32>
    %2 = vector.transfer_read %1[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x7x16xf32>, vector<1x1x7x16xf32>
    return %2 : vector<1x1x7x16xf32>
}

// CHECK: func.func @vec_write_read(%[[A:.+]]: vector<1x7x16xf32>, %[[B:.+]]: tensor<1x1x7x16xf32>)
// CHECK: %[[EXTRACT:.+]] = vector.extract %[[A]][0] : vector<1x7x16xf32>
// CHECK: %[[BROADCAST:.+]] = vector.broadcast %[[EXTRACT]] : vector<7x16xf32> to vector<1x1x7x16xf32>
// CHECK: return %[[BROADCAST]]
