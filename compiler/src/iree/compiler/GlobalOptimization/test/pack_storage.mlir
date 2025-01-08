// RUN: iree-opt --split-input-file --allow-unregistered-dialect \
// RUN:          --pass-pipeline="builtin.module(util.func(iree-global-opt-pack-storage))" \
// RUN:          %s | FileCheck %s


func.func @addition(%arg0 : tensor<4xi8>, %arg1 : tensor<4xi8>, %unused : tensor<4xi1>) -> tensor<4xi8> {
  %0 = arith.xori %arg0, %arg1 : tensor<4xi8>
  return %0 : tensor<4xi8>
}

