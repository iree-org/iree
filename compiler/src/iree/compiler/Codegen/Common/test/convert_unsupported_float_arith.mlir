// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-convert-unsupported-float-arith{source-types=f8E4M3FNUZ,bf16 target-type=f32}))" %s | FileCheck %s

// CHECK-LABEL: func.func @negf_f8_unsupported
// CHECK-SAME: (%[[ARG0:.*]]: f8E4M3FNUZ) -> f8E4M3FNUZ
// CHECK: %[[EXT:.*]] = arith.extf %[[ARG0]] {{.*}} : f8E4M3FNUZ to f32
// CHECK: %[[NEG:.*]] = arith.negf %[[EXT]] : f32
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[NEG]] {{.*}} : f32 to f8E4M3FNUZ
// CHECK: return %[[TRUNC]] : f8E4M3FNUZ
func.func @negf_f8_unsupported(%arg0 : f8E4M3FNUZ) -> f8E4M3FNUZ {
    %0 = arith.negf %arg0 : f8E4M3FNUZ
    return %0 : f8E4M3FNUZ
}

// -----

func.func @bf16_expansion(%x: bf16) -> bf16 {
// CHECK-LABEL: @bf16_expansion
// CHECK-SAME: [[X:%.+]]: bf16
// CHECK-DAG: [[C:%.+]] = arith.constant {{.*}} : bf16
// CHECK-DAG: [[X_EXP:%.+]] = arith.extf [[X]] {{.*}} : bf16 to f32
// CHECK-DAG: [[C_EXP:%.+]] = arith.extf [[C]] {{.*}} : bf16 to f32
// CHECK: [[Y_EXP:%.+]] = arith.addf [[X_EXP]], [[C_EXP]] : f32
// CHECK: [[Y:%.+]] = arith.truncf [[Y_EXP]] {{.*}} : f32 to bf16
// CHECK: return [[Y]]
  %c = arith.constant 1.0 : bf16
  %y = arith.addf %x, %c : bf16
  func.return %y : bf16
}
