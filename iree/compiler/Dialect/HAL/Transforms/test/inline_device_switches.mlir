// RUN: iree-opt -split-input-file -iree-hal-inline-device-switches -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @simple_constants
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device
// CHECK-SAME: %[[ARG:.+]]: i32
func @simple_constants(%device : !hal.device, %arg : i32) -> i32 {
  // CHECK-DAG: %[[C0:.+]] = constant 0
  %c0 = constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = constant 1
  %c1 = constant 1 : i32
  // CHECK-DAG: %[[C2:.+]] = constant 2
  %c2 = constant 2 : i32
  // CHECK-DAG: %[[C3:.+]] = constant 3
  // CHECK-DAG: %[[C4:.+]] = constant 4
  %0 = hal.device.switch(%device : !hal.device) -> i32
    // CHECK-NEXT: %[[IS0:.+]] = hal.device.match.id %[[DEVICE]], pattern = ["vulkan-v1.?-*"] : (!hal.device) -> i1
    // CHECK-NEXT: cond_br %[[IS0]], ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:  br ^bb7(%[[C1]] : i32)
    #hal.device.match.id<"vulkan-v1.?-*">(%c1a = %c1 : i32) {
      hal.return %c1a : i32
    },
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:  %[[IS1L:.+]] = hal.device.match.id %arg0, pattern = ["vmla"] : (!hal.device) -> i1
    // CHECK-NEXT:  %[[IS1R:.+]] = hal.device.match.id %arg0, pattern = ["vulkan-*"] : (!hal.device) -> i1
    // CHECK-NEXT:  %[[IS1:.+]] = or %[[IS1L]], %[[IS1R]] : i1
    // CHECK-NEXT:  cond_br %[[IS1]], ^bb3, ^bb6
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:  %[[EQZ:.+]] = cmpi "eq", %[[ARG]], %[[C2]] : i32
    // CHECK-NEXT:  cond_br %[[EQZ]], ^bb4, ^bb5
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT:  br ^bb7(%[[C3]] : i32)
    // CHECK-NEXT: ^bb5:
    // CHECK-NEXT:  br ^bb7(%[[C4]] : i32)
    #hal.match.any<[#hal.device.match.id<"vmla">, #hal.device.match.id<"vulkan-*">]>(%arga = %arg : i32, %c2a = %c2 : i32) {
      %eqz = cmpi "eq", %arga, %c2a : i32
      cond_br %eqz, ^bb_true, ^bb_false
    ^bb_true:
      %c3 = constant 3 : i32
      hal.return %c3 : i32
    ^bb_false:
      %c4 = constant 4 : i32
      hal.return %c4 : i32
    },
    // CHECK-NEXT: ^bb6:
    // CHECK-NEXT:  br ^bb7(%[[C0]] : i32)
    #hal.match.always(%c0b = %c0 : i32) {
      hal.return %c0b : i32
    }
  // CHECK-NEXT: ^bb7(%[[RES:.+]]: i32):
  // CHECK-NEXT: return %[[RES]] : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: @no_results
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device
func @no_results(%device : !hal.device) {
  hal.device.switch(%device : !hal.device)
    // CHECK-NEXT: %[[IS0:.+]] = hal.device.match.id %[[DEVICE]], pattern = ["vulkan-v1.?-*"] : (!hal.device) -> i1
    // CHECK-NEXT: cond_br %[[IS0]], ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:  "some.op_a"()
    // CHECK-NEXT:  br ^bb5
    #hal.device.match.id<"vulkan-v1.?-*">() {
      "some.op_a"() : () -> ()
      hal.return
    },
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:  %[[IS1L:.+]] = hal.device.match.id %arg0, pattern = ["vmla"] : (!hal.device) -> i1
    // CHECK-NEXT:  %[[IS1R:.+]] = hal.device.match.id %arg0, pattern = ["vulkan-*"] : (!hal.device) -> i1
    // CHECK-NEXT:  %[[IS1:.+]] = or %[[IS1L]], %[[IS1R]] : i1
    // CHECK-NEXT:  cond_br %[[IS1]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:  "some.op_b"()
    // CHECK-NEXT:  br ^bb5
    #hal.match.any<[#hal.device.match.id<"vmla">, #hal.device.match.id<"vulkan-*">]>() {
      "some.op_b"() : () -> ()
      hal.return
    },
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT:  "some.op_c"()
    // CHECK-NEXT:  br ^bb5
    #hal.match.always() {
      "some.op_c"() : () -> ()
      hal.return
    }
  // CHECK-NEXT: ^bb5:
  // CHECK-NEXT:  return
  return
}
