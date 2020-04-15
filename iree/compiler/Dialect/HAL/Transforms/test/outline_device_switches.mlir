// RUN: iree-opt -split-input-file -iree-hal-outline-device-switches %s | IreeFileCheck %s

// CHECK-LABEL: @simple_constants
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device
func @simple_constants(%device : !hal.device) -> i32 {
  // CHECK-DAG: %[[C0:.+]] = constant 0
  %c0 = constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = constant 1
  %c1 = constant 1 : i32
  // CHECK-DAG: %[[C2:.+]] = constant 2
  %c2 = constant 2 : i32
  %0 = hal.device.switch(%device : !hal.device) -> i32
    // CHECK-NEXT: %[[IS0:.+]] = hal.device.match.id %[[DEVICE]], pattern = ["vulkan-v1.?-*"] : (!hal.device) -> i1
    // CHECK-NEXT: cond_br %[[IS0]], ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: %[[RES0:.+]] = call @simple_constants_switch_0_0_0_0(%[[C1]]) : (i32) -> i32
    // CHECK-NEXT: br ^bb7(%[[RES0]] : i32)
    #hal.device.match.id<"vulkan-v1.?-*">(%c1a = %c1 : i32) {
      hal.return %c1a : i32
    },
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: %[[IS1L:.+]] = hal.device.match.id %arg0, pattern = ["vmla"] : (!hal.device) -> i1
    // CHECK-NEXT: %[[IS1R:.+]] = hal.device.match.id %arg0, pattern = ["vulkan-*"] : (!hal.device) -> i1
    // CHECK-NEXT: %[[IS1:.+]] = or %[[IS1L]], %[[IS1R]] : i1
    // CHECK-NEXT: cond_br %[[IS1]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT: %[[RES1:.+]] = call @simple_constants_switch_0_0_0_1(%[[C2]]) : (i32) -> i32
    // CHECK-NEXT: br ^bb7(%[[RES1]] : i32)
    #hal.match.any<[#hal.device.match.id<"vmla">, #hal.device.match.id<"vulkan-*">]>(%c2a = %c2 : i32) {
      hal.return %c2a : i32
    },
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT: %[[IS2:.+]] = constant 1 : i1
    // CHECK-NEXT: cond_br %[[IS2]], ^bb5, ^bb6
    // CHECK-NEXT: ^bb5:
    // CHECK-NEXT: %[[RES2:.+]] = call @simple_constants_switch_0_0_0_2(%[[C0]]) : (i32) -> i32
    // CHECK-NEXT: br ^bb7(%[[RES2]] : i32)
    #hal.match.always(%c0a = %c0 : i32) {
      hal.return %c0a : i32
    }
    // CHECK-NEXT: ^bb6:
    // CHECK-NEXT: iree.unreachable
  // CHECK-NEXT: ^bb7(%[[RES:.+]]: i32):
  // CHECK-NEXT: return %[[RES]] : i32
  return %0 : i32
}

// CHECK: func @simple_constants_switch_0_0_0_0(%arg0: i32) -> i32
// CHECK-NEXT: return %arg0 : i32
// CHECK: func @simple_constants_switch_0_0_0_1(%arg0: i32) -> i32
// CHECK-NEXT: return %arg0 : i32
// CHECK: func @simple_constants_switch_0_0_0_2(%arg0: i32) -> i32
// CHECK-NEXT: return %arg0 : i32

// -----

// CHECK-LABEL: @no_results
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device
func @no_results(%device : !hal.device) {
  hal.device.switch(%device : !hal.device)
    // CHECK-NEXT: %[[IS0:.+]] = hal.device.match.id %[[DEVICE]], pattern = ["vulkan-v1.?-*"] : (!hal.device) -> i1
    // CHECK-NEXT: cond_br %[[IS0]], ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: call @no_results_switch_0_0_0_0() : () -> ()
    // CHECK-NEXT: br ^bb7
    #hal.device.match.id<"vulkan-v1.?-*">() {
      hal.return
    },
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: %[[IS1L:.+]] = hal.device.match.id %arg0, pattern = ["vmla"] : (!hal.device) -> i1
    // CHECK-NEXT: %[[IS1R:.+]] = hal.device.match.id %arg0, pattern = ["vulkan-*"] : (!hal.device) -> i1
    // CHECK-NEXT: %[[IS1:.+]] = or %[[IS1L]], %[[IS1R]] : i1
    // CHECK-NEXT: cond_br %[[IS1]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT: call @no_results_switch_0_0_0_1() : () -> ()
    // CHECK-NEXT: br ^bb7
    #hal.match.any<[#hal.device.match.id<"vmla">, #hal.device.match.id<"vulkan-*">]>() {
      hal.return
    },
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT: %[[IS2:.+]] = constant 1 : i1
    // CHECK-NEXT: cond_br %[[IS2]], ^bb5, ^bb6
    // CHECK-NEXT: ^bb5:
    // CHECK-NEXT: call @no_results_switch_0_0_0_2() : () -> ()
    // CHECK-NEXT: br ^bb7
    #hal.match.always() {
      hal.return
    }
    // CHECK-NEXT: ^bb6:
    // CHECK-NEXT: br ^bb7
  // CHECK-NEXT: ^bb7:
  // CHECK-NEXT: return
  return
}

// CHECK: func @no_results_switch_0_0_0_0()
// CHECK: func @no_results_switch_0_0_0_1()
// CHECK: func @no_results_switch_0_0_0_2()
