// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-hal-inline-device-switches --canonicalize %s | FileCheck %s

// CHECK-LABEL: @simple_constants
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device
// CHECK-SAME: %[[ARG:.+]]: i32
func.func @simple_constants(%device : !hal.device, %arg : i32) -> i32 {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0
  %c0 = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1
  %c1 = arith.constant 1 : i32
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2
  %c2 = arith.constant 2 : i32
  // CHECK-DAG: %[[C3:.+]] = arith.constant 3
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4
  %0 = hal.device.switch<%device : !hal.device> -> i32
    // CHECK-NEXT: %{{.+}}, %[[IS0:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "vulkan-v1.?-*") : i1, i1 = false
    // CHECK-NEXT: cf.cond_br %[[IS0]], ^bb3(%[[C1]] : i32), ^bb1
    #hal.device.match.id<"vulkan-v1.?-*"> {
      hal.return %c1 : i32
    },
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:  %{{.+}}, %[[IS1L:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "vmvx") : i1, i1 = false
    // CHECK-NEXT:  %{{.+}}, %[[IS1R:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "vulkan-*") : i1, i1 = false
    // CHECK-NEXT:  %[[IS1:.+]] = arith.ori %[[IS1L]], %[[IS1R]] : i1
    // CHECK-NEXT:  cf.cond_br %[[IS1]], ^bb2, ^bb3(%[[C0]] : i32)
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:  %[[EQZ:.+]] = arith.cmpi eq, %[[ARG]], %[[C2]] : i32
    // CHECK-NEXT:  cf.cond_br %[[EQZ]], ^bb3(%[[C3]] : i32), ^bb3(%[[C4]] : i32)
    #hal.match.any<[#hal.device.match.id<"vmvx">, #hal.device.match.id<"vulkan-*">]> {
      %eqz = arith.cmpi eq, %arg, %c2 : i32
      cf.cond_br %eqz, ^bb_true, ^bb_false
    ^bb_true:
      %c3 = arith.constant 3 : i32
      hal.return %c3 : i32
    ^bb_false:
      %c4 = arith.constant 4 : i32
      hal.return %c4 : i32
    },
    #hal.match.always {
      hal.return %c0 : i32
    }
  // CHECK-NEXT: ^bb3(%[[RES:.+]]: i32):
  // CHECK-NEXT: return %[[RES]] : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: @no_results
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device
func.func @no_results(%device : !hal.device) {
  hal.device.switch<%device : !hal.device>
    // CHECK-NEXT:  %{{.+}}, %[[IS0:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "vulkan-v1.?-*") : i1, i1 = false
    // CHECK-NEXT:  cf.cond_br %[[IS0]], ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:  "some.op_a"()
    // CHECK-NEXT:  cf.br ^bb5
    #hal.device.match.id<"vulkan-v1.?-*"> {
      "some.op_a"() : () -> ()
      hal.return
    },
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:  %{{.+}}, %[[IS1L:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "vmvx") : i1, i1 = false
    // CHECK-NEXT:  %{{.+}}, %[[IS1R:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "vulkan-*") : i1, i1 = false
    // CHECK-NEXT:  %[[IS1:.+]] = arith.ori %[[IS1L]], %[[IS1R]] : i1
    // CHECK-NEXT:  cf.cond_br %[[IS1]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:  "some.op_b"()
    // CHECK-NEXT:  cf.br ^bb5
    #hal.match.any<[#hal.device.match.id<"vmvx">, #hal.device.match.id<"vulkan-*">]> {
      "some.op_b"() : () -> ()
      hal.return
    },
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT:  "some.op_c"()
    // CHECK-NEXT:  cf.br ^bb5
    #hal.match.always {
      "some.op_c"() : () -> ()
      hal.return
    }
  // CHECK-NEXT: ^bb5:
  // CHECK-NEXT:  return
  return
}
