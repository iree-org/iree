// RUN: iree-opt --split-input-file --iree-hal-initialize-devices --cse %s | FileCheck %s

// Tests that #hal.device.ordinal<*> gets the device with the given ordinal.

// CHECK: util.global private @device_123 : !hal.device
util.global private @device_123 = #hal.device.ordinal<123> : !hal.device

// CHECK-NEXT: util.initializer
//  CHECK-DAG: %[[DEVICE:.+]] = hal.devices.get %c123
//  CHECK-DAG: %[[NULL_DEVICE:.+]] = util.null : !hal.device
//  CHECK-DAG: %[[IS_NULL:.+]] = util.cmp.eq %[[DEVICE]], %[[NULL_DEVICE]]
// CHECK-NEXT: scf.if %[[IS_NULL]] {
//      CHECK:   util.status.check_ok %c5_i32, "HAL device `device_123` not found or unavailable: #hal.device.ordinal<123>"
//      CHECK: util.global.store %[[DEVICE]], @device_123

// -----

// Tests that #hal.device.fallback<*> references the specified device global.

util.global private @device_base : !hal.device

// CHECK: util.global private @device_fallback : !hal.device
util.global private @device_fallback = #hal.device.fallback<@device_base> : !hal.device

// CHECK-NEXT: util.initializer
//  CHECK-DAG: %[[DEVICE:.+]] = util.global.load @device_base : !hal.device
//  CHECK-DAG: %[[IS_NULL:.+]] = util.cmp.eq %[[DEVICE]], %{{.+}}
// CHECK-NEXT: scf.if %[[IS_NULL]] {
//      CHECK:   util.status.check_ok %c5_i32, "HAL device `device_fallback` not found or unavailable: #hal.device.fallback<@device_base>"
//      CHECK: util.global.store %[[DEVICE]], @device_fallback

// -----

// Tests that #hal.device.target<*> enumerates all devices.

// CHECK: util.global private @device_a : !hal.device
util.global private @device_a = #hal.device.target<"a", [
  #hal.executable.target<"backend0", "format0">,
  #hal.executable.target<"backend1", "format1">
]> : !hal.device

// CHECK-NEXT: util.initializer
//  CHECK-DAG: %[[NULL_DEVICE:.+]] = util.null : !hal.device
//  CHECK-DAG: %[[DEVICE_COUNT:.+]] = hal.devices.count
//      CHECK: %[[WHILE:.+]]:2 = scf.while (%arg0 = %c0, %arg1 = %[[NULL_DEVICE]])
//  CHECK-DAG:   %[[IS_DEVICE_NULL:.+]] = util.cmp.eq %arg1, %[[NULL_DEVICE]]
//  CHECK-DAG:   %[[IS_END:.+]] = arith.cmpi slt, %arg0, %[[DEVICE_COUNT]]
//  CHECK-DAG:   %[[CONTINUE:.+]] = arith.andi %[[IS_DEVICE_NULL]], %[[IS_END]]
// CHECK-NEXT:   scf.condition(%[[CONTINUE]]) %arg0, %arg1
// CHECK-NEXT: } do {
// CHECK-NEXT:  ^bb0(%arg0: index, %arg1: !hal.device)
//  CHECK-DAG:   %[[DEVICE_N:.+]] = hal.devices.get %arg0 : !hal.device

// NOTE: this is the fallback path for device matching unregistered targets.
// Real targets can have much more complex logic if they so choose.
//  CHECK-DAG:   %{{.+}}, %[[ID_MATCH:.+]] = hal.device.query<%[[DEVICE_N]] : !hal.device> key("hal.device.id" :: "a")
// CHECK-NEXT:   %[[ANY_FORMAT_MATCH:.+]] = scf.if %[[ID_MATCH]] -> (i1) {
//  CHECK-DAG:     %{{.+}}, %[[FORMAT0_MATCH:.+]] = hal.device.query<%[[DEVICE_N]] : !hal.device> key("hal.executable.format" :: "format0")
//  CHECK-DAG:     %{{.+}}, %[[FORMAT1_MATCH:.+]] = hal.device.query<%[[DEVICE_N]] : !hal.device> key("hal.executable.format" :: "format1")
//  CHECK-DAG:     %[[FORMAT_MATCH_OR:.+]] = arith.ori %[[FORMAT0_MATCH]], %[[FORMAT1_MATCH]]
//  CHECK-DAG:     scf.yield %[[FORMAT_MATCH_OR]]
// CHECK-NEXT:   } else {
//  CHECK-DAG:     scf.yield %false

//  CHECK-DAG:   %[[YIELD_DEVICE:.+]] = arith.select %[[ANY_FORMAT_MATCH]], %[[DEVICE_N]], %[[NULL_DEVICE]]
//  CHECK-DAG:   %[[NEXT_I:.+]] = arith.addi %arg0, %c1
// CHECK-NEXT:   scf.yield %[[NEXT_I]], %[[YIELD_DEVICE]]
//  CHECK-DAG: %[[IS_NULL:.+]] = util.cmp.eq %[[WHILE]]#1, %[[NULL_DEVICE]]
// CHECK-NEXT: scf.if %[[IS_NULL]] {
//      CHECK:   util.status.check_ok %c5_i32, "HAL device `device_a` not found or unavailable: #hal.device.target<{{.+}}>"
//      CHECK: util.global.store %[[WHILE]]#1, @device_a

// -----

// Tests that #hal.device.select<*> expands to a chain of ifs.

util.global private @fallback : !hal.device

// CHECK: util.global private @selected : !hal.device
util.global private @selected = #hal.device.select<[
  #hal.device.ordinal<2> : !hal.device,
  #hal.device.ordinal<1> : !hal.device,
  #hal.device.fallback<@fallback> : !hal.device
]> : !hal.device

// CHECK-NEXT: util.initializer
//  CHECK-DAG: %[[NULL_DEVICE:.+]] = util.null : !hal.device
//  CHECK-DAG: %[[DEVICE_2:.+]] = hal.devices.get %c2
//  CHECK-DAG: %[[NOT_DEVICE_2:.+]] = util.cmp.eq %[[DEVICE_2]], %[[NULL_DEVICE]]
// CHECK-NEXT: %[[IF_0:.+]] = scf.if %[[NOT_DEVICE_2]]
//  CHECK-DAG:   %[[DEVICE_1:.+]] = hal.devices.get %c1
//  CHECK-DAG:   %[[NOT_DEVICE_1:.+]] = util.cmp.eq %[[DEVICE_1]], %[[NULL_DEVICE]]
// CHECK-NEXT:   %[[IF_1:.+]] = scf.if %[[NOT_DEVICE_1]]
//  CHECK-DAG:     %[[DEVICE_FALLBACK:.+]] = util.global.load @fallback
// CHECK-NEXT:     scf.yield %[[DEVICE_FALLBACK]]
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %[[DEVICE_1]]
// CHECK-NEXT:   }
// CHECK-NEXT:   scf.yield %[[IF_1]]
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[DEVICE_2]]
// CHECK-NEXT: }
//  CHECK-DAG: %[[IS_NULL:.+]] = util.cmp.eq %[[IF_0]], %[[NULL_DEVICE]]
// CHECK-NEXT: scf.if %[[IS_NULL]] {
//      CHECK:   util.status.check_ok %c5_i32, "HAL device `selected` not found or unavailable: #hal.device.select<{{.+}}>"
//      CHECK: util.global.store %[[IF_0]], @selected
