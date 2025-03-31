// RUN: iree-opt --split-input-file --iree-hal-materialize-resource-caches %s | FileCheck %s

#pipeline_layout_0 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#pipeline_layout_1 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK: hal.executable private @exe
hal.executable private @exe {
  // CHECK: hal.executable.variant public @vmvx
  hal.executable.variant @vmvx target(<"vmvx", "vmvx-bytecode-fb">) {
    // CHECK-NOT: hal.executable.condition
    hal.executable.condition(%device: !hal.device) -> i1 {
      %ok, %selected = hal.device.query<%device : !hal.device> key("some" :: "feature") : i1, i1
      hal.return %selected : i1
    }
    hal.executable.export @entry0 ordinal(0) layout(#pipeline_layout_0)
    hal.executable.export @entry0_alias ordinal(0) layout(#pipeline_layout_0)
    hal.executable.export @entry1 ordinal(1) layout(#pipeline_layout_1)
    // CHECK-NOT: hal.executable.constant.block
    hal.executable.constant.block() -> (i32, i32) as ("foo", "bar") {
      %c123 = arith.constant 123 : i32
      %c456 = arith.constant 456 : i32
      hal.return %c123, %c456 : i32, i32
    }
    // CHECK-NOT: hal.executable.constant.block
    hal.executable.constant.block(%device: !hal.device) -> i32 as "baz" {
      %ok, %query = hal.device.query<%device : !hal.device> key("sys" :: "baz") : i1, i32
      cf.cond_br %ok, ^bb_ok, ^bb_fail
    ^bb_ok:
      hal.return %query : i32
    ^bb_fail:
      %dummy = arith.constant 0 : i32
      hal.return %dummy : i32
    }
  }
}

// CHECK: util.global private @device = #hal.device.ordinal<0>
util.global private @device = #hal.device.ordinal<0> : !hal.device

// Cached resources for the device.
// CHECK: util.global private @__device_executable_0_exe : !hal.executable

// Device initializer for all resources used with the device:
// CHECK: util.initializer
// CHECK:   %[[DEVICE:.+]] = util.global.load @device

// Switch on the supported formats:
// CHECK:   %{{.+}}, %[[FORMAT_VMVX:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.executable.format" :: "vmvx-bytecode-fb")
// CHECK:   %[[VMVX_CONDITION:.+]] = scf.execute_region -> i1 {
// CHECK:     %{{.+}}, %[[FEATURE:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("some" :: "feature")
// CHECK:     scf.yield %[[FEATURE]]
// CHECK:   }
// CHECK:   %[[VMVX_VARIANT_SELECTED:.+]] = arith.andi %[[FORMAT_VMVX]], %[[VMVX_CONDITION]]
// CHECK-DAG: %[[VARIANT_VMVX:.+]] = arith.constant 0
// CHECK-DAG: %[[VARIANT_DEFAULT:.+]] = arith.constant -1
// CHECK:   %[[VARIANT_INDEX:.+]] = arith.select %[[VMVX_VARIANT_SELECTED]], %[[VARIANT_VMVX]], %[[VARIANT_DEFAULT]]
// CHECK:   %[[RET:.+]] = scf.index_switch %[[VARIANT_INDEX]] -> !hal.executable
// CHECK:   case 0 {

// Constant block initializers:
// CHECK:     %[[CONST_01:.+]]:2 = util.call @__device_executable_0_exe_constant_block_0()
// CHECK:     %[[CONST_2:.+]] = util.call @__device_executable_0_exe_constant_block_1(%[[DEVICE]])

// Executable creation:
// CHECK:     %[[EXE:.+]] = hal.executable.create
// CHECK-SAME:  device(%[[DEVICE]] : !hal.device)
// CHECK-SAME:  target(@exe::@vmvx)
// CHECK-SAME:  constants([%[[CONST_01]]#0, %[[CONST_01]]#1, %[[CONST_2]]])
// CHECK-SAME:  : !hal.executable

// CHECK:     scf.yield %[[EXE]] : !hal.executable
// CHECK:   }
// CHECK:   default {
// CHECK:     %[[C14:.+]] = arith.constant 14 : i32
// CHECK:     util.status.check_ok %[[C14]], "HAL device `device` does not support any variant of executable `exe`; available formats: [vmvx-bytecode-fb]"
// CHECK:     %[[NULL:.+]] = util.null : !hal.executable
// CHECK:     scf.yield %[[NULL]] : !hal.executable
// CHECK:   }
// CHECK:   util.global.store %[[RET]], @__device_executable_0_exe : !hal.executable

// Constant block functions (here we ensure all blocks are cloned):
// CHECK: util.func private @__device_executable_0_exe_constant_block_0() -> (i32, i32)
// CHECK-DAG: %[[C0:.+]] = arith.constant 123
// CHECK-DAG: %[[C1:.+]] = arith.constant 456
// CHECK: util.return %[[C0]], %[[C1]]
// CHECK: util.func private @__device_executable_0_exe_constant_block_1(%[[BLOCK_DEVICE:.+]]: !hal.device) -> i32
// CHECK:   %[[OK:.+]], %[[VALUE:.+]] = hal.device.query<%[[BLOCK_DEVICE]] : !hal.device> key("sys" :: "baz")
// CHECK:   cf.cond_br %[[OK]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   util.return %[[VALUE]]
// CHECK: ^bb2:
// CHECK:   %[[DUMMY:.+]] = arith.constant 0
// CHECK:   util.return %[[DUMMY]]

// CHECK-LABEL: @exeLookup
util.func public @exeLookup() -> !hal.executable {
  %device = util.global.load @device : !hal.device
  // CHECK: %[[EXE:.+]] = util.global.load @__device_executable_0_exe : !hal.executable
  %0 = hal.executable.lookup device(%device : !hal.device)
                         executable(@exe) : !hal.executable
  // CHECK-NEXT: util.return %[[EXE]]
  util.return %0 : !hal.executable
}

// -----

// Tests that fallback resources are reused instead of being created again
// when a device selects a fallback.

// CHECK: hal.executable private @exe
hal.executable private @exe {
  // CHECK: hal.executable.variant public @vmvx
  hal.executable.variant @vmvx target(<"vmvx", "vmvx-bytecode-fb">) {
    // CHECK-NOT: hal.executable.condition
    hal.executable.condition(%device: !hal.device) -> i1 {
      %ok, %selected = hal.device.query<%device : !hal.device> key("some" :: "feature") : i1, i1
      hal.return %selected : i1
    }
    hal.executable.export @entry0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>,
      #hal.pipeline.binding<storage_buffer>
    ]>)
    // CHECK-NOT: hal.executable.constant.block
    hal.executable.constant.block() -> (i32, i32) as ("foo", "bar") {
      %c123 = arith.constant 123 : i32
      %c456 = arith.constant 456 : i32
      hal.return %c123, %c456 : i32, i32
    }
  }
}

// CHECK: util.global private @primary_device
util.global private @primary_device = #hal.device.ordinal<0> : !hal.device
// CHECK-NEXT: util.global private @__primary_device_executable_0_exe
// CHECK-NEXT: util.initializer
//      CHECK:   util.global.load @primary_device
//      CHECK:   hal.executable.create
//      CHECK:   util.global.store {{.+}}, @__primary_device_executable_0_exe
//      CHECK: util.func private @__primary_device_executable_0_exe_constant_block_0

// CHECK: util.global private @optional_device
util.global private @optional_device = #hal.device.select<[
  #hal.device.ordinal<1> : !hal.device,
  #hal.device.fallback<@primary_device> : !hal.device
]> : !hal.device
// CHECK-NEXT: util.global private @__optional_device_executable_0_exe
// CHECK-NEXT: util.initializer
//  CHECK-DAG:   %[[OPTIONAL_DEVICE:.+]] = util.global.load @optional_device
//  CHECK-DAG:   %[[PRIMARY_DEVICE:.+]] = util.global.load @primary_device
//  CHECK-DAG:   %[[DEVICE_EQ:.+]] = util.cmp.eq %[[OPTIONAL_DEVICE]], %[[PRIMARY_DEVICE]]
//  CHECK-DAG:   %[[INDEX:.+]] = arith.select %[[DEVICE_EQ]]
//  CHECK-DAG:   scf.index_switch %[[INDEX]]
//      CHECK:   case 0
//      CHECK:     %[[PRIMARY_EXE:.+]] = util.global.load @__primary_device_executable_0_exe
//      CHECK:     util.global.store %[[PRIMARY_EXE]], @__optional_device_executable_0_exe
//      CHECK:   default
//      CHECK:     hal.executable.create
//      CHECK:     util.global.store {{.+}}, @__optional_device_executable_0_exe
//      CHECK: util.func private @__optional_device_executable_0_exe_constant_block_0

// CHECK-LABEL: @fallbackLookup
util.func public @fallbackLookup() -> (!hal.executable, !hal.executable) {
  %primary_device = util.global.load @primary_device : !hal.device
  // CHECK: %[[PRIMARY_EXE_LOOKUP:.+]] = util.global.load @__primary_device_executable_0_exe
  %0 = hal.executable.lookup device(%primary_device : !hal.device)
                         executable(@exe) : !hal.executable
  %optional_device = util.global.load @optional_device : !hal.device
  // CHECK: %[[OPTIONAL_EXE_LOOKUP:.+]] = util.global.load @__optional_device_executable_0_exe
  %1 = hal.executable.lookup device(%optional_device : !hal.device)
                         executable(@exe) : !hal.executable
  util.return %0, %1 : !hal.executable, !hal.executable
}

// -----

// Tests that resources only used by optional devices force the resources to
// be created on fallbacks. This isn't optimal as we should really only be
// creating them if the fallback is selected but that's more complex than it's
// worth today given the limited usage of fallbacks.

hal.executable private @exe {
  hal.executable.variant @vmvx target(<"vmvx", "vmvx-bytecode-fb">) {
    hal.executable.export @entry0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>
    ]>)
  }
}

// CHECK-LABEL: util.global private @primary_device
util.global private @primary_device = #hal.device.ordinal<0> : !hal.device
// CHECK-NEXT: util.global private @__primary_device_executable_0_exe
// CHECK-NEXT: util.initializer
//      CHECK:   util.global.load @primary_device
//      CHECK:   hal.executable.create
//      CHECK:   util.global.store {{.+}}, @__primary_device_executable_0_exe

// CHECK-LABEL: util.global private @optional_device_0
util.global private @optional_device_0 = #hal.device.select<[
  #hal.device.ordinal<1> : !hal.device,
  #hal.device.fallback<@primary_device> : !hal.device
]> : !hal.device
// CHECK-NEXT: util.global private @__optional_device_0_executable_0_exe
// CHECK-NEXT: util.initializer
//  CHECK-DAG:   %[[OPTIONAL_DEVICE_0:.+]] = util.global.load @optional_device_0
//  CHECK-DAG:   %[[PRIMARY_DEVICE:.+]] = util.global.load @primary_device
//  CHECK-DAG:   %[[DEVICE_EQ:.+]] = util.cmp.eq %[[OPTIONAL_DEVICE_0]], %[[PRIMARY_DEVICE]]
//  CHECK-DAG:   %[[INDEX:.+]] = arith.select %[[DEVICE_EQ]]
//  CHECK-DAG:   scf.index_switch %[[INDEX]]
//      CHECK:     util.global.load @__primary_device_executable_0_exe
//      CHECK:     util.global.store {{.+}}, @__optional_device_0_executable_0_exe

// CHECK-LABEL: util.global private @optional_device_1
util.global private @optional_device_1 = #hal.device.select<[
  #hal.device.ordinal<2> : !hal.device,
  #hal.device.fallback<@optional_device_0> : !hal.device
]> : !hal.device
// CHECK-NEXT: util.global private @__optional_device_1_executable_0_exe
// CHECK-NEXT: util.initializer
//  CHECK-DAG:   %[[OPTIONAL_DEVICE_1:.+]] = util.global.load @optional_device_1
//  CHECK-DAG:   %[[OPTIONAL_DEVICE_0:.+]] = util.global.load @optional_device_0
//  CHECK-DAG:   %[[DEVICE_EQ:.+]] = util.cmp.eq %[[OPTIONAL_DEVICE_1]], %[[OPTIONAL_DEVICE_0]]
//  CHECK-DAG:   %[[INDEX:.+]] = arith.select %[[DEVICE_EQ]]
//  CHECK-DAG:   scf.index_switch %[[INDEX]]
//      CHECK:     util.global.load @__optional_device_0_executable_0_exe
//      CHECK:     util.global.store {{.+}}, @__optional_device_1_executable_0_exe

// CHECK-LABEL: @fallbackOnlyLookup
util.func public @fallbackOnlyLookup() -> !hal.executable {
  %optional_device_1 = util.global.load @optional_device_1 : !hal.device
  // CHECK: util.global.load @__optional_device_1_executable_0_exe
  %0 = hal.executable.lookup device(%optional_device_1 : !hal.device)
                         executable(@exe) : !hal.executable
  util.return %0 : !hal.executable
}

// -----

// Tests that materialization no-ops when resource caches have already been
// materialized. Today this is rather simplistic and just bails if the names
// match with the expectation being that users are mostly just running through
// with --compile-to=hal and not trying to mutate intermediate HAL state. We
// could rework the pass to support only materializing what's required based on
// what resources are looked up.

#pipeline_layout_0 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

util.global private @device : !hal.device

util.global private @_executable_exe : !hal.executable
util.initializer {
  %c0 = arith.constant 0 : index
  %affinity = arith.constant -1 : i64
  %device = hal.devices.get %c0 : !hal.device
  %format_ok, %format_supported = hal.device.query<%device : !hal.device> key("hal.executable.format" :: "some-format") : i1, i1
  %c-1 = arith.constant -1 : index
  %variant = arith.select %format_supported, %c0, %c-1 : index
  %selected = scf.index_switch %variant -> !hal.executable
  case 0 {
    %exe = hal.executable.create device(%device : !hal.device) affinity(%affinity) target(@exe0::@vmvx) : !hal.executable
    scf.yield %exe : !hal.executable
  }
  default {
    %null = util.null : !hal.executable
    scf.yield %null : !hal.executable
  }
  util.global.store %selected, @_executable_exe : !hal.executable
  util.return
}

hal.executable @exe {
  hal.executable.variant @vmvx target(<"vmvx", "vmvx-bytecode-fb">) {
    hal.executable.export @entry ordinal(0) layout(#pipeline_layout_0) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
  }
}

// CHECK-LABEL: @exeLookup
util.func public @exeLookup(%device : !hal.device) -> !hal.executable {
  // CHECK: %[[EXE:.+]] = util.global.load @_executable_exe : !hal.executable
  %0 = util.global.load @_executable_exe : !hal.executable
  // CHECK-NEXT: util.return %[[EXE]]
  util.return %0 : !hal.executable
}
