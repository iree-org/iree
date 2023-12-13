// RUN: iree-opt --split-input-file --iree-spirv-link-executables %s | FileCheck %s

#vulkan_target = #hal.executable.target<"vulkan", "vulkan-spirv-fb">

#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable private @dispatch_0 {
  hal.executable.variant @spirv target(#vulkan_target) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "foo" {
      %c1 = arith.constant 1 : i32
      hal.return %c1 : i32
    }
    hal.executable.export @dispatch_0 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
        spirv.func @dispatch_0() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @dispatch_0
        spirv.ExecutionMode @dispatch_0 "LocalSize", 32, 1, 1
      }
    }
  }
}
hal.executable private @dispatch_1 {
  hal.executable.variant @spirv target(#vulkan_target) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "baz" {
      %c2 = arith.constant 2 : i32
      hal.return %c2 : i32
    }
    hal.executable.export @dispatch_1 ordinal(0) layout(#pipeline_layout) attributes {
      workgroup_size = [64: index, 1: index, 1: index], subgroup_size = 64: index
    } {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
        spirv.func @dispatch_1() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @dispatch_1
        spirv.ExecutionMode @dispatch_1 "LocalSize", 8, 4, 1
      }
    }
  }
}
hal.executable private @dispatch_2 {
  hal.executable.variant @spirv target(#vulkan_target) {
    hal.executable.export @dispatch_2 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      hal.return %c4, %c4, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
        spirv.func @dispatch_2() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @dispatch_2
        spirv.ExecutionMode @dispatch_2 "LocalSize", 16, 16, 1
      }
    }
  }
}
func.func @basic_linking() -> () attributes {
  testing.func.a = @dispatch_0,
  testing.func.b = @dispatch_0::@spirv,
  testing.func.c = @dispatch_0::@spirv::@dispatch_0
} {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer attributes {
    testing.op.a = @dispatch_0,
    testing.op.b = @dispatch_0::@spirv,
    testing.op.c = @dispatch_0::@spirv::@dispatch_0
  }
  %c1 = arith.constant 1 : index
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_0::@spirv::@dispatch_0) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_1::@spirv::@dispatch_1) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_2::@spirv::@dispatch_2) workgroups([%c1, %c1, %c1])
  return
}
util.initializer {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
  %c1 = arith.constant 1 : index
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_0::@spirv::@dispatch_0) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_1::@spirv::@dispatch_1) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_2::@spirv::@dispatch_2) workgroups([%c1, %c1, %c1])
  util.initializer.return
}

// All executables (including their interfaces and entry points) should be
// linked together into a single executable.

//  CHECK-NOT: hal.executable private @dispatch_0
//  CHECK-NOT: hal.executable private @dispatch_1
//  CHECK-NOT: hal.executable private @dispatch_2

//      CHECK: hal.executable private @link_executables_linked_spirv {
// CHECK-NEXT:   hal.executable.variant public @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
// CHECK-NEXT:     hal.executable.constant.block(%arg0: !hal.device) -> i32 as "foo"
// CHECK-NEXT:       = arith.constant 1
//      CHECK:     hal.executable.export public @dispatch_0 ordinal(0)
//      CHECK:       hal.return %c1, %c1, %c1
//      CHECK:     hal.executable.constant.block(%arg0: !hal.device) -> i32 as "baz"
// CHECK-NEXT:       = arith.constant 2
//      CHECK:     hal.executable.export public @dispatch_1 ordinal(1)
// CHECK-SAME:       {subgroup_size = 64 : index, workgroup_size = [64 : index, 1 : index, 1 : index]}
//      CHECK:       hal.return %c1, %c1, %c1
//      CHECK:     hal.executable.export public @dispatch_2 ordinal(2)
//      CHECK:       hal.return %c4, %c4, %c1
//      CHECK:     builtin.module {
//      CHECK:       spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []>
// CHECK-NEXT:         spirv.func @dispatch_0()
//      CHECK:         spirv.EntryPoint "GLCompute" @dispatch_0
//      CHECK:         spirv.ExecutionMode @dispatch_0 "LocalSize", 32, 1, 1
//      CHECK:       spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []>
// CHECK-NEXT:         spirv.func @dispatch_1()
//      CHECK:         spirv.EntryPoint "GLCompute" @dispatch_1
//      CHECK:         spirv.ExecutionMode @dispatch_1 "LocalSize", 8, 4, 1
//      CHECK:       spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []>
// CHECK-NEXT:         spirv.func @dispatch_2()
//      CHECK:         spirv.EntryPoint "GLCompute" @dispatch_2
//      CHECK:         spirv.ExecutionMode @dispatch_2 "LocalSize", 16, 16, 1
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
//
// CHECK:       func.func @basic_linking()
// CHECK:           testing.func.a = @link_executables_linked_spirv
// CHECK-SAME:      testing.func.b = @link_executables_linked_spirv::@vulkan_spirv_fb
// CHECK-SAME:      testing.func.c = @link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_0
// CHECK:           testing.op.a = @link_executables_linked_spirv
// CHECK-SAME:      testing.op.b = @link_executables_linked_spirv::@vulkan_spirv_fb
// CHECK-SAME:      testing.op.c = @link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_0
// CHECK:         hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_0) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_1) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_2) workgroups([%c1, %c1, %c1])
//
// CHECK:       util.initializer
// CHECK:         hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_0) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_1) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_2) workgroups([%c1, %c1, %c1])

// -----

#vulkan_target_0 = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader], []>,
    api=Vulkan, Unknown:DiscreteGPU, #spirv.resource_limits<>>}>
#vulkan_target_1 = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, CooperativeMatrixKHR], [SPV_KHR_cooperative_matrix]>,
    api=Vulkan, Unknown:DiscreteGPU, #spirv.resource_limits<>>}>

#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable private @dispatch_0 {
  hal.executable.variant @spirv target(#vulkan_target_0) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "foo" {
      %c1 = arith.constant 1 : i32
      hal.return %c1 : i32
    }
    hal.executable.export @dispatch_0 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
        spirv.func @dispatch_0() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @dispatch_0
        spirv.ExecutionMode @dispatch_0 "LocalSize", 32, 1, 1
      }
    }
  }
}
hal.executable private @dispatch_1 {
  hal.executable.variant @spirv target(#vulkan_target_1) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "baz" {
      %c2 = arith.constant 2 : i32
      hal.return %c2 : i32
    }
    hal.executable.export @dispatch_1 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
        spirv.func @dispatch_1() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @dispatch_1
        spirv.ExecutionMode @dispatch_1 "LocalSize", 4, 8, 1
      }
    }
  }
}
hal.executable private @dispatch_2 {
  hal.executable.variant @spirv target(#vulkan_target_0) {
    hal.executable.export @dispatch_2 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
        spirv.func @dispatch_2() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @dispatch_2
        spirv.ExecutionMode @dispatch_2 "LocalSize", 8, 8, 2
      }
    }
  }
}
hal.executable private @dispatch_3 {
  hal.executable.variant @spirv target(#vulkan_target_1) {
    hal.executable.export @dispatch_3 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
        spirv.func @dispatch_3() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @dispatch_3
        spirv.ExecutionMode @dispatch_3 "LocalSize", 16, 8, 2
      }
    }
  }
}
func.func @two_target_environments_1() -> () {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
  %c1 = arith.constant 1 : index
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_0::@spirv::@dispatch_0) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_2::@spirv::@dispatch_2) workgroups([%c1, %c1, %c1])
  return
}
func.func @two_target_environments_2() -> () {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
  %c1 = arith.constant 1 : index
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_1::@spirv::@dispatch_1) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_3::@spirv::@dispatch_3) workgroups([%c1, %c1, %c1])
  return
}

// We have two different target environments, so we need to form two different
// hal.executable.variant ops inside the hal.executable op.
// Note that we might be further merge these two hal.executable.variant ops here
// given that the deduced target requirements across all dispatches are actually
// the same. But that can happen in another place.

//      CHECK: #[[TARGET0:.+]] = #hal.executable.target<"vulkan", "vulkan-spirv-fb",
// CHECK-SAME:   #spirv.target_env<#spirv.vce<v1.6, [Shader], []>
//      CHECK: #[[TARGET1:.+]] = #hal.executable.target<"vulkan", "vulkan-spirv-fb",
// CHECK-SAME:   #spirv.target_env<#spirv.vce<v1.6, [Shader, CooperativeMatrixKHR], [SPV_KHR_cooperative_matrix]>

//      CHECK: hal.executable private @link_executables_linked_spirv {
//      CHECK:   hal.executable.variant public @vulkan_spirv_fb_0 target(#[[TARGET0]]) {
//      CHECK:     hal.executable.constant.block(%arg0: !hal.device) -> i32 as "foo"
// CHECK-NEXT:       = arith.constant 1 : i32
//      CHECK:     hal.executable.export public @dispatch_0 ordinal(0)
//      CHECK:     hal.executable.export public @dispatch_2 ordinal(1)
//      CHECK:     builtin.module {
//      CHECK:       spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
//      CHECK:         spirv.func @dispatch_0()
//      CHECK:         spirv.EntryPoint "GLCompute" @dispatch_0
//      CHECK:         spirv.ExecutionMode @dispatch_0 "LocalSize", 32, 1, 1
//      CHECK:       }
//      CHECK:       spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
//      CHECK:         spirv.func @dispatch_2()
//      CHECK:         spirv.EntryPoint "GLCompute" @dispatch_2
//      CHECK:         spirv.ExecutionMode @dispatch_2 "LocalSize", 8, 8, 2
//      CHECK:       }
//      CHECK:     }
//      CHECK:   }
//      CHECK:   hal.executable.variant public @vulkan_spirv_fb_1 target(#[[TARGET1]]) {
//      CHECK:     hal.executable.constant.block(%arg0: !hal.device) -> i32 as "baz"
// CHECK-NEXT:       = arith.constant 2 : i32
//      CHECK:     hal.executable.export public @dispatch_1 ordinal(0)
//      CHECK:     hal.executable.export public @dispatch_3 ordinal(1)
//      CHECK:     builtin.module {
//      CHECK:       spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
//      CHECK:         spirv.func @dispatch_1()
//      CHECK:         spirv.EntryPoint "GLCompute" @dispatch_1
//      CHECK:         spirv.ExecutionMode @dispatch_1 "LocalSize", 4, 8, 1
//      CHECK:       }
//      CHECK:       spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
//      CHECK:         spirv.func @dispatch_3()
//      CHECK:         spirv.EntryPoint "GLCompute" @dispatch_3
//      CHECK:         spirv.ExecutionMode @dispatch_3 "LocalSize", 16, 8, 2
//      CHECK:       }
//      CHECK:     }
//      CHECK:   }
//      CHECK: }

// Check usages are updated properly. Note that for one executable, we only
// ever load one variants out of it; we cannot invoke entry points from
// different variants of the same executable in the same command buffer.
// So here we have two separate functions with two separate command buffers
// for testing purposes.

//      CHECK: func.func @two_target_environments_1()
//      CHECK:   hal.command_buffer.dispatch.symbol{{.+}} target(@link_executables_linked_spirv::@vulkan_spirv_fb_0::@dispatch_0)
//      CHECK:   hal.command_buffer.dispatch.symbol{{.+}} target(@link_executables_linked_spirv::@vulkan_spirv_fb_0::@dispatch_2)

//      CHECK: func.func @two_target_environments_2()
//      CHECK:   hal.command_buffer.dispatch.symbol{{.+}} target(@link_executables_linked_spirv::@vulkan_spirv_fb_1::@dispatch_1)
//      CHECK:   hal.command_buffer.dispatch.symbol{{.+}} target(@link_executables_linked_spirv::@vulkan_spirv_fb_1::@dispatch_3)
