// RUN: iree-opt --split-input-file --iree-spirv-link-executables %s | FileCheck %s

// A test case for
// * one variant per executable,
// * same target for all variants across all executables.
//
// For such case we can link all executables into one, with just one variant.

#vulkan_target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv"]}>

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
  %dispatch_0_exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_0) : !hal.executable
  %dispatch_1_exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_1) : !hal.executable
  %dispatch_2_exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_2) : !hal.executable
  %dispatch_0_ordinal = hal.executable.export.ordinal target(@dispatch_0::@spirv::@dispatch_0) : index
  %dispatch_1_ordinal = hal.executable.export.ordinal target(@dispatch_1::@spirv::@dispatch_1) : index
  %dispatch_2_ordinal = hal.executable.export.ordinal target(@dispatch_2::@spirv::@dispatch_2) : index
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%dispatch_0_exe : !hal.executable)[%dispatch_0_ordinal] workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%dispatch_1_exe : !hal.executable)[%dispatch_1_ordinal] workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%dispatch_2_exe : !hal.executable)[%dispatch_2_ordinal] workgroups([%c1, %c1, %c1])
  return
}
util.initializer {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
  %c1 = arith.constant 1 : index
  %dispatch_0_exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_0) : !hal.executable
  %dispatch_1_exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_1) : !hal.executable
  %dispatch_2_exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_2) : !hal.executable
  %dispatch_0_ordinal = hal.executable.export.ordinal target(@dispatch_0::@spirv::@dispatch_0) : index
  %dispatch_1_ordinal = hal.executable.export.ordinal target(@dispatch_1::@spirv::@dispatch_1) : index
  %dispatch_2_ordinal = hal.executable.export.ordinal target(@dispatch_2::@spirv::@dispatch_2) : index
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%dispatch_0_exe : !hal.executable)[%dispatch_0_ordinal] workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%dispatch_1_exe : !hal.executable)[%dispatch_1_ordinal] workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%dispatch_2_exe : !hal.executable)[%dispatch_2_ordinal] workgroups([%c1, %c1, %c1])
  util.return
}

//  CHECK-NOT: hal.executable private @dispatch_0
//  CHECK-NOT: hal.executable private @dispatch_1
//  CHECK-NOT: hal.executable private @dispatch_2

//      CHECK: hal.executable private @link_executables_linked_spirv {
// CHECK-NEXT:   hal.executable.variant public @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
//      CHECK:     hal.executable.constant.block(%arg0: !hal.device) -> i32 as "foo"
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
// CHECK-DAG:     %[[DISPATCH_0_EXE:.+]] = hal.executable.lookup device(%{{.+}}) executable(@link_executables_linked_spirv) : !hal.executable
// CHECK-DAG:     %[[DISPATCH_1_EXE:.+]] = hal.executable.lookup device(%{{.+}}) executable(@link_executables_linked_spirv) : !hal.executable
// CHECK-DAG:     %[[DISPATCH_2_EXE:.+]] = hal.executable.lookup device(%{{.+}}) executable(@link_executables_linked_spirv) : !hal.executable
// CHECK-DAG:     %[[DISPATCH_0_ORDINAL:.+]] = hal.executable.export.ordinal target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_0)
// CHECK-DAG:     %[[DISPATCH_1_ORDINAL:.+]] = hal.executable.export.ordinal target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_1)
// CHECK-DAG:     %[[DISPATCH_2_ORDINAL:.+]] = hal.executable.export.ordinal target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_2)
// CHECK:         hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%[[DISPATCH_0_EXE]] : !hal.executable)[%[[DISPATCH_0_ORDINAL]]] workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%[[DISPATCH_1_EXE]] : !hal.executable)[%[[DISPATCH_1_ORDINAL]]] workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%[[DISPATCH_2_EXE]] : !hal.executable)[%[[DISPATCH_2_ORDINAL]]] workgroups([%c1, %c1, %c1])
//
// CHECK:       util.initializer
// CHECK-DAG:     %[[DISPATCH_0_EXE:.+]] = hal.executable.lookup device(%{{.+}}) executable(@link_executables_linked_spirv) : !hal.executable
// CHECK-DAG:     %[[DISPATCH_1_EXE:.+]] = hal.executable.lookup device(%{{.+}}) executable(@link_executables_linked_spirv) : !hal.executable
// CHECK-DAG:     %[[DISPATCH_2_EXE:.+]] = hal.executable.lookup device(%{{.+}}) executable(@link_executables_linked_spirv) : !hal.executable
// CHECK-DAG:     %[[DISPATCH_0_ORDINAL:.+]] = hal.executable.export.ordinal target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_0)
// CHECK-DAG:     %[[DISPATCH_1_ORDINAL:.+]] = hal.executable.export.ordinal target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_1)
// CHECK-DAG:     %[[DISPATCH_2_ORDINAL:.+]] = hal.executable.export.ordinal target(@link_executables_linked_spirv::@vulkan_spirv_fb::@dispatch_2)
// CHECK:         hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%[[DISPATCH_0_EXE]] : !hal.executable)[%[[DISPATCH_0_ORDINAL]]] workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%[[DISPATCH_1_EXE]] : !hal.executable)[%[[DISPATCH_1_ORDINAL]]] workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%[[DISPATCH_2_EXE]] : !hal.executable)[%[[DISPATCH_2_ORDINAL]]] workgroups([%c1, %c1, %c1])

// -----

// A test case for
// * one variant per executable,
// * different targets for variants across executables.
//
// For such case we need to link into multiple executables, with each one
// having one variant containing all entry points needing the same target.

#vulkan_target_0 = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.spirv.features = ["vulkan-spirv"]}>
#vulkan_target_1 = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.spirv.features = ["vulkan-spirv", "subgroup=1"]}>

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
    hal.executable.condition(%arg0: !hal.device) -> i1 {
      %ok, %value = hal.device.query<%arg0 : !hal.device> key("hal.dispatch" :: "subgroup") : i1, i32 = 0 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = arith.andi %value, %c1_i32 : i32
      %1 = arith.cmpi ne, %0, %c0_i32 : i32
      %2 = arith.andi %ok, %1 : i1
      hal.return %2 : i1
    }
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
    hal.executable.condition(%arg0: !hal.device) -> i1 {
      %ok, %value = hal.device.query<%arg0 : !hal.device> key("hal.dispatch" :: "subgroup") : i1, i32 = 0 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = arith.andi %value, %c1_i32 : i32
      %1 = arith.cmpi ne, %0, %c0_i32 : i32
      %2 = arith.andi %ok, %1 : i1
      hal.return %2 : i1
    }
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
func.func @two_target_environments() -> () {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
  %c1 = arith.constant 1 : index
  %dispatch_0_exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_0) : !hal.executable
  %dispatch_1_exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_1) : !hal.executable
  %dispatch_2_exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_2) : !hal.executable
  %dispatch_3_exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_3) : !hal.executable
  %dispatch_0_ordinal = hal.executable.export.ordinal target(@dispatch_0::@spirv::@dispatch_0) : index
  %dispatch_1_ordinal = hal.executable.export.ordinal target(@dispatch_1::@spirv::@dispatch_1) : index
  %dispatch_2_ordinal = hal.executable.export.ordinal target(@dispatch_2::@spirv::@dispatch_2) : index
  %dispatch_3_ordinal = hal.executable.export.ordinal target(@dispatch_3::@spirv::@dispatch_3) : index
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%dispatch_0_exe : !hal.executable)[%dispatch_0_ordinal] workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%dispatch_1_exe : !hal.executable)[%dispatch_1_ordinal] workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%dispatch_2_exe : !hal.executable)[%dispatch_2_ordinal] workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%dispatch_3_exe : !hal.executable)[%dispatch_3_ordinal] workgroups([%c1, %c1, %c1])
  return
}

//  CHECK-DAG: #[[TARGET0:.+]] = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv"]}
//  CHECK-DAG: #[[TARGET1:.+]] = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv", "subgroup=1"]}

//      CHECK: hal.executable private @link_executables_linked_spirv_0 {
//      CHECK:   hal.executable.variant public @vulkan_spirv_fb target(#[[TARGET0]]) {
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
//      CHECK: }
//      CHECK: hal.executable private @link_executables_linked_spirv_1 {
//      CHECK:   hal.executable.variant public @vulkan_spirv_fb target(#[[TARGET1]]) {
//      CHECK:     hal.executable.condition(%arg0: !hal.device) -> i1
// CHECK-NEXT:       hal.device.query<%arg0 : !hal.device> key("hal.dispatch" :: "subgroup")
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

//      CHECK: func.func @two_target_environments()
//  CHECK-DAG:   %[[DISPATCH_0_EXE:.+]] = hal.executable.lookup device(%{{.+}}) executable(@link_executables_linked_spirv_0) : !hal.executable
//  CHECK-DAG:   %[[DISPATCH_1_EXE:.+]] = hal.executable.lookup device(%{{.+}}) executable(@link_executables_linked_spirv_1) : !hal.executable
//  CHECK-DAG:   %[[DISPATCH_2_EXE:.+]] = hal.executable.lookup device(%{{.+}}) executable(@link_executables_linked_spirv_0) : !hal.executable
//  CHECK-DAG:   %[[DISPATCH_3_EXE:.+]] = hal.executable.lookup device(%{{.+}}) executable(@link_executables_linked_spirv_1) : !hal.executable
//  CHECK-DAG:   %[[DISPATCH_0_ORDINAL:.+]] = hal.executable.export.ordinal target(@link_executables_linked_spirv_0::@vulkan_spirv_fb::@dispatch_0)
//  CHECK-DAG:   %[[DISPATCH_1_ORDINAL:.+]] = hal.executable.export.ordinal target(@link_executables_linked_spirv_1::@vulkan_spirv_fb::@dispatch_1)
//  CHECK-DAG:   %[[DISPATCH_2_ORDINAL:.+]] = hal.executable.export.ordinal target(@link_executables_linked_spirv_0::@vulkan_spirv_fb::@dispatch_2)
//  CHECK-DAG:   %[[DISPATCH_3_ORDINAL:.+]] = hal.executable.export.ordinal target(@link_executables_linked_spirv_1::@vulkan_spirv_fb::@dispatch_3)
//      CHECK:   hal.command_buffer.dispatch{{.+}} target(%[[DISPATCH_0_EXE]] : !hal.executable)[%[[DISPATCH_0_ORDINAL]]]
//      CHECK:   hal.command_buffer.dispatch{{.+}} target(%[[DISPATCH_1_EXE]] : !hal.executable)[%[[DISPATCH_1_ORDINAL]]]
//      CHECK:   hal.command_buffer.dispatch{{.+}} target(%[[DISPATCH_2_EXE]] : !hal.executable)[%[[DISPATCH_2_ORDINAL]]]
//      CHECK:   hal.command_buffer.dispatch{{.+}} target(%[[DISPATCH_3_EXE]] : !hal.executable)[%[[DISPATCH_3_ORDINAL]]]

// -----

// A test case for
// * multiple variants per executable,
// * different targets for variants in the same executable.
//
// For such case we can only link two executables together if they have the
// same set of target requirements.

#vulkan_target_0 = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.spirv.features = ["vulkan-spirv"]}>
#vulkan_target_1 = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.spirv.features = ["vulkan-spirv", "subgroup=1"]}>
#vulkan_target_2 = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.spirv.features = ["vulkan-spirv", "subgroup=2"]}>

#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable private @dispatch_0 {
  hal.executable.variant @spirv_0 target(#vulkan_target_0) {
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
  hal.executable.variant @spirv_1 target(#vulkan_target_1) {
    hal.executable.condition(%arg0: !hal.device) -> i1 {
      %ok, %value = hal.device.query<%arg0 : !hal.device> key("hal.dispatch" :: "subgroup") : i1, i32 = 0 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = arith.andi %value, %c1_i32 : i32
      %1 = arith.cmpi ne, %0, %c0_i32 : i32
      %2 = arith.andi %ok, %1 : i1
      hal.return %2 : i1
    }
    hal.executable.constant.block(%device: !hal.device) -> i32 as "foo" {
      %c2 = arith.constant 2 : i32
      hal.return %c2 : i32
    }
    hal.executable.export @dispatch_0 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c2 = arith.constant 2 : index
      hal.return %c2, %c2, %c2 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
        spirv.func @dispatch_0() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @dispatch_0
        spirv.ExecutionMode @dispatch_0 "LocalSize", 64, 1, 1
      }
    }
  }
}
// dispatch_1 has the same target requirements across all variants like
// dispatch_0. So it can link with dispatch_0.
hal.executable private @dispatch_1 {
  hal.executable.variant @spirv_0 target(#vulkan_target_0) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "baz" {
      %c3 = arith.constant 3 : i32
      hal.return %c3 : i32
    }
    hal.executable.export @dispatch_1 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c3 = arith.constant 3 : index
      hal.return %c3, %c3, %c3 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
        spirv.func @dispatch_1() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @dispatch_1
        spirv.ExecutionMode @dispatch_1 "LocalSize", 4, 8, 1
      }
    }
  }
  hal.executable.variant @spirv_1 target(#vulkan_target_1) {
    hal.executable.condition(%arg0: !hal.device) -> i1 {
      %ok, %value = hal.device.query<%arg0 : !hal.device> key("hal.dispatch" :: "subgroup") : i1, i32 = 0 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = arith.andi %value, %c1_i32 : i32
      %1 = arith.cmpi ne, %0, %c0_i32 : i32
      %2 = arith.andi %ok, %1 : i1
      hal.return %2 : i1
    }
    hal.executable.constant.block(%device: !hal.device) -> i32 as "baz" {
      %c4 = arith.constant 4 : i32
      hal.return %c4 : i32
    }
    hal.executable.export @dispatch_1 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c4 = arith.constant 4 : index
      hal.return %c4, %c4, %c4 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
        spirv.func @dispatch_1() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @dispatch_1
        spirv.ExecutionMode @dispatch_1 "LocalSize", 8, 8, 1
      }
    }
  }
}
// dispatch_2 does not have the same number of variants like dispatch_0 or
// dispatch_1, so it can link with neither.
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
// dispatch_3 have two variants but one of the variants has different target
// requirement. So cannot link either.
hal.executable private @dispatch_3 {
  hal.executable.variant @spirv_0 target(#vulkan_target_0) {
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
  hal.executable.variant @spirv_1 target(#vulkan_target_2) {
    hal.executable.condition(%arg0: !hal.device) -> i1 {
      %ok, %value = hal.device.query<%arg0 : !hal.device> key("hal.dispatch" :: "subgroup") : i1, i32 = 0 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
      %0 = arith.andi %value, %c2_i32 : i32
      %1 = arith.cmpi ne, %0, %c0_i32 : i32
      %2 = arith.andi %ok, %1 : i1
      hal.return %2 : i1
    }
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

//  CHECK-DAG: #[[TARGET0:.+]] = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv"]}
//  CHECK-DAG: #[[TARGET1:.+]] = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv", "subgroup=1"]}
//  CHECK-DAG: #[[TARGET2:.+]] = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv", "subgroup=2"]}

//      CHECK: hal.executable private @link_executables_linked_spirv {
//      CHECK:   hal.executable.variant public @vulkan_spirv_fb_0 target(#[[TARGET0]]) {
//      CHECK:     hal.executable.constant.block(%arg0: !hal.device) -> i32 as "foo"
// CHECK-NEXT:       = arith.constant 1 : i32
//      CHECK:     hal.executable.export public @dispatch_0 ordinal(0)
//      CHECK:       = arith.constant 1 : index
//      CHECK:     hal.executable.constant.block(%arg0: !hal.device) -> i32 as "baz" {
// CHECK-NEXT:       = arith.constant 3 : i32
//      CHECK:     hal.executable.export public @dispatch_1 ordinal(1)
//      CHECK:       = arith.constant 3 : index
//      CHECK:     builtin.module {
//      CHECK:       spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
//      CHECK:         spirv.func @dispatch_0()
//      CHECK:         spirv.EntryPoint "GLCompute" @dispatch_0
//      CHECK:         spirv.ExecutionMode @dispatch_0 "LocalSize", 32, 1, 1
//      CHECK:       }
//      CHECK:       spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
//      CHECK:         spirv.func @dispatch_1()
//      CHECK:         spirv.EntryPoint "GLCompute" @dispatch_1
//      CHECK:         spirv.ExecutionMode @dispatch_1 "LocalSize", 4, 8, 1
//      CHECK:       }
//      CHECK:     }
//      CHECK:   }
//      CHECK:   hal.executable.variant public @vulkan_spirv_fb_1 target(#[[TARGET1]]) {
//      CHECK:     hal.executable.condition(%arg0: !hal.device) -> i1
// CHECK-NEXT:       %{{.+}}, %[[V:.+]] = hal.device.query<%arg0 : !hal.device> key("hal.dispatch" :: "subgroup")
//      CHECK:       %[[TARGET:.+]] = arith.constant 1 : i32
// CHECK-NEXT:       %{{.+}} = arith.andi %[[V]], %[[TARGET]] : i32
//      CHECK:     hal.executable.constant.block(%arg0: !hal.device) -> i32 as "foo"
// CHECK-NEXT:       = arith.constant 2 : i32
//      CHECK:     hal.executable.export public @dispatch_0 ordinal(0)
//      CHECK:       = arith.constant 2 : index
//      CHECK:     hal.executable.constant.block(%arg0: !hal.device) -> i32 as "baz"
// CHECK-NEXT:       = arith.constant 4 : i32
//      CHECK:     hal.executable.export public @dispatch_1 ordinal(1)
//      CHECK:       = arith.constant 4 : index
//      CHECK:     builtin.module {
//      CHECK:       spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
//      CHECK:         spirv.func @dispatch_0()
//      CHECK:         spirv.EntryPoint "GLCompute" @dispatch_0
//      CHECK:         spirv.ExecutionMode @dispatch_0 "LocalSize", 64, 1, 1
//      CHECK:       }
//      CHECK:       spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
//      CHECK:         spirv.func @dispatch_1()
//      CHECK:         spirv.EntryPoint "GLCompute" @dispatch_1
//      CHECK:         spirv.ExecutionMode @dispatch_1 "LocalSize", 8, 8, 1
//      CHECK:       }
//      CHECK:     }
//      CHECK:   }
//      CHECK: }
//      CHECK: hal.executable private @dispatch_2 {
//      CHECK:   hal.executable.variant public @spirv target(#[[TARGET0]]) {
//      CHECK:     hal.executable.export public @dispatch_2 ordinal(0)
//      CHECK:   }
//      CHECK: }
//      CHECK: hal.executable private @dispatch_3 {
//      CHECK:   hal.executable.variant public @spirv_0 target(#[[TARGET0]]) {
//      CHECK:     hal.executable.export public @dispatch_3 ordinal(0)
//      CHECK:   }
//      CHECK:   hal.executable.variant public @spirv_1 target(#[[TARGET2]]) {
//      CHECK:     hal.executable.export public @dispatch_3 ordinal(0)
//      CHECK:   }
//      CHECK: }
