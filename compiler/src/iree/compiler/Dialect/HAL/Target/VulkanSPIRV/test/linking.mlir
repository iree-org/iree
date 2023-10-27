// TODO(antiagainst): Re-enable SPIR-V linking once the tensorflow integration
// crash is fixed.
// RUN-disabled: iree-opt --split-input-file --iree-hal-link-target-executables='target=vulkan-spirv'  %s | FileCheck %s
// RUN: iree-opt --split-input-file %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb">

#pipeline_layout_0 = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

#pipeline_layout_1 = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable private @call_dispatch_0  {
  hal.executable.variant @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
    hal.executable.export @call_dispatch_0 ordinal(0) layout(#pipeline_layout_0) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spirv.func @call_dispatch_0() "None" {
          spirv.Return
        }
        spirv.EntryPoint "GLCompute" @call_dispatch_0
        spirv.ExecutionMode @call_dispatch_0 "LocalSize", 32, 1, 1
      }
    }
  }
}
hal.executable private @call_dispatch_1  {
  hal.executable.variant @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
    hal.executable.export @call_dispatch_1 ordinal(0) layout(#pipeline_layout_1) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spirv.func @call_dispatch_1() "None" {
          spirv.Return
        }
        spirv.EntryPoint "GLCompute" @call_dispatch_1
        spirv.ExecutionMode @call_dispatch_1 "LocalSize", 4, 4, 1
      }
    }
  }
}
hal.executable private @call_dispatch_2  {
  hal.executable.variant @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
    hal.executable.export @call_dispatch_2 ordinal(0) layout(#pipeline_layout_0) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spirv.func @call_dispatch_2() "None" {
          spirv.Return
        }
        spirv.EntryPoint "GLCompute" @call_dispatch_2
        spirv.ExecutionMode @call_dispatch_2 "LocalSize", 32, 1, 1
      }
    }
  }
}
hal.executable private @call_dispatch_3  {
  hal.executable.variant @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
    hal.executable.export @call_dispatch_3 ordinal(0) layout(#pipeline_layout_1) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spirv.func @call_dispatch_3() "None" {
          spirv.Return
        }
        spirv.EntryPoint "GLCompute" @call_dispatch_3
        spirv.ExecutionMode @call_dispatch_3 "LocalSize", 8, 2, 2
      }
    }
  }
}
hal.executable private @call_dispatch_4  {
  hal.executable.variant @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
    hal.executable.export @call_dispatch_4 ordinal(0) layout(#pipeline_layout_1) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spirv.func @call_dispatch_4() "None" {
          spirv.Return
        }
        spirv.EntryPoint "GLCompute" @call_dispatch_4
        spirv.ExecutionMode @call_dispatch_4 "LocalSize", 2, 8, 1
      }
    }
  }
}

// Two groups should be created, according to their interfaces.

//      CHECK: hal.executable private @linking_linked_vulkan_0 {
// CHECK-NEXT:   hal.executable.variant public @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
// CHECK-NEXT:     hal.executable.export public @call_dispatch_1 ordinal(0) layout(#pipeline_layout_0)
// CHECK-NEXT:     hal.executable.export public @call_dispatch_3 ordinal(1) layout(#pipeline_layout_0)
// CHECK-NEXT:     hal.executable.export public @call_dispatch_4 ordinal(2) layout(#pipeline_layout_0)
// CHECK-NEXT:     module  {
// CHECK-NEXT:       spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
// CHECK-NEXT:         spirv.func @call_dispatch_1() "None" {
// CHECK-NEXT:           spirv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spirv.EntryPoint "GLCompute" @call_dispatch_1
// CHECK-NEXT:         spirv.ExecutionMode @call_dispatch_1 "LocalSize", 4, 4, 1
// CHECK-NEXT:         spirv.func @call_dispatch_3() "None" {
// CHECK-NEXT:           spirv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spirv.EntryPoint "GLCompute" @call_dispatch_3
// CHECK-NEXT:         spirv.ExecutionMode @call_dispatch_3 "LocalSize", 8, 2, 2
// CHECK-NEXT:         spirv.func @call_dispatch_4() "None" {
// CHECK-NEXT:           spirv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spirv.EntryPoint "GLCompute" @call_dispatch_4
// CHECK-NEXT:         spirv.ExecutionMode @call_dispatch_4 "LocalSize", 2, 8, 1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

//      CHECK: hal.executable private @linking_linked_vulkan {
// CHECK-NEXT:   hal.executable.variant public @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
// CHECK-NEXT:     hal.executable.export public @call_dispatch_0 ordinal(0) layout(#pipeline_layout_1)
// CHECK-NEXT:     hal.executable.export public @call_dispatch_2 ordinal(1) layout(#pipeline_layout_1)
// CHECK-NEXT:     module  {
// CHECK-NEXT:       spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
// CHECK-NEXT:         spirv.func @call_dispatch_0() "None" {
// CHECK-NEXT:           spirv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spirv.EntryPoint "GLCompute" @call_dispatch_0
// CHECK-NEXT:         spirv.ExecutionMode @call_dispatch_0 "LocalSize", 32, 1, 1
// CHECK-NEXT:         spirv.func @call_dispatch_2() "None" {
// CHECK-NEXT:           spirv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spirv.EntryPoint "GLCompute" @call_dispatch_2
// CHECK-NEXT:         spirv.ExecutionMode @call_dispatch_2 "LocalSize", 32, 1, 1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
