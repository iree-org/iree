// TODO(antiagainst): Re-enable SPIR-V linking once the tensorflow integration
// crash is fixed.
// RUN-disabled: iree-opt --split-input-file --iree-hal-link-target-executables='target=vulkan-spirv'  %s | FileCheck %s
// RUN: iree-opt --split-input-file %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb">

#executable_layout_0 = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

#executable_layout_1 = #hal.executable.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable private @call_dispatch_0  {
  hal.executable.variant @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
    hal.executable.export @call_dispatch_0 ordinal(0) layout(#executable_layout_0)
    builtin.module {
      spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spv.func @call_dispatch_0() "None" {
          spv.Return
        }
        spv.EntryPoint "GLCompute" @call_dispatch_0
        spv.ExecutionMode @call_dispatch_0 "LocalSize", 32, 1, 1
      }
    }
  }
}
hal.executable private @call_dispatch_1  {
  hal.executable.variant @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
    hal.executable.export @call_dispatch_1 ordinal(0) layout(#executable_layout_1)
    builtin.module {
      spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spv.func @call_dispatch_1() "None" {
          spv.Return
        }
        spv.EntryPoint "GLCompute" @call_dispatch_1
        spv.ExecutionMode @call_dispatch_1 "LocalSize", 4, 4, 1
      }
    }
  }
}
hal.executable private @call_dispatch_2  {
  hal.executable.variant @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
    hal.executable.export @call_dispatch_2 ordinal(0) layout(#executable_layout_0)
    builtin.module {
      spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spv.func @call_dispatch_2() "None" {
          spv.Return
        }
        spv.EntryPoint "GLCompute" @call_dispatch_2
        spv.ExecutionMode @call_dispatch_2 "LocalSize", 32, 1, 1
      }
    }
  }
}
hal.executable private @call_dispatch_3  {
  hal.executable.variant @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
    hal.executable.export @call_dispatch_3 ordinal(0) layout(#executable_layout_1)
    builtin.module {
      spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spv.func @call_dispatch_3() "None" {
          spv.Return
        }
        spv.EntryPoint "GLCompute" @call_dispatch_3
        spv.ExecutionMode @call_dispatch_3 "LocalSize", 8, 2, 2
      }
    }
  }
}
hal.executable private @call_dispatch_4  {
  hal.executable.variant @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
    hal.executable.export @call_dispatch_4 ordinal(0) layout(#executable_layout_1)
    builtin.module {
      spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spv.func @call_dispatch_4() "None" {
          spv.Return
        }
        spv.EntryPoint "GLCompute" @call_dispatch_4
        spv.ExecutionMode @call_dispatch_4 "LocalSize", 2, 8, 1
      }
    }
  }
}

// Two groups should be created, according to their interfaces.

//      CHECK: hal.executable private @linking_linked_vulkan_0 {
// CHECK-NEXT:   hal.executable.variant public @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
// CHECK-NEXT:     hal.executable.export public @call_dispatch_1 ordinal(0) layout(#executable_layout_0)
// CHECK-NEXT:     hal.executable.export public @call_dispatch_3 ordinal(1) layout(#executable_layout_0)
// CHECK-NEXT:     hal.executable.export public @call_dispatch_4 ordinal(2) layout(#executable_layout_0)
// CHECK-NEXT:     module  {
// CHECK-NEXT:       spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
// CHECK-NEXT:         spv.func @call_dispatch_1() "None" {
// CHECK-NEXT:           spv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spv.EntryPoint "GLCompute" @call_dispatch_1
// CHECK-NEXT:         spv.ExecutionMode @call_dispatch_1 "LocalSize", 4, 4, 1
// CHECK-NEXT:         spv.func @call_dispatch_3() "None" {
// CHECK-NEXT:           spv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spv.EntryPoint "GLCompute" @call_dispatch_3
// CHECK-NEXT:         spv.ExecutionMode @call_dispatch_3 "LocalSize", 8, 2, 2
// CHECK-NEXT:         spv.func @call_dispatch_4() "None" {
// CHECK-NEXT:           spv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spv.EntryPoint "GLCompute" @call_dispatch_4
// CHECK-NEXT:         spv.ExecutionMode @call_dispatch_4 "LocalSize", 2, 8, 1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

//      CHECK: hal.executable private @linking_linked_vulkan {
// CHECK-NEXT:   hal.executable.variant public @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
// CHECK-NEXT:     hal.executable.export public @call_dispatch_0 ordinal(0) layout(#executable_layout_1)
// CHECK-NEXT:     hal.executable.export public @call_dispatch_2 ordinal(1) layout(#executable_layout_1)
// CHECK-NEXT:     module  {
// CHECK-NEXT:       spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
// CHECK-NEXT:         spv.func @call_dispatch_0() "None" {
// CHECK-NEXT:           spv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spv.EntryPoint "GLCompute" @call_dispatch_0
// CHECK-NEXT:         spv.ExecutionMode @call_dispatch_0 "LocalSize", 32, 1, 1
// CHECK-NEXT:         spv.func @call_dispatch_2() "None" {
// CHECK-NEXT:           spv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spv.EntryPoint "GLCompute" @call_dispatch_2
// CHECK-NEXT:         spv.ExecutionMode @call_dispatch_2 "LocalSize", 32, 1, 1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
