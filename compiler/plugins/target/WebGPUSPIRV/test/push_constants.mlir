// RUN: iree-opt --iree-hal-serialize-all-executables %s | FileCheck %s

#layout = #hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer>]>
hal.executable public @push_constant {
  hal.executable.variant @webgpu target(<"webgpu-spirv", "webgpu-wgsl-fb">) {
    hal.executable.export public @main ordinal(0) layout(#layout) attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], [SPV_KHR_non_semantic_info]> {
        spirv.GlobalVariable @parameters : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, PushConstant>
        spirv.GlobalVariable @output bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, StorageBuffer>
        spirv.func @main() "None" {
          %zero = spirv.Constant 0 : i32
          %parameters = spirv.mlir.addressof @parameters : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, PushConstant>
          %source = spirv.AccessChain %parameters[%zero, %zero] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, PushConstant>, i32, i32 -> !spirv.ptr<i32, PushConstant>
          %value = spirv.Load "PushConstant" %source : i32
          %output = spirv.mlir.addressof @output : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, StorageBuffer>
          %target = spirv.AccessChain %output[%zero, %zero] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
          spirv.Store "StorageBuffer" %target, %value : i32
          spirv.Return
        }
        spirv.EntryPoint "GLCompute" @main
        spirv.ExecutionMode @main "LocalSize", 1, 1, 1
      }
    }
  }
}

// CHECK: hal.executable.binary
// CHECK-SAME: webgpu-wgsl-fb
