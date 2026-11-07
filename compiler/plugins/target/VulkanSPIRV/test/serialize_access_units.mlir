// RUN: iree-opt --iree-hal-serialize-all-executables %s -o %t
// RUN: dump_executable_metadata %t | FileCheck %s

// Exports with identical HAL layouts but different storage access units must
// retain distinct descriptor layouts. A third export shares the word layout.
// The word storage is referenced only from a helper, and its first struct
// member is narrower than the second. An empty struct must not be indexed.
// CHECK-LABEL: binary @spirv version 1
// CHECK: "entry_point": "byte"
// CHECK: "pipeline_layout_ordinal": 0
// CHECK: "entry_point": "word"
// CHECK: "pipeline_layout_ordinal": 1
// CHECK: "entry_point": "word_again"
// CHECK: "pipeline_layout_ordinal": 1
// CHECK: "descriptor_set_layouts": [
// CHECK: "minimum_access_bytes": 1
// CHECK: "minimum_access_bytes": 4
// CHECK: "minimum_access_bytes": 1
hal.executable @access_units {
  hal.executable.variant @spirv target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @byte ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
    hal.executable.export public @word ordinal(1) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
    hal.executable.export public @word_again ordinal(2) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader, Int8], []> {
        spirv.GlobalVariable @bytes bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i8, stride=1> [0])>, StorageBuffer>
        spirv.GlobalVariable @mixed bind(0, 0) : !spirv.ptr<!spirv.struct<(i8 [0], i32 [4])>, StorageBuffer>
        spirv.GlobalVariable @empty bind(0, 1) : !spirv.ptr<!spirv.struct<()>, StorageBuffer>
        spirv.func @byte() "None" {
          %0 = spirv.mlir.addressof @bytes : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i8, stride=1> [0])>, StorageBuffer>
          spirv.Return
        }
        spirv.func @helper() "None" {
          %0 = spirv.mlir.addressof @mixed : !spirv.ptr<!spirv.struct<(i8 [0], i32 [4])>, StorageBuffer>
          %1 = spirv.mlir.addressof @empty : !spirv.ptr<!spirv.struct<()>, StorageBuffer>
          spirv.Return
        }
        spirv.func @word() "None" {
          spirv.FunctionCall @helper() : () -> ()
          spirv.Return
        }
        spirv.func @word_again() "None" {
          spirv.FunctionCall @word() : () -> ()
          spirv.Return
        }
        spirv.EntryPoint "GLCompute" @byte
        spirv.EntryPoint "GLCompute" @word
        spirv.EntryPoint "GLCompute" @word_again
        spirv.ExecutionMode @byte "LocalSize", 1, 1, 1
        spirv.ExecutionMode @word "LocalSize", 1, 1, 1
        spirv.ExecutionMode @word_again "LocalSize", 1, 1, 1
      }
    }
  }
}

// BDA physical pointers have no descriptor globals at serialization time.
// Internal bindings carry the target's narrow-storage access contract.
// A single export may omit its ordinal.
// CHECK-LABEL: binary @bda_word version 1
// CHECK: "bda_dispatch_layout": {
// CHECK: "minimum_access_bytes": 4
hal.executable @bda_word {
  hal.executable.variant @bda_word target(<"vulkan-spirv", "vulkan-spirv-bda-v1">) {
    hal.executable.export public @main layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>)
    builtin.module {
      spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.3, [Shader, PhysicalStorageBufferAddresses], [SPV_KHR_physical_storage_buffer]> {
        spirv.func @main() "None" {
          spirv.Return
        }
        spirv.EntryPoint "GLCompute" @main
        spirv.ExecutionMode @main "LocalSize", 1, 1, 1
      }
    }
  }
}

// BDA physical pointers have no descriptor globals at serialization time.
// Internal bindings carry the target's narrow-storage access contract.
// A single export may omit its ordinal.
// CHECK-LABEL: binary @bda_native version 1
// CHECK: "bda_dispatch_layout": {
// CHECK: "minimum_access_bytes": 1
hal.executable @bda_native {
  hal.executable.variant @bda_native target(#hal.executable.target<"vulkan-spirv", "vulkan-spirv-bda-v1", {iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <compute = fp16|fp32|int16|int32, storage = b8|b16|b32, subgroup = none, subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>}>) {
    hal.executable.export public @main layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>)
    builtin.module {
      spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.3, [Shader, PhysicalStorageBufferAddresses], [SPV_KHR_physical_storage_buffer]> {
        spirv.func @main() "None" {
          spirv.Return
        }
        spirv.EntryPoint "GLCompute" @main
        spirv.ExecutionMode @main "LocalSize", 1, 1, 1
      }
    }
  }
}

// External objects declare their own layout; absent access metadata stays byte
// granular instead of borrowing internal-codegen target assumptions.
// CHECK-LABEL: binary @bda_external version 1
// CHECK: "bda_dispatch_layout": {
// CHECK: "minimum_access_bytes": 1
hal.executable @bda_external {
  hal.executable.variant @bda_external target(<"vulkan-spirv", "vulkan-spirv-bda-v1">) objects([
    #hal.executable.object<{path = "noop.spv", data = dense<"0x0302230700060100000000000500000000000000110002000100000011000200e31400000e000300e4140000010000000f00050005000000030000006d61696e0000000010000600030000001100000001000000010000000100000013000200010000002100030002000000010000003600050001000000030000000000000002000000f800020004000000fd00010038000100"> : vector<148xi8>}>
  ]) {
    hal.executable.export public @main ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>)
  }
}
