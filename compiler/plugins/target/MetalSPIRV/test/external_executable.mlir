// RUN: iree-opt --split-input-file --iree-hal-transformation-pipeline %s | FileCheck %s

// Test that external Metal executables (with hal.executable.objects) are
// serialized correctly, producing a valid metal-msl-fb binary.
// This mirrors the Vulkan external executable test pattern.

module attributes {
  hal.device.targets = [
    #hal.device.target<"metal", [
      #hal.executable.target<"metal-spirv", "metal-msl-fb", {
        iree_codegen.target_info = #iree_gpu.target<arch = "apple", features = "spirv:v1.3,cap:Shader", wgp = <
          compute = fp32|fp16|int32, storage = b32|b16, subgroup = shuffle|arithmetic,
          subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
          max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 32768,
          max_workgroup_counts = [65535, 65535, 65535]>>
      }>
    ]> : !hal.device
  ]
} {

// External executable with an MSL source object.
// The compiler should embed the MSL source in the flatbuffer without
// attempting SPIR-V codegen.
hal.executable public @extern_dispatch {
  hal.executable.variant public @metal_msl_fb target(<"metal-spirv", "metal-msl-fb", {
    iree_codegen.target_info = #iree_gpu.target<arch = "apple", features = "spirv:v1.3,cap:Shader", wgp = <
      compute = fp32|fp16|int32, storage = b32|b16, subgroup = shuffle|arithmetic,
      subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 32768,
      max_workgroup_counts = [65535, 65535, 65535]>>
  }>) objects([
    #hal.executable.object<{
      // Inline MSL source as data.
      data = dense<[
        // "#include <metal_stdlib>\nusing namespace metal;\n"
        // "kernel void entry(device float* a [[buffer(0)]]) { a[0] = 1.0; }\n"
        0x23, 0x69, 0x6E, 0x63, 0x6C, 0x75, 0x64, 0x65
      ]> : vector<8xi8>
    }>
  ]) {
    hal.executable.export public @entry ordinal(0)
      layout(#hal.pipeline.layout<bindings = [
        #hal.pipeline.binding<storage_buffer, Indirect>
      ], flags = Indirect>)
      attributes {
        workgroup_size = [32 : index, 1 : index, 1 : index]
      }
  }
}

// CHECK:        hal.executable.binary public @metal_msl_fb attributes {
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = "metal-msl-fb"

}

// -----

// Test external executable with a pre-compiled .metallib binary object.
// The compiler should detect the "MTLB" magic and embed it as a metallib
// (not MSL source) in the flatbuffer.

module attributes {
  hal.device.targets = [
    #hal.device.target<"metal", [
      #hal.executable.target<"metal-spirv", "metal-msl-fb", {
        iree_codegen.target_info = #iree_gpu.target<arch = "apple", features = "spirv:v1.3,cap:Shader", wgp = <
          compute = fp32|fp16|int32, storage = b32|b16, subgroup = shuffle|arithmetic,
          subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
          max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 32768,
          max_workgroup_counts = [65535, 65535, 65535]>>
      }>
    ]> : !hal.device
  ]
} {

// External executable with a metallib-like binary object (MTLB magic header).
hal.executable public @extern_metallib {
  hal.executable.variant public @metal_msl_fb target(<"metal-spirv", "metal-msl-fb", {
    iree_codegen.target_info = #iree_gpu.target<arch = "apple", features = "spirv:v1.3,cap:Shader", wgp = <
      compute = fp32|fp16|int32, storage = b32|b16, subgroup = shuffle|arithmetic,
      subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 32768,
      max_workgroup_counts = [65535, 65535, 65535]>>
  }>) objects([
    #hal.executable.object<{
      // Bytes starting with "MTLB" magic (0x4D544C42) to trigger metallib path.
      data = dense<[
        0x4D, 0x54, 0x4C, 0x42, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
      ]> : vector<16xi8>
    }>
  ]) {
    hal.executable.export public @entry ordinal(0)
      layout(#hal.pipeline.layout<bindings = [
        #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
        #hal.pipeline.binding<storage_buffer, Indirect>
      ], flags = Indirect>)
      attributes {
        workgroup_size = [64 : index, 1 : index, 1 : index]
      }
  }
}

// CHECK:        hal.executable.binary public @metal_msl_fb attributes {
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = "metal-msl-fb"

}
