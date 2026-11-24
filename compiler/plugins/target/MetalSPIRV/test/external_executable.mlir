// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(iree-hal-serialize-target-executables{target=metal-spirv}))' \
// RUN:   --iree-metal-compile-to-metallib=false --verify-diagnostics %s | FileCheck %s

// External variants (no inner module, as from hal.dispatch.extern or
// hal.executable.source) are serialized from their single object: a
// precompiled .metallib or MSL source. The objects are embedded so the test
// needs no files and no Metal toolchain.

// A .metallib object, recognized by its "MTLB" magic. Its content is copied
// as is, so the magic alone is enough here.

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"metal-spirv", "metal-msl-fb">

// CHECK-LABEL: hal.executable public @metallib_object
//       CHECK:   hal.executable.binary public @metal_msl_fb <
//  CHECK-SAME:     format = "metal-msl-fb"
//  CHECK-SAME:     mime_type = "application/x-flatbuffers"
hal.executable public @metallib_object {
  hal.executable.variant public @metal_msl_fb target(#executable_target) objects([
    #hal.executable.object<{data = dense<[77, 84, 76, 66]> : vector<4xi8>}>
  ]) {
    hal.executable.export public @k ordinal(0) layout(#pipeline_layout) <
      workgroup_size = [64 : index, 1 : index, 1 : index]
    >
  }
}

// -----

// An MSL source object: `kernel void k() {}`.

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"metal-spirv", "metal-msl-fb">

// CHECK-LABEL: hal.executable public @msl_source_object
//       CHECK:   hal.executable.binary public @metal_msl_fb <
//  CHECK-SAME:     format = "metal-msl-fb"
hal.executable public @msl_source_object {
  hal.executable.variant public @metal_msl_fb target(#executable_target) objects([
    #hal.executable.object<{data = dense<[107, 101, 114, 110, 101, 108, 32, 118, 111, 105, 100, 32, 107, 40, 41, 32, 123, 125, 10]> : vector<19xi8>}>
  ]) {
    hal.executable.export public @k ordinal(0) layout(#pipeline_layout) <
      workgroup_size = [64 : index, 1 : index, 1 : index]
    >
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"metal-spirv", "metal-msl-fb">

hal.executable public @no_object {
  // expected-error @+2 {{external Metal variants need exactly one object}}
  // expected-error @+1 {{failed to serialize executable for target backend metal-spirv}}
  hal.executable.variant public @metal_msl_fb target(#executable_target) {
    hal.executable.export public @k ordinal(0) layout(#pipeline_layout) <
      workgroup_size = [64 : index, 1 : index, 1 : index]
    >
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"metal-spirv", "metal-msl-fb">

hal.executable public @missing_workgroup_size {
  // expected-error @+1 {{failed to serialize executable for target backend metal-spirv}}
  hal.executable.variant public @metal_msl_fb target(#executable_target) objects([
    #hal.executable.object<{data = dense<[77, 84, 76, 66]> : vector<4xi8>}>
  ]) {
    // expected-error @+1 {{external Metal exports need a 3D workgroup_size}}
    hal.executable.export public @k ordinal(0) layout(#pipeline_layout)
  }
}

// -----

// Exports declared out of ordinal order. The runtime looks pipelines up by
// ordinal, so they are placed at their ordinals, not in declaration order.

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"metal-spirv", "metal-msl-fb">

// CHECK-LABEL: hal.executable public @unordered_exports
//       CHECK:   hal.executable.binary public @metal_msl_fb <
hal.executable public @unordered_exports {
  hal.executable.variant public @metal_msl_fb target(#executable_target) objects([
    #hal.executable.object<{data = dense<[77, 84, 76, 66]> : vector<4xi8>}>
  ]) {
    hal.executable.export public @k1 ordinal(1) layout(#pipeline_layout) <
      workgroup_size = [64 : index, 1 : index, 1 : index]
    >
    hal.executable.export public @k0 ordinal(0) layout(#pipeline_layout) <
      workgroup_size = [64 : index, 1 : index, 1 : index]
    >
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"metal-spirv", "metal-msl-fb">

hal.executable public @duplicate_ordinal {
  // expected-error @+1 {{failed to serialize executable for target backend metal-spirv}}
  hal.executable.variant public @metal_msl_fb target(#executable_target) objects([
    #hal.executable.object<{data = dense<[77, 84, 76, 66]> : vector<4xi8>}>
  ]) {
    hal.executable.export public @k0 ordinal(0) layout(#pipeline_layout) <
      workgroup_size = [64 : index, 1 : index, 1 : index]
    >
    // expected-error @+1 {{export ordinal 0 is out of range or duplicated}}
    hal.executable.export public @k1 ordinal(0) layout(#pipeline_layout) <
      workgroup_size = [64 : index, 1 : index, 1 : index]
    >
  }
}
