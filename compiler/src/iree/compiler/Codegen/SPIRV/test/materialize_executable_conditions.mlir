// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-materialize-executable-conditions)))' --mlir-print-local-scope %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  <0, bindings = [
    <0, storage_buffer, ReadOnly>,
    <1, storage_buffer, ReadOnly>,
    <2, storage_buffer>
  ]>
]>

#indirect_pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  <0, bindings = [
    <0, storage_buffer, ReadOnly>,
    <1, storage_buffer, ReadOnly>,
    <2, storage_buffer>
  ], flags = Indirect>
]>

hal.executable private @dispatch_executable {
  // CHECK-LABEL: hal.executable.variant public @test_assumed_capabilities
  //  CHECK-SAME: target(<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv"]}>)
  //   CHECK-NOT:   hal.executable.condition
  hal.executable.variant public @test_assumed_capabilities target(
      #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
        spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, GroupNonUniform], []>, #spirv.resource_limits<>>
      }>
    ) {
    hal.executable.export public @test_assumed_capabilities ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
        spirv.func @test_assumed_capabilities() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @test_assumed_capabilities
        spirv.ExecutionMode @test_assumed_capabilities "LocalSize", 64, 1, 1
      }
    }
  }

  // CHECK-LABEL: hal.executable.variant public @test_subgroup_capabilities
  //  CHECK-SAME: target(<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv", "subgroup.ops=3"]}>)
  //  CHECK-NEXT:   hal.executable.condition(%[[DEV:.+]]: !hal.device) -> i1 {
  //  CHECK-NEXT:   %[[T:.+]] = arith.constant true
  //  CHECK-NEXT:   %[[OK:.+]], %[[V:.+]] = hal.device.query<%[[DEV]] : !hal.device>
  //  CHECK-SAME:     key("hal.dispatch" :: "subgroup.ops") : i1, i32 = 0 : i32
  //  CHECK-NEXT:   %[[ZERO:.+]] = arith.constant 0 : i32
  //  CHECK-NEXT:   %[[TARGET:.+]] = arith.constant 3 : i32
  //  CHECK-NEXT:   %[[CHECK:.+]] = arith.andi %[[V]], %[[TARGET]] : i32
  //  CHECK-NEXT:   %[[CMP:.+]] = arith.cmpi ne, %[[CHECK]], %[[ZERO]] : i32
  //  CHECK-NEXT:   %[[AND:.+]] = arith.andi %[[OK]], %[[CMP]] : i1
  //  CHECK-NEXT:   %[[RESULT:.+]] = arith.andi %[[T]], %[[AND]] : i1
  //  CHECK-NEXT:   hal.return %[[RESULT]] : i1
  //  CHECK-NEXT: }
  hal.executable.variant public @test_subgroup_capabilities target(
      #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
        spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [GroupNonUniformShuffle, GroupNonUniformArithmetic], []>, #spirv.resource_limits<>>
      }>
    ) {
    hal.executable.export public @test_subgroup_capabilities ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [GroupNonUniformShuffle, GroupNonUniformArithmetic], []> {
        spirv.func @test_subgroup_capabilities() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @test_subgroup_capabilities
        spirv.ExecutionMode @test_subgroup_capabilities "LocalSize", 64, 1, 1
      }
    }
  }

  // CHECK-LABEL: hal.executable.variant public @test_8bit_storage_capabilities
  //  CHECK-SAME: target(<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv", "storage.bitwidths=1"]}>)
  //  CHECK-NEXT:   hal.executable.condition(%[[DEV:.+]]: !hal.device) -> i1 {
  //  CHECK-NEXT:   %[[T:.+]] = arith.constant true
  //  CHECK-NEXT:   %[[OK:.+]], %[[V:.+]] = hal.device.query<%[[DEV]] : !hal.device>
  //  CHECK-SAME:     key("hal.dispatch" :: "storage.bitwidths") : i1, i32 = 0 : i32
  //  CHECK-NEXT:   %[[ZERO:.+]] = arith.constant 0 : i32
  //  CHECK-NEXT:   %[[TARGET:.+]] = arith.constant 1 : i32
  //  CHECK-NEXT:   %[[CHECK:.+]] = arith.andi %[[V]], %[[TARGET]] : i32
  //  CHECK-NEXT:   %[[CMP:.+]] = arith.cmpi ne, %[[CHECK]], %[[ZERO]] : i32
  //  CHECK-NEXT:   %[[AND:.+]] = arith.andi %[[OK]], %[[CMP]] : i1
  //  CHECK-NEXT:   %[[RESULT:.+]] = arith.andi %[[T]], %[[AND]] : i1
  //  CHECK-NEXT:   hal.return %[[RESULT]] : i1
  //  CHECK-NEXT: }
  hal.executable.variant public @test_8bit_storage_capabilities target(
      #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
        spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [UniformAndStorageBuffer8BitAccess, StorageBuffer8BitAccess], []>, #spirv.resource_limits<>>
      }>
    ) {
    hal.executable.export public @test_8bit_storage_capabilities ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires
        #spirv.vce<v1.0, [UniformAndStorageBuffer8BitAccess, StorageBuffer8BitAccess], []> {
        spirv.func @test_8bit_storage_capabilities() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @test_8bit_storage_capabilities
        spirv.ExecutionMode @test_8bit_storage_capabilities "LocalSize", 64, 1, 1
      }
    }
  }

  // CHECK-LABEL: hal.executable.variant public @test_16bit_storage_capabilities
  //  CHECK-SAME: target(<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv", "storage.bitwidths=2"]}>)
  //  CHECK-NEXT:   hal.executable.condition(%[[DEV:.+]]: !hal.device) -> i1 {
  //  CHECK-NEXT:   %[[T:.+]] = arith.constant true
  //  CHECK-NEXT:   %[[OK:.+]], %[[V:.+]] = hal.device.query<%[[DEV]] : !hal.device>
  //  CHECK-SAME:     key("hal.dispatch" :: "storage.bitwidths") : i1, i32 = 0 : i32
  //  CHECK-NEXT:   %[[ZERO:.+]] = arith.constant 0 : i32
  //  CHECK-NEXT:   %[[TARGET:.+]] = arith.constant 2 : i32
  //  CHECK-NEXT:   %[[CHECK:.+]] = arith.andi %[[V]], %[[TARGET]] : i32
  //  CHECK-NEXT:   %[[CMP:.+]] = arith.cmpi ne, %[[CHECK]], %[[ZERO]] : i32
  //  CHECK-NEXT:   %[[AND:.+]] = arith.andi %[[OK]], %[[CMP]] : i1
  //  CHECK-NEXT:   %[[RESULT:.+]] = arith.andi %[[T]], %[[AND]] : i1
  //  CHECK-NEXT:   hal.return %[[RESULT]] : i1
  //  CHECK-NEXT: }
  hal.executable.variant public @test_16bit_storage_capabilities target(
      #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
        spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [StorageBuffer16BitAccess, StorageUniform16], []>, #spirv.resource_limits<>>
      }>
    ) {
    hal.executable.export public @test_16bit_storage_capabilities ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires
        #spirv.vce<v1.0, [StorageBuffer16BitAccess, StorageUniform16], []> {
        spirv.func @test_16bit_storage_capabilities() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @test_16bit_storage_capabilities
        spirv.ExecutionMode @test_16bit_storage_capabilities "LocalSize", 64, 1, 1
      }
    }
  }

  // CHECK-LABEL: hal.executable.variant public @test_int_compute_capabilities
  //  CHECK-SAME: target(<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv", "compute.bitwidths.int=7"]}>)
  //       CHECK:   %{{.+}}, %[[V:.+]] = hal.device.query<%{{.+}} : !hal.device>
  //  CHECK-SAME:     key("hal.dispatch" :: "compute.bitwidths.int") : i1, i32 = 0 : i32
  //       CHECK:   %[[TARGET:.+]] = arith.constant 7 : i32
  //       CHECK:   %{{.+}} = arith.andi %[[V]], %[[TARGET]] : i32
  hal.executable.variant public @test_int_compute_capabilities target(
      #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
        spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Int64, Int16, Int8], []>, #spirv.resource_limits<>>
      }>
    ) {
    hal.executable.export public @test_int_compute_capabilities ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Int64, Int16, Int8], []> {
        spirv.func @test_int_compute_capabilities() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @test_int_compute_capabilities
        spirv.ExecutionMode @test_int_compute_capabilities "LocalSize", 64, 1, 1
      }
    }
  }

  // CHECK-LABEL: hal.executable.variant public @test_float_compute_capabilities
  //  CHECK-SAME: target(<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv", "compute.bitwidths.fp=3"]}>)
  //       CHECK:   %{{.+}}, %[[V:.+]] = hal.device.query<%{{.+}} : !hal.device>
  //  CHECK-SAME:     key("hal.dispatch" :: "compute.bitwidths.fp") : i1, i32 = 0 : i32
  //       CHECK:   %[[TARGET:.+]] = arith.constant 3 : i32
  //       CHECK:   %{{.+}} = arith.andi %[[V]], %[[TARGET]] : i32
  hal.executable.variant public @test_float_compute_capabilities target(
      #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
        spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Float16, Float64], []>, #spirv.resource_limits<>>
      }>
    ) {
    hal.executable.export public @test_float_compute_capabilities ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Float16, Float64], []> {
        spirv.func @test_float_compute_capabilities() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @test_float_compute_capabilities
        spirv.ExecutionMode @test_float_compute_capabilities "LocalSize", 64, 1, 1
      }
    }
  }

  // CHECK-LABEL: hal.executable.variant public @test_dot_product_capabilities
  //  CHECK-SAME: target(<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv", "dotprod.ops=1"]}>)
  //       CHECK:   %{{.+}}, %[[V:.+]] = hal.device.query<%{{.+}} : !hal.device>
  //  CHECK-SAME:     key("hal.dispatch" :: "dotprod.ops") : i1, i32 = 0 : i32
  //       CHECK:   %[[TARGET:.+]] = arith.constant 1 : i32
  //       CHECK:   %{{.+}} = arith.andi %[[V]], %[[TARGET]] : i32
  hal.executable.variant public @test_dot_product_capabilities target(
      #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
        spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [DotProduct, DotProductInput4x8Bit], []>, #spirv.resource_limits<>>
      }>
    ) {
    hal.executable.export public @test_dot_product_capabilities ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [DotProduct, DotProductInput4x8Bit], []> {
        spirv.func @test_dot_product_capabilities() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @test_dot_product_capabilities
        spirv.ExecutionMode @test_dot_product_capabilities "LocalSize", 64, 1, 1
      }
    }
  }

  // CHECK-LABEL: hal.executable.variant public @test_cooperative_matrix_capabilities
  //  CHECK-SAME: target(<"vulkan-spirv", "vulkan-spirv-fb", {iree.spirv.features = ["vulkan-spirv", "coopmatrix.ops=1"]}>)
  //       CHECK:   %{{.+}}, %[[V:.+]] = hal.device.query<%{{.+}} : !hal.device>
  //  CHECK-SAME:     key("hal.dispatch" :: "coopmatrix.ops") : i1, i32 = 0 : i32
  //       CHECK:   %[[TARGET:.+]] = arith.constant 1 : i32
  //       CHECK:   %{{.+}} = arith.andi %[[V]], %[[TARGET]] : i32
  hal.executable.variant public @test_cooperative_matrix_capabilities target(
      #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
        spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [CooperativeMatrixKHR], []>, #spirv.resource_limits<>>
      }>
    ) {
    hal.executable.export public @test_cooperative_matrix_capabilities ordinal(0) layout(#pipeline_layout) attributes {
      iree.spirv.coopmatrix.shape = array<i64: 16, 16, 16>, iree.spirv.coopmatrix.type = [f16, f16]
    } {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [CooperativeMatrixKHR], []> {
        spirv.func @test_cooperative_matrix_capabilities() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @test_cooperative_matrix_capabilities
        spirv.ExecutionMode @test_cooperative_matrix_capabilities "LocalSize", 64, 1, 1
      }
    }
  }

  // CHECK-LABEL: hal.executable.variant public @test_address_capabilities
  //  CHECK-SAME: target(<"vulkan-spirv", "vulkan-spirv-fb-ptr",
  //  CHECK-SAME:   {hal.bindings.indirect, iree.spirv.features = ["vulkan-spirv", "compute.bitwidths.int=4", "address.mode=1"]}>)
  //       CHECK:   %{{.+}}, %[[V0:.+]] = hal.device.query<%{{.+}} : !hal.device>
  //  CHECK-SAME:     key("hal.dispatch" :: "compute.bitwidths.int") : i1, i32 = 0 : i32
  //       CHECK:   %[[TARGET0:.+]] = arith.constant 4 : i32
  //       CHECK:   %{{.+}} = arith.andi %[[V0]], %[[TARGET0]] : i32
  //       CHECK:   %{{.+}}, %[[V1:.+]] = hal.device.query<%{{.+}} : !hal.device>
  //  CHECK-SAME:     key("hal.dispatch" :: "address.mode") : i1, i32 = 0 : i32
  //       CHECK:   %[[TARGET1:.+]] = arith.constant 1 : i32
  //       CHECK:   %{{.+}} = arith.andi %[[V1]], %[[TARGET1]] : i32
  hal.executable.variant public @test_address_capabilities target(
      #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb-ptr", {
        spirv.target_env = #spirv.target_env<#spirv.vce<v1.5,
                                                        [Int64, PhysicalStorageBufferAddresses],
                                                        [SPV_KHR_physical_storage_buffer]>,
                                             #spirv.resource_limits<>>,
        hal.bindings.indirect}>
    ) {
    hal.executable.export public @test_address_capabilities ordinal(0) layout(#indirect_pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      spirv.module Physical64 GLSL450 requires
          #spirv.vce<v1.5, [Int64, PhysicalStorageBufferAddresses], [SPV_KHR_physical_storage_buffer]> {
        spirv.func @test_address_capabilities() "None" { spirv.Return }
        spirv.EntryPoint "GLCompute" @test_address_capabilities
        spirv.ExecutionMode @test_address_capabilities "LocalSize", 64, 1, 1
      }
    }
  }
}
