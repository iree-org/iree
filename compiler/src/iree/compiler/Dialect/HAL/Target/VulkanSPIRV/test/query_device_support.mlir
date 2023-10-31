// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(iree-hal-translate-executables))" %s | FileCheck %s

hal.executable private @extern_dispatch_all_subgroup {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6, [
        Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic,
        GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative,
        GroupNonUniformClustered, GroupNonUniformQuad], []>,
        api=Vulkan, AMD:DiscreteGPU,
        #spirv.resource_limits<>>}>)
      objects([#hal.executable.object<{path = "/does/not/exist.spv"}>]) {
    hal.executable.export public @main ordinal(0) layout(
        #hal.pipeline.layout<push_constants = 1, sets = [
          <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer>
          ]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %c1 = arith.constant 1 : index 
      hal.return %c1, %c1, %c1 : index, index, index 
    }
  }
}

// CHECK-LABEL: hal.executable private @extern_dispatch_all_subgroup
//       CHECK:  hal.executable.condition
//  CHECK-SAME:    %[[DEVICE:.+]]: !hal.device) -> i1
//       CHECK:    %[[OK:.+]], %[[QUERY:.+]] = hal.device.query<%arg0 : !hal.device> key("hal.device.vulkan.subgroup_operations" :: "255") : i1, i1 = false
//       CHECK:    %[[SUCCESS:.+]] = arith.andi %[[OK]], %[[QUERY]] : i1
//       CHECK:    hal.return %[[SUCCESS]] : i1

// -----

hal.executable private @extern_dispatch_ballot_vote {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6, [
        Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformBallot], []>,
        api=Vulkan, AMD:DiscreteGPU,
        #spirv.resource_limits<>>}>)
      objects([#hal.executable.object<{path = "/does/not/exist.spv"}>]) {
    hal.executable.export public @main ordinal(0) layout(
        #hal.pipeline.layout<push_constants = 1, sets = [
          <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer>
          ]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %c1 = arith.constant 1 : index 
      hal.return %c1, %c1, %c1 : index, index, index 
    }
  }
}

// CHECK-LABEL: hal.executable private @extern_dispatch_ballot_vote
//       CHECK:  hal.executable.condition
//  CHECK-SAME:    %[[DEVICE:.+]]: !hal.device) -> i1
//       CHECK:    %[[OK:.+]], %[[QUERY:.+]] = hal.device.query<%arg0 : !hal.device> key("hal.device.vulkan.subgroup_operations" :: "11") : i1, i1 = false
//       CHECK:    %[[SUCCESS:.+]] = arith.andi %[[OK]], %[[QUERY]] : i1
//       CHECK:    hal.return %[[SUCCESS]] : i1

// -----

hal.executable private @extern_dispatch_ballot_vote {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6, [
        Shader, GroupNonUniform], []>,
        api=Vulkan, AMD:DiscreteGPU,
        #spirv.resource_limits<>>}>)
      objects([#hal.executable.object<{path = "/does/not/exist.spv"}>]) {
    hal.executable.export public @main ordinal(0) layout(
        #hal.pipeline.layout<push_constants = 1, sets = [
          <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer>
          ]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %c1 = arith.constant 1 : index 
      hal.return %c1, %c1, %c1 : index, index, index 
    }
  }
}

// Verify that we do not construct a condition region when the only bit set is
// the basic bit which is implied by the minimum supported SPIR-V version of 1.3
// CHECK-LABEL: hal.executable private @extern_dispatch_ballot_vote
//       CHECK-NOT:  hal.executable.condition
