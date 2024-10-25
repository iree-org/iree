// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{vectorize-padding=true}))" --split-input-file %s | FileCheck %s

func.func @test_subbyte_6_i1() attributes {translation_info = #iree_codegen.translation_info<CPUDoubleTilingExpert>} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<8xi1>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<8xi1>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<8xi1>>
  %3 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [6], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<8xi1>> -> tensor<6xi1>
  %4 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<8xi1>> -> tensor<6xi1>
  %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<8xi1>> -> tensor<6xi1>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<6xi1>, tensor<6xi1>) outs(%3 : tensor<6xi1>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[6], [8], [0], [0]]>} {
  ^bb0(%in: i1, %in_0: i1, %out: i1):
    %7 = arith.addi %in, %in_0 : i1
    linalg.yield %7 : i1
  } -> tensor<6xi1>
  flow.dispatch.tensor.store %6, %2, offsets = [0], sizes = [6], strides = [1] : tensor<6xi1> -> !flow.dispatch.tensor<writeonly:tensor<8xi1>>
  return
}

// CHECK-LABEL: @test_subbyte_6_i1

// CHECK: %[[TR1:.+]] = vector.transfer_read %[[.+]][%c0], %false : tensor<6xi1>, vector<8xi1>
// CHECK: %[[ESS1:.+]] = vector.extract_strided_slice %[[TR1]] {offsets = [0], sizes = [6], strides = [1]} : vector<8xi1> to vector<6xi1>

// CHECK: %[[TR2:.+]] = vector.transfer_read %[[.+]][%c0], %false : tensor<6xi1>, vector<8xi1>
// CHECK: %[[ESS2:.+]] = vector.extract_strided_slice %[[TR2]] {offsets = [0], sizes = [6], strides = [1]} : vector<8xi1> to vector<6xi1>

// CHECK: %[[ISS:.+]] = vector.insert_strided_slice
// CHECK-SAME: {offsets = [0], strides = [1]} : vector<6xi1> into vector<8xi1>
// CHECK: vector.transfer_write %[[ISS]], %[[.+]][%c0] {in_bounds = [true]} : vector<8xi1>, tensor<6xi1>

