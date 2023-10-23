// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-convert-to-spirv))))' %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-convert-to-spirv{index-bits=64}))))' %s | FileCheck %s --check-prefix=INDEX64

#pipeline_layout = #hal.pipeline.layout<push_constants = 5, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @push_constant {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Int64, Shader], []>, #spirv.resource_limits<>>}>) {
    hal.executable.export @push_constant layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      // CHECK-LABEL: spirv.module
      // CHECK: spirv.GlobalVariable @__push_constant_var__ : !spirv.ptr<!spirv.struct<(!spirv.array<5 x i32, stride=4> [0])>, PushConstant>
      // CHECK: spirv.func @push_constant()
      func.func @push_constant() -> index {
        // CHECK-DAG: %[[INDEX_0:.+]] = spirv.Constant 0 : i32
        // CHECK-DAG: %[[INDEX_1:.+]] = spirv.Constant 2 : i32
        // CHECK: %[[ADDR:.+]] = spirv.mlir.addressof @__push_constant_var__ : !spirv.ptr<!spirv.struct<(!spirv.array<5 x i32, stride=4> [0])>, PushConstant>
        // CHECK: %[[AC:.+]] = spirv.AccessChain %[[ADDR]][%[[INDEX_0]], %[[INDEX_1]]] : !spirv.ptr<!spirv.struct<(!spirv.array<5 x i32, stride=4> [0])>, PushConstant>
        // CHECK: spirv.Load "PushConstant" %[[AC]] : i32
        // INDEX64-DAG: %[[INDEX_0:.+]] = spirv.Constant 0 : i32
        // INDEX64-DAG: %[[INDEX_1:.+]] = spirv.Constant 2 : i32
        // INDEX64: %[[ADDR:.+]] = spirv.mlir.addressof @__push_constant_var__ : !spirv.ptr<!spirv.struct<(!spirv.array<5 x i32, stride=4> [0])>, PushConstant>
        // INDEX64: %[[AC:.+]] = spirv.AccessChain %[[ADDR]][%[[INDEX_0]], %[[INDEX_1]]] : !spirv.ptr<!spirv.struct<(!spirv.array<5 x i32, stride=4> [0])>, PushConstant>
        // INDEX64: %[[LOAD:.+]] = spirv.Load "PushConstant" %[[AC]] : i32
        // INDEX64: spirv.UConvert %[[LOAD]] : i32 to i64
        %0 = hal.interface.constant.load[2] : i32
        %1 = arith.index_castui %0 : i32 to index
        return %1 : index
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 5, sets = [
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<3, bindings = [
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable private @resource_bindings_in_same_func {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Int64, Shader], []>, #spirv.resource_limits<>>}>) {
    hal.executable.export @resource_bindings_in_same_func layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      // CHECK-LABEL: spirv.module
      // CHECK: spirv.GlobalVariable @[[ARG0:.+]] bind(1, 2) : !spirv.ptr<!spirv.struct<(!spirv.array<16 x f32, stride=4> [0])>, StorageBuffer>
      // CHECK: spirv.GlobalVariable @[[ARG1_0:.+]] bind(1, 3) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.array<16 x f32, stride=4> [0])>, StorageBuffer>
      // CHECK: spirv.GlobalVariable @[[ARG1_1:.+]] bind(1, 3) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.array<4 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
      // CHECK: spirv.GlobalVariable @[[RET0:.+]] bind(3, 4) : !spirv.ptr<!spirv.struct<(!spirv.array<16 x f32, stride=4> [0])>, StorageBuffer>
      // CHECK: spirv.func @resource_bindings_in_same_entry_func()
      func.func @resource_bindings_in_same_entry_func() -> f32 {
        %c0 = arith.constant 0 : index

        // Same type
        // CHECK: spirv.mlir.addressof @[[ARG0]]
        // CHECK: spirv.mlir.addressof @[[ARG0]]
        %0 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>
        %1 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>

        // Different type
        // CHECK: spirv.mlir.addressof @[[ARG1_0]]
        // CHECK: spirv.mlir.addressof @[[ARG1_1]]
        %2 = hal.interface.binding.subspan set(1) binding(3) type(storage_buffer) : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>
        %3 = hal.interface.binding.subspan set(1) binding(3) type(storage_buffer) : memref<4xvector<4xf32>, #spirv.storage_class<StorageBuffer>>

        // CHECK: spirv.mlir.addressof @[[RET0]]
        %4 = hal.interface.binding.subspan set(3) binding(4) type(storage_buffer) : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>

        %5 = memref.load %0[%c0, %c0] : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>
        %6 = memref.load %1[%c0, %c0] : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>

        %7 = memref.load %2[%c0, %c0] : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>
        %8 = memref.load %3[%c0] : memref<4xvector<4xf32>, #spirv.storage_class<StorageBuffer>>

        %9 = memref.load %4[%c0, %c0] : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>

        %10 = arith.addf %5, %6 : f32
        %11 = arith.addf %7, %9 : f32
        %12 = arith.addf %10, %11 : f32
        %13 = vector.extractelement %8[%c0 : index] : vector<4xf32>
        %14 = arith.addf %12, %13 : f32

        return %14 : f32
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 5, sets = [
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<3, bindings = [
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable private @resource_bindings_in_multi_entry_func {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Int64, Shader], []>, #spirv.resource_limits<>>}>) {
    hal.executable.export @resource_bindings_in_entry_func1 layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    hal.executable.export @resource_bindings_in_entry_func2 layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      // CHECK-LABEL: spirv.module
      // CHECK: spirv.GlobalVariable @[[FUNC1_ARG:.+]] bind(1, 2) : !spirv.ptr<!spirv.struct<(!spirv.array<16 x f32, stride=4> [0])>, StorageBuffer>
      // CHECK: spirv.GlobalVariable @[[FUNC1_RET:.+]] bind(3, 4) : !spirv.ptr<!spirv.struct<(!spirv.array<4 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
      // CHECK: spirv.GlobalVariable @[[FUNC2_ARG:.+]] bind(1, 2) : !spirv.ptr<!spirv.struct<(!spirv.array<16 x f32, stride=4> [0])>, StorageBuffer>
      // CHECK: spirv.GlobalVariable @[[FUNC2_RET:.+]] bind(3, 4) : !spirv.ptr<!spirv.struct<(!spirv.array<16 x f32, stride=4> [0])>, StorageBuffer>

      // CHECK: spirv.func @resource_bindings_in_entry_func1()
      func.func @resource_bindings_in_entry_func1() -> f32 {
        // CHECK: spirv.mlir.addressof @[[FUNC1_ARG]]
        // CHECK: spirv.mlir.addressof @[[FUNC1_RET]]
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>
        %1 = hal.interface.binding.subspan set(3) binding(4) type(storage_buffer) : memref<4xvector<4xf32>, #spirv.storage_class<StorageBuffer>>

        %2 = memref.load %0[%c0, %c0] : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>
        %3 = memref.load %1[%c0] : memref<4xvector<4xf32>, #spirv.storage_class<StorageBuffer>>

        %4 = vector.extractelement %3[%c0 : index] : vector<4xf32>
        %5 = arith.addf %2, %4 : f32

        return %5 : f32
      }

      // CHECK: spirv.func @resource_bindings_in_entry_func2()
      func.func @resource_bindings_in_entry_func2() -> f32 {
        // CHECK: spirv.mlir.addressof @[[FUNC2_ARG]]
        // CHECK: spirv.mlir.addressof @[[FUNC2_RET]]
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<4x4xf32, #spirv.storage_class<StorageBuffer>> // Same type as previous function
        %1 = hal.interface.binding.subspan set(3) binding(4) type(storage_buffer) : memref<4x4xf32, #spirv.storage_class<StorageBuffer>> // Different type as previous function

        %2 = memref.load %0[%c0, %c0] : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>
        %3 = memref.load %1[%c0, %c0] : memref<4x4xf32, #spirv.storage_class<StorageBuffer>>

        %4 = arith.addf %2, %3 : f32

        return %4 : f32
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @interface_binding {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Int64, Shader], []>, #spirv.resource_limits<>>}>) {
    hal.executable.export @interface_binding layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func.func @interface_binding() -> f32 {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<8x5xf32, #spirv.storage_class<StorageBuffer>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<5xf32, #spirv.storage_class<StorageBuffer>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<8x5xf32, #spirv.storage_class<StorageBuffer>>

        %3 = memref.load %0[%c0, %c0] : memref<8x5xf32, #spirv.storage_class<StorageBuffer>>
        %4 = memref.load %1[%c0] : memref<5xf32, #spirv.storage_class<StorageBuffer>>
        %5 = memref.load %2[%c0, %c0] : memref<8x5xf32, #spirv.storage_class<StorageBuffer>>

        %6 = arith.addf %3, %4 : f32
        %8 = arith.addf %6, %5 : f32

        return %8 : f32
      }
    }
  }
}

// Explicitly check the variable symbols

// CHECK-LABEL: spirv.module
//       CHECK:   spirv.GlobalVariable @__resource_var_0_0_ bind(0, 0)
//       CHECK:   spirv.GlobalVariable @__resource_var_0_1_ bind(0, 1)
//       CHECK:   spirv.GlobalVariable @__resource_var_0_2_ bind(0, 2)
//       CHECK:   spirv.func
//       CHECK:   %{{.+}} = spirv.mlir.addressof @__resource_var_0_0_
//       CHECK:   %{{.+}} = spirv.mlir.addressof @__resource_var_0_1_
//       CHECK:   %{{.+}} = spirv.mlir.addressof @__resource_var_0_2_

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @interface_wg_id {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Int64, Shader], []>, #spirv.resource_limits<>>}>) {
    hal.executable.export @interface_wg_id layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func.func @interface_wg_id() -> index {
        %0 = hal.interface.workgroup.id[0] : index
        %1 = hal.interface.workgroup.id[1] : index
        %2 = arith.addi %0, %1 : index
        return %2 : index
      }
    }
  }
}

// CHECK-LABEL: spirv.module
//   CHECK-DAG:   spirv.GlobalVariable @[[WGID:.+]] built_in("WorkgroupId")
//       CHECK:   spirv.func
//       CHECK:     %[[ADDR1:.+]] = spirv.mlir.addressof @[[WGID]]
//       CHECK:     %[[VAL1:.+]] = spirv.Load "Input" %[[ADDR1]]
//       CHECK:     %[[WGIDX:.+]] = spirv.CompositeExtract %[[VAL1]][0 : i32]
//       CHECK:     %[[ADDR2:.+]] = spirv.mlir.addressof @[[WGID]]
//       CHECK:     %[[VAL2:.+]] = spirv.Load "Input" %[[ADDR2]]
//       CHECK:     %[[WGIDY:.+]] = spirv.CompositeExtract %[[VAL2]][1 : i32]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @interface_wg_count {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Int64, Shader], []>, #spirv.resource_limits<>>}>) {
    hal.executable.export @interface_wg_count layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func.func @interface_wg_count() -> index {
        %0 = hal.interface.workgroup.count[0] : index
        %1 = hal.interface.workgroup.count[1] : index
        %2 = arith.addi %0, %1 : index
        return %2 : index
      }
    }
  }
}
// CHECK-LABEL: spirv.module
//   CHECK-DAG:   spirv.GlobalVariable @[[WGCOUNT:.+]] built_in("NumWorkgroups")
//       CHECK:   spirv.func
//       CHECK:     %[[ADDR1:.+]] = spirv.mlir.addressof @[[WGCOUNT]]
//       CHECK:     %[[VAL1:.+]] = spirv.Load "Input" %[[ADDR1]]
//       CHECK:     %[[WGIDX:.+]] = spirv.CompositeExtract %[[VAL1]][0 : i32]
//       CHECK:     %[[ADDR2:.+]] = spirv.mlir.addressof @[[WGCOUNT]]
//       CHECK:     %[[VAL2:.+]] = spirv.Load "Input" %[[ADDR2]]
//       CHECK:     %[[WGIDY:.+]] = spirv.CompositeExtract %[[VAL2]][1 : i32]
//   INDEX64-DAG:   spirv.GlobalVariable @[[WGCOUNT:.+]] built_in("NumWorkgroups")
//       INDEX64:   spirv.func
//       INDEX64:     %[[ADDR1:.+]] = spirv.mlir.addressof @[[WGCOUNT]]
//       INDEX64:     %[[VAL1:.+]] = spirv.Load "Input" %[[ADDR1]]
//       INDEX64:     %[[WGIDX:.+]] = spirv.CompositeExtract %[[VAL1]][0 : i32]
//       INDEX64:     %[[WGXEXT:.+]] = spirv.UConvert %[[WGIDX]] : i32 to i64
//       INDEX64:     %[[ADDR2:.+]] = spirv.mlir.addressof @[[WGCOUNT]]
//       INDEX64:     %[[VAL2:.+]] = spirv.Load "Input" %[[ADDR2]]
//       INDEX64:     %[[WGIDY:.+]] = spirv.CompositeExtract %[[VAL2]][1 : i32]
//       INDEX64:     %[[WGYEXT:.+]] = spirv.UConvert %[[WGIDY]] : i32 to i64
