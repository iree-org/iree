// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(builtin.module(iree-convert-to-spirv)))' %s | IreeFileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 5, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @push_constant {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>}> {
    hal.executable.entry_point @push_constant layout(#executable_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      // CHECK-LABEL: spv.module
      // CHECK: spv.GlobalVariable @__push_constant_var__ : !spv.ptr<!spv.struct<(!spv.array<5 x i32, stride=4> [0])>, PushConstant>
      // CHECK: spv.func @push_constant()
      func @push_constant() {
        // CHECK-DAG: %[[INDEX_0:.+]] = spv.Constant 0 : i32
        // CHECK-DAG: %[[INDEX_1:.+]] = spv.Constant 2 : i32
        // CHECK: %[[ADDR:.+]] = spv.mlir.addressof @__push_constant_var__ : !spv.ptr<!spv.struct<(!spv.array<5 x i32, stride=4> [0])>, PushConstant>
        // CHECK: %[[AC:.+]] = spv.AccessChain %[[ADDR]][%[[INDEX_0]], %[[INDEX_1]]] : !spv.ptr<!spv.struct<(!spv.array<5 x i32, stride=4> [0])>, PushConstant>
        // CHECK: spv.Load "PushConstant" %[[AC]] : i32
        %0 = hal.interface.constant.load[2] : index
        return
      }
    }
  }
}

// -----

#executable_layout = #hal.executable.layout<push_constants = 5, sets = [
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<3, bindings = [
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable private @resource_bindings_in_same_func {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>}> {
    hal.executable.entry_point @resource_bindings_in_same_func layout(#executable_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      // CHECK-LABEL: spv.module
      // CHECK: spv.GlobalVariable @[[ARG0:.+]] bind(1, 2) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
      // CHECK: spv.GlobalVariable @[[ARG1_0:.+]] bind(1, 3) {aliased} : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
      // CHECK: spv.GlobalVariable @[[ARG1_1:.+]] bind(1, 3) {aliased} : !spv.ptr<!spv.struct<(!spv.array<4 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
      // CHECK: spv.GlobalVariable @[[RET0:.+]] bind(3, 4) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
      // CHECK: spv.func @resource_bindings_in_same_entry_func()
      func @resource_bindings_in_same_entry_func() {
        %c0 = arith.constant 0 : index

        // Same type
        // CHECK: spv.mlir.addressof @[[ARG0]]
        // CHECK: spv.mlir.addressof @[[ARG0]]
        %0 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<4x4xf32>
        %1 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<4x4xf32>

        // Different type
        // CHECK: spv.mlir.addressof @[[ARG1_0]]
        // CHECK: spv.mlir.addressof @[[ARG1_1]]
        %2 = hal.interface.binding.subspan set(1) binding(3) type(storage_buffer) : memref<4x4xf32>
        %3 = hal.interface.binding.subspan set(1) binding(3) type(storage_buffer) : memref<4xvector<4xf32>>

        // CHECK: spv.mlir.addressof @[[RET0]]
        %4 = hal.interface.binding.subspan set(3) binding(4) type(storage_buffer) : memref<4x4xf32>

        %5 = memref.load %0[%c0, %c0] : memref<4x4xf32>
        %6 = memref.load %1[%c0, %c0] : memref<4x4xf32>

        %7 = memref.load %2[%c0, %c0] : memref<4x4xf32>
        %8 = memref.load %3[%c0] : memref<4xvector<4xf32>>

        %9 = memref.load %4[%c0, %c0] : memref<4x4xf32>

        return
      }
    }
  }
}

// -----

#executable_layout = #hal.executable.layout<push_constants = 5, sets = [
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<3, bindings = [
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable private @resource_bindings_in_multi_entry_func {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>}> {
    hal.executable.entry_point @resource_bindings_in_entry_func1 layout(#executable_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    hal.executable.entry_point @resource_bindings_in_entry_func2 layout(#executable_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      // CHECK-LABEL: spv.module
      // CHECK: spv.GlobalVariable @[[FUNC1_ARG:.+]] bind(1, 2) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
      // CHECK: spv.GlobalVariable @[[FUNC1_RET:.+]] bind(3, 4) : !spv.ptr<!spv.struct<(!spv.array<4 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
      // CHECK: spv.GlobalVariable @[[FUNC2_ARG:.+]] bind(1, 2) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
      // CHECK: spv.GlobalVariable @[[FUNC2_RET:.+]] bind(3, 4) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>

      // CHECK: spv.func @resource_bindings_in_entry_func1()
      func @resource_bindings_in_entry_func1() {
        // CHECK: spv.mlir.addressof @[[FUNC1_ARG]]
        // CHECK: spv.mlir.addressof @[[FUNC1_RET]]
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<4x4xf32>
        %1 = hal.interface.binding.subspan set(3) binding(4) type(storage_buffer) : memref<4xvector<4xf32>>

        %2 = memref.load %0[%c0, %c0] : memref<4x4xf32>
        %3 = memref.load %1[%c0] : memref<4xvector<4xf32>>

        return
      }

      // CHECK: spv.func @resource_bindings_in_entry_func2()
      func @resource_bindings_in_entry_func2() {
        // CHECK: spv.mlir.addressof @[[FUNC2_ARG]]
        // CHECK: spv.mlir.addressof @[[FUNC2_RET]]
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<4x4xf32> // Same type as previous function
        %1 = hal.interface.binding.subspan set(3) binding(4) type(storage_buffer) : memref<4x4xf32> // Different type as previous function

        %2 = memref.load %0[%c0, %c0] : memref<4x4xf32>
        %3 = memref.load %1[%c0, %c0] : memref<4x4xf32>

        return
      }
    }
  }
}

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @interface_binding {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>}> {
    hal.executable.entry_point @interface_binding layout(#executable_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func @interface_binding() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<8x5xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<5xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<8x5xf32>

        %3 = memref.load %0[%c0, %c0] : memref<8x5xf32>
        %4 = memref.load %1[%c0] : memref<5xf32>
        %5 = memref.load %2[%c0, %c0] : memref<8x5xf32>

        return
      }
    }
  }
}

// Explicitly check the variable symbols

// CHECK-LABEL: spv.module
//       CHECK:   spv.GlobalVariable @__resource_var_0_0_ bind(0, 0)
//       CHECK:   spv.GlobalVariable @__resource_var_0_1_ bind(0, 1)
//       CHECK:   spv.GlobalVariable @__resource_var_0_2_ bind(0, 2)
//       CHECK:   spv.func
//       CHECK:   %{{.+}} = spv.mlir.addressof @__resource_var_0_0_
//       CHECK:   %{{.+}} = spv.mlir.addressof @__resource_var_0_1_
//       CHECK:   %{{.+}} = spv.mlir.addressof @__resource_var_0_2_

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @interface_wg_id {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>}> {
    hal.executable.entry_point @interface_wg_id layout(#executable_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func @interface_wg_id() {
        %0 = hal.interface.workgroup.id[0] : index
        %1 = hal.interface.workgroup.id[1] : index
        return
      }
    }
  }
}

// CHECK-LABEL: spv.module
//   CHECK-DAG:   spv.GlobalVariable @[[WGID:.+]] built_in("WorkgroupId")
//       CHECK:   spv.func
//       CHECK:     %[[ADDR1:.+]] = spv.mlir.addressof @[[WGID]]
//       CHECK:     %[[VAL1:.+]] = spv.Load "Input" %[[ADDR1]]
//       CHECK:     %[[WGIDX:.+]] = spv.CompositeExtract %[[VAL1]][0 : i32]
//       CHECK:     %[[ADDR2:.+]] = spv.mlir.addressof @[[WGID]]
//       CHECK:     %[[VAL2:.+]] = spv.Load "Input" %[[ADDR2]]
//       CHECK:     %[[WGIDY:.+]] = spv.CompositeExtract %[[VAL2]][1 : i32]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @interface_wg_count {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>}> {
    hal.executable.entry_point @interface_wg_count layout(#executable_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func @interface_wg_count() {
        %0 = hal.interface.workgroup.count[0] : index
        %1 = hal.interface.workgroup.count[1] : index
        return
      }
    }
  }
}
// CHECK-LABEL: spv.module
//   CHECK-DAG:   spv.GlobalVariable @[[WGCOUNT:.+]] built_in("NumWorkgroups")
//       CHECK:   spv.func
//       CHECK:     %[[ADDR1:.+]] = spv.mlir.addressof @[[WGCOUNT]]
//       CHECK:     %[[VAL1:.+]] = spv.Load "Input" %[[ADDR1]]
//       CHECK:     %[[WGIDX:.+]] = spv.CompositeExtract %[[VAL1]][0 : i32]
//       CHECK:     %[[ADDR2:.+]] = spv.mlir.addressof @[[WGCOUNT]]
//       CHECK:     %[[VAL2:.+]] = spv.Load "Input" %[[ADDR2]]
//       CHECK:     %[[WGIDY:.+]] = spv.CompositeExtract %[[VAL2]][1 : i32]
