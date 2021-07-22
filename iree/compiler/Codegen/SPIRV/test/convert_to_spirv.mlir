// RUN: iree-opt -split-input-file -iree-convert-to-spirv %s | IreeFileCheck %s

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>} {
  // CHECK-LABEL: spv.module
  // CHECK: spv.GlobalVariable @__push_constant_var__ : !spv.ptr<!spv.struct<(!spv.array<5 x i32, stride=4> [0])>, PushConstant>
  // CHECK: spv.func @push_constant()
  func @push_constant() {
    // CHECK: %[[INDEX_0:.+]] = spv.Constant 0 : i32
    // CHECK: %[[INDEX_1:.+]] = spv.Constant 2 : i32
    // CHECK: %[[ADDR:.+]] = spv.mlir.addressof @__push_constant_var__ : !spv.ptr<!spv.struct<(!spv.array<5 x i32, stride=4> [0])>, PushConstant>
    // CHECK: %[[AC:.+]] = spv.AccessChain %[[ADDR]][%[[INDEX_0]], %[[INDEX_1]]] : !spv.ptr<!spv.struct<(!spv.array<5 x i32, stride=4> [0])>, PushConstant>
    // CHECK: spv.Load "PushConstant" %[[AC]] : i32
    %0 = hal.interface.load.constant offset = 2 : index
    return
  }

  hal.interface @io attributes {push_constants = 5 : index, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>} {
  // CHECK-LABEL: spv.module
  // CHECK: spv.GlobalVariable @__resource_var_3_4_ bind(3, 4) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK: spv.GlobalVariable @__resource_var_1_2__0 bind(1, 2) {aliased} : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK: spv.GlobalVariable @__resource_var_1_2_ bind(1, 2) {aliased} : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK: spv.func @resource_bindings_in_same_entry_func()
  func @resource_bindings_in_same_entry_func() {
    %c0 = constant 0 : index
    %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4x4xf32>
    %1 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4x4xf32>
    %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4x4xf32>
    return
  }

  hal.interface @io attributes {push_constants = 5 : index, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
  }
}

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>} {
  // CHECK-LABEL: spv.module
  // CHECK: spv.GlobalVariable @[[FUNC2_RET:.+]] bind(3, 4) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK: spv.GlobalVariable @[[FUNC2_ARG:.+]] bind(1, 2) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK: spv.GlobalVariable @[[FUNC1_RET:.+]] bind(3, 4) : !spv.ptr<!spv.struct<(!spv.array<4 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  // CHECK: spv.GlobalVariable @[[FUNC1_ARG:.+]] bind(1, 2) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>

  // CHECK: spv.func @resource_bindings_in_entry_func1()
  func @resource_bindings_in_entry_func1() {
    // CHECK: spv.mlir.addressof @[[FUNC1_ARG:.+]]
    // CHECK: spv.mlir.addressof @[[FUNC1_RET:.+]]
    %c0 = constant 0 : index
    %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4x4xf32>
    %1 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4xvector<4xf32>>
    return
  }

  // CHECK: spv.func @resource_bindings_in_entry_func2()
  func @resource_bindings_in_entry_func2() {
    // CHECK: spv.mlir.addressof @[[FUNC2_ARG]]
    // CHECK: spv.mlir.addressof @[[FUNC2_RET]]
    %c0 = constant 0 : index
    %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4x4xf32>
    %1 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4x4xf32>
    return
  }

  hal.interface @io attributes {push_constants = 5 : index, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
  }
}

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, SwiftShader:CPU, {}>}  {
  func @interface_binding() {
    %c0 = constant 0 : index
    %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<8x5xf32>
    %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<5xf32>
    %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<8x5xf32>
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}
// CHECK-LABEL: spv.module
//   CHECK-DAG:   spv.GlobalVariable @[[RET0:.+]] bind(0, 2)
//   CHECK-DAG:   spv.GlobalVariable @[[ARG1:.+]] bind(0, 1)
//   CHECK-DAG:   spv.GlobalVariable @[[ARG0:.+]] bind(0, 0)
//       CHECK:   spv.func
//   CHECK-DAG:   %{{.+}} = spv.mlir.addressof @[[RET0]]
//   CHECK-DAG:   %{{.+}} = spv.mlir.addressof @[[ARG0]]
//   CHECK-DAG:   %{{.+}} = spv.mlir.addressof @[[ARG1]]

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, SwiftShader:CPU, {}>}  {
  func @interface_wg_id() {
    %0 = hal.interface.workgroup.id[0] : index
    %1 = hal.interface.workgroup.id[1] : index
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
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

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, SwiftShader:CPU, {}>}  {
  func @interface_wg_count() {
    %0 = hal.interface.workgroup.count[0] : index
    %1 = hal.interface.workgroup.count[1] : index
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
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
