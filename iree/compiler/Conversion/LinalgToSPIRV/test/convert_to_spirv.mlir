// RUN: iree-opt -split-input-file -iree-codegen-convert-to-spirv %s | IreeFileCheck %s

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  // CHECK-LABEL: spv.module
  // CHECK: spv.globalVariable @__push_constant_var__ : !spv.ptr<!spv.struct<(!spv.array<5 x i32, stride=4> [0])>, PushConstant>
  // CHECK: spv.func @push_constant()
  func @push_constant() {
    // CHECK: %[[INDEX_0:.+]] = spv.constant 0 : i32
    // CHECK: %[[INDEX_1:.+]] = spv.constant 2 : i32
    // CHECK: %[[ADDR:.+]] = spv.mlir.addressof @__push_constant_var__ : !spv.ptr<!spv.struct<(!spv.array<5 x i32, stride=4> [0])>, PushConstant>
    // CHECK: %[[AC:.+]] = spv.AccessChain %[[ADDR]][%[[INDEX_0]], %[[INDEX_1]]] : !spv.ptr<!spv.struct<(!spv.array<5 x i32, stride=4> [0])>, PushConstant>
    // CHECK: spv.Load "PushConstant" %[[AC]] : i32
    %0 = hal.interface.load.constant offset = 2 : index
    return
  }

  hal.interface @legacy_io attributes {push_constants = 5 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  // CHECK-LABEL: spv.module
  // CHECK: spv.globalVariable @[[RET:.+]] bind(3, 4) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK: spv.globalVariable @[[ARG0:.+]] bind(1, 2) {aliased} : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK: spv.globalVariable @[[ARG1:.+]] bind(1, 2) {aliased} : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK: spv.func @resource_bindings_in_same_entry_func()
  func @resource_bindings_in_same_entry_func() {
    %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4x4xf32>
    %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4x4xf32>
    %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4x4xf32>
    return
  }

  hal.interface @legacy_io attributes {push_constants = 5 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
  }
}

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  // CHECK-LABEL: spv.module
  // CHECK: spv.globalVariable @[[FUNC2_RET:.+]] bind(3, 4) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK: spv.globalVariable @[[FUNC2_ARG:.+]] bind(1, 2) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK: spv.globalVariable @[[FUNC1_RET:.+]] bind(3, 4) : !spv.ptr<!spv.struct<(!spv.array<4 x vector<4xf32>, stride=16> [0])>, StorageBuffer>
  // CHECK: spv.globalVariable @[[FUNC1_ARG:.+]] bind(1, 2) : !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4> [0])>, StorageBuffer>

  // CHECK: spv.func @resource_bindings_in_entry_func1()
  func @resource_bindings_in_entry_func1() {
    // CHECK: spv.mlir.addressof @[[FUNC1_ARG:.+]]
    // CHECK: spv.mlir.addressof @[[FUNC1_RET:.+]]
    %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4x4xf32>
    %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4xvector<4xf32>>
    return
  }

  // CHECK: spv.func @resource_bindings_in_entry_func2()
  func @resource_bindings_in_entry_func2() {
    // CHECK: spv.mlir.addressof @[[FUNC2_ARG]]
    // CHECK: spv.mlir.addressof @[[FUNC2_RET]]
    %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4x4xf32>
    %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4x4xf32>
    return
  }

  hal.interface @legacy_io attributes {push_constants = 5 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
  }
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, CooperativeMatrixNV, Int8, StorageBuffer8BitAccess], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix, SPV_KHR_8bit_storage]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  // CHECK-LABEL: spv.module
  // CHECK: spv.func @kernel_matmul
  func @kernel_matmul(%arg0: memref<8x32xi8>, %arg1: memref<32x8xi8>, %arg2: memref<8x8xi32>) attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
    %c0 = constant 0 : index
    %cst = constant 0 : i32
    %cst_i8 = constant 0 : i8
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst_i8 : memref<8x32xi8>, vector<8x32xi8>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst_i8 : memref<32x8xi8>, vector<32x8xi8>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst : memref<8x8xi32>, vector<8x8xi32>
    %3 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} %0, %1, %2 : vector<8x32xi8>, vector<32x8xi8> into vector<8x8xi32>
    vector.transfer_write %3, %arg2[%c0, %c0] : vector<8x8xi32>, memref<8x8xi32>
    // CHECK: %[[A:.+]] = spv.CooperativeMatrixLoadNV %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK: %[[B:.+]] = spv.CooperativeMatrixLoadNV %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK: %[[C:.+]] = spv.CooperativeMatrixLoadNV %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK: %[[R:.+]] = spv.CooperativeMatrixMulAddNV %[[A]], %[[B]], %[[C]]
    // CHECK: spv.CooperativeMatrixStoreNV %{{.*}}, %[[R]], %{{.*}}, %{{.*}}
    return
  }
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, CooperativeMatrixNV, Int8, Float16, StorageUniform16, StorageBuffer8BitAccess, Float16Buffer], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix, SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  // CHECK-LABEL: spv.module
  func @kernel_matmul_licm(%arg0: memref<4096x4096xi8>, %arg1: memref<4096x4096xi8>, %arg2: memref<4096x4096xi32>) attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
    %c32 = constant 32 : index
    %c4096 = constant 4096 : index
    %c0 = constant 0 : index
    %c0_i32 = constant 0 : i32
    %c0_i8 = constant 0 : i8
    // CHECK: %[[C:.+]] = spv.CooperativeMatrixLoadNV %{{.*}}, %{{.*}}, %{{.*}}
    %4 = vector.transfer_read %arg2[%c0, %c0], %c0_i32 {masked = [false, false]} : memref<4096x4096xi32>, vector<16x16xi32>
    // CHECK: %[[ACC:.+]] = spv.Variable : !spv.ptr<!spv.coopmatrix<16x16xi32, Subgroup>, Function>
    // CHECK: spv.loop {
      // CHECK: spv.Branch ^[[BB:.+]](%{{.*}}, %[[C]] : i32, !spv.coopmatrix<16x16xi32, Subgroup>)
      // CHECK: ^[[BB]](%{{.*}}: i32, %[[C1:.+]]: !spv.coopmatrix<16x16xi32, Subgroup>)
    %5 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %4) -> (vector<16x16xi32>) {
      // CHECK: %[[A:.+]] = spv.CooperativeMatrixLoadNV %{{.*}}, %{{.*}}, %{{.*}}
      %6 = vector.transfer_read %arg0[%c0, %arg3], %c0_i8 {masked = [false, false]} : memref<4096x4096xi8>, vector<16x32xi8>
      // CHECK: %[[B:.+]] = spv.CooperativeMatrixLoadNV %{{.*}}, %{{.*}}, %{{.*}}
      %7 = vector.transfer_read %arg1[%arg3, %c0], %c0_i8 {masked = [false, false]} : memref<4096x4096xi8>, vector<32x16xi8>
      // CHECK: %[[R:.+]] = spv.CooperativeMatrixMulAddNV %[[A]], %[[B]], %[[C1]]
      %8 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} %6, %7, %arg4 : vector<16x32xi8>, vector<32x16xi8> into vector<16x16xi32>
      // CHECK: spv.Store "Function" %[[ACC]], %[[R]] : !spv.coopmatrix<16x16xi32, Subgroup>
      // CHECK: spv.Branch ^[[BB]](%{{.*}}, %[[R]] : i32, !spv.coopmatrix<16x16xi32, Subgroup>)
      scf.yield %8 : vector<16x16xi32>
    }
    // CHECK: %[[ACCv:.+]] = spv.Load "Function" %[[ACC]] : !spv.coopmatrix<16x16xi32, Subgroup>
    // CHECK: spv.CooperativeMatrixStoreNV %{{.*}}, %[[ACCv]], %{{.*}}, %{{.*}}
    vector.transfer_write %5, %arg2[%c0, %c0] : vector<16x16xi32>, memref<4096x4096xi32>
    return
  }
}

// -----

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, CooperativeMatrixNV, Int8, Float16, StorageUniform16, StorageBuffer8BitAccess, Float16Buffer], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix, SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  // CHECK-LABEL: spv.module
  func @kernel_matmul_vector_memref(%arg0: memref<4096x256xvector<4xi32>>, %arg1: memref<4096x256xvector<4xi32>>, %arg2: memref<4096x1024xvector<4xi32>>) attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
    %c32 = constant 32 : index
    %c4096 = constant 4096 : index
    %c0 = constant 0 : index
    %cst = constant dense<0> : vector<4xi32>
    // CHECK: %[[C:.+]] = spv.CooperativeMatrixLoadNV %{{.*}}, %{{.*}}, %{{.*}}
    %4 = vector.transfer_read %arg2[%c0, %c0], %cst : memref<4096x1024xvector<4xi32>>, vector<16x16xi32>
    // CHECK: %[[ACC:.+]] = spv.Variable : !spv.ptr<!spv.coopmatrix<16x16xi32, Subgroup>, Function>
    // CHECK: spv.loop {
      // CHECK: spv.Branch ^[[BB:.+]](%{{.*}}, %[[C]] : i32, !spv.coopmatrix<16x16xi32, Subgroup>)
      // CHECK: ^[[BB]](%{{.*}}: i32, %[[C1:.+]]: !spv.coopmatrix<16x16xi32, Subgroup>)
    %5 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %4) -> (vector<16x16xi32>) {
      // CHECK: %[[A:.+]] = spv.CooperativeMatrixLoadNV %{{.*}}, %{{.*}}, %{{.*}}
      %6 = vector.transfer_read %arg0[%c0, %arg3], %cst : memref<4096x256xvector<4xi32>>, vector<16x32xi8>
      // CHECK: %[[B:.+]] = spv.CooperativeMatrixLoadNV %{{.*}}, %{{.*}}, %{{.*}}
      %7 = vector.transfer_read %arg1[%arg3, %c0], %cst : memref<4096x256xvector<4xi32>>, vector<32x16xi8>
      // CHECK: %[[R:.+]] = spv.CooperativeMatrixMulAddNV %[[A]], %[[B]], %[[C1]]
      %8 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} %6, %7, %arg4 : vector<16x32xi8>, vector<32x16xi8> into vector<16x16xi32>
      // CHECK: spv.Store "Function" %[[ACC]], %[[R]] : !spv.coopmatrix<16x16xi32, Subgroup>
      // CHECK: spv.Branch ^[[BB]](%{{.*}}, %[[R]] : i32, !spv.coopmatrix<16x16xi32, Subgroup>)
      scf.yield %8 : vector<16x16xi32>
    }
    // CHECK: %[[ACCv:.+]] = spv.Load "Function" %[[ACC]] : !spv.coopmatrix<16x16xi32, Subgroup>
    // CHECK: spv.CooperativeMatrixStoreNV %{{.*}}, %[[ACCv]], %{{.*}}, %{{.*}}
    vector.transfer_write %5, %arg2[%c0, %c0] : vector<16x16xi32>, memref<4096x1024xvector<4xi32>>
    return
  }
}

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, SwiftShader:CPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 16384 : i32, max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>, subgroup_size = 4 : i32}>}  {
  func @interface_binding() {
    %c0 = constant 0 : index
    %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<8x5xf32>
    %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<5xf32>
    %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<8x5xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}
// CHECK-LABEL: spv.module
//   CHECK-DAG:   spv.globalVariable @[[RET0:.+]] bind(0, 2)
//   CHECK-DAG:   spv.globalVariable @[[ARG1:.+]] bind(0, 1)
//   CHECK-DAG:   spv.globalVariable @[[ARG0:.+]] bind(0, 0)
//       CHECK:   spv.func
//   CHECK-DAG:   %{{.+}} = spv.mlir.addressof @[[RET0]]
//   CHECK-DAG:   %{{.+}} = spv.mlir.addressof @[[ARG0]]
//   CHECK-DAG:   %{{.+}} = spv.mlir.addressof @[[ARG1]]

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, SwiftShader:CPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 16384 : i32, max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>, subgroup_size = 4 : i32}>}  {
  func @interface_wg_id() {
    %0 = hal.interface.workgroup.id[0] : index
    %1 = hal.interface.workgroup.id[1] : index
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}
// CHECK-LABEL: spv.module
//   CHECK-DAG:   spv.globalVariable @[[WGID:.+]] built_in("WorkgroupId")
//       CHECK:   spv.func
//       CHECK:     %[[ADDR1:.+]] = spv.mlir.addressof @[[WGID]]
//       CHECK:     %[[VAL1:.+]] = spv.Load "Input" %[[ADDR1]]
//       CHECK:     %[[WGIDX:.+]] = spv.CompositeExtract %[[VAL1]][0 : i32]
//       CHECK:     %[[ADDR2:.+]] = spv.mlir.addressof @[[WGID]]
//       CHECK:     %[[VAL2:.+]] = spv.Load "Input" %[[ADDR2]]
//       CHECK:     %[[WGIDY:.+]] = spv.CompositeExtract %[[VAL2]][1 : i32]

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, SwiftShader:CPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 16384 : i32, max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>, subgroup_size = 4 : i32}>}  {
  func @interface_wg_count() {
    %0 = hal.interface.workgroup.count[0] : index
    %1 = hal.interface.workgroup.count[1] : index
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}
// CHECK-LABEL: spv.module
//   CHECK-DAG:   spv.globalVariable @[[WGCOUNT:.+]] built_in("NumWorkgroups")
//       CHECK:   spv.func
//       CHECK:     %[[ADDR1:.+]] = spv.mlir.addressof @[[WGCOUNT]]
//       CHECK:     %[[VAL1:.+]] = spv.Load "Input" %[[ADDR1]]
//       CHECK:     %[[WGIDX:.+]] = spv.CompositeExtract %[[VAL1]][0 : i32]
//       CHECK:     %[[ADDR2:.+]] = spv.mlir.addressof @[[WGCOUNT]]
//       CHECK:     %[[VAL2:.+]] = spv.Load "Input" %[[ADDR2]]
//       CHECK:     %[[WGIDY:.+]] = spv.CompositeExtract %[[VAL2]][1 : i32]
