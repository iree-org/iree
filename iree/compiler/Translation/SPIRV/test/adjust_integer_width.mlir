// RUN: iree-opt -iree-spirv-adjust-integer-width -verify-diagnostics -o - %s | IreeFileCheck %s

module{
  spv.module "Logical" "GLSL450" {
    spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    // CHECK: spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    // CHECK: spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
    func @foo_i64(%arg0 : i64, %arg1 : i64) -> () {
      // CHECK: spv._address_of {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.AccessChain {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.Load "StorageBuffer" %{{.*}} : i32
      // CHECK: spv._address_of {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.AccessChain {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.Store "StorageBuffer" %{{.*}} %{{.*}} : i32
      %0 = spv._address_of @constant_arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      %4 = spv._address_of @constant_arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %5 = spv.constant 0 : i32
      %6 = spv.AccessChain %4[%5] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %3 : i64
      spv.Return
    }
  }

  spv.module "Logical" "GLSL450" {
    spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    // CHECK: spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
    // CHECK: spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
    func @foo_i1(%arg0 : i1, %arg1 : i1) -> () {
      // CHECK: spv._address_of {{.*}} : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
      // CHECK: spv.AccessChain {{.*}} : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
      // CHECK: spv.Load "StorageBuffer" %{{.*}} : i8
      // CHECK-NEXT: spv.INotEqual {{.*}} : i8
      // CHECK: spv._address_of {{.*}} : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
      // CHECK: spv.AccessChain {{.*}} : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
      // CHECK: spv.Select {{.*}} : i1, i8
      // CHECK: spv.Store "StorageBuffer" {{.*}} : i8
      %0 = spv._address_of @constant_arg_0 : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i1
      %4 = spv._address_of @constant_arg_1 : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
      %5 = spv.constant 0 : i32
      %6 = spv.AccessChain %4[%5] : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %3 : i1
      spv.Return
    }
  }
}
