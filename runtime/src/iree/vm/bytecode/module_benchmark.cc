// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/benchmark.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_benchmark_module_c.h"

namespace {

struct native_import_module_s;
struct native_import_module_state_s;
typedef struct native_import_module_t native_import_module_t;
typedef struct native_import_module_state_t native_import_module_state_t;

// vm.import private @native_import_module.add_1(%arg0 : i32) -> i32
static iree_status_t native_import_module_add_1(
    iree_vm_stack_t* stack, iree_vm_native_function_flags_t flags,
    iree_byte_span_t args_storage, iree_byte_span_t rets_storage,
    iree_vm_native_function_target_t target_fn, void* module,
    void* module_state) {
  // Add 1 to arg0 and return.
  int32_t arg0 = *reinterpret_cast<int32_t*>(args_storage.data);
  int32_t ret0 = arg0 + 1;
  *reinterpret_cast<int32_t*>(rets_storage.data) = ret0;
  return iree_ok_status();
}

static const iree_vm_native_export_descriptor_t
    native_import_module_exports_[] = {
        {iree_make_cstring_view("add_1"), iree_make_cstring_view("0i_i"), 0,
         NULL},
};
static const iree_vm_native_function_ptr_t native_import_module_funcs_[] = {
    {(iree_vm_native_function_shim_t)native_import_module_add_1, NULL},
};
static_assert(IREE_ARRAYSIZE(native_import_module_funcs_) ==
                  IREE_ARRAYSIZE(native_import_module_exports_),
              "function pointer table must be 1:1 with exports");
static const iree_vm_native_module_descriptor_t
    native_import_module_descriptor_ = {
        /*.name=*/iree_make_cstring_view("native_import_module"),
        /*.version=*/0u,
        /*.attr_count=*/0,
        /*.attrs=*/NULL,
        /*.dependency_count=*/0,
        /*.dependencies=*/NULL,
        /*.import_count=*/0,
        /*.imports=*/NULL,
        /*.export_count=*/IREE_ARRAYSIZE(native_import_module_exports_),
        /*.exports=*/native_import_module_exports_,
        /*.import_count=*/IREE_ARRAYSIZE(native_import_module_funcs_),
        /*.imports=*/native_import_module_funcs_,
};

static iree_status_t native_import_module_create(
    iree_vm_instance_t* instance, iree_allocator_t allocator,
    iree_vm_module_t** out_module) {
  iree_vm_module_t interface;
  IREE_RETURN_IF_ERROR(iree_vm_module_initialize(&interface, NULL));
  return iree_vm_native_module_create(&interface,
                                      &native_import_module_descriptor_,
                                      instance, allocator, out_module);
}

// Benchmarks the given exported function, optionally passing in arguments.
static iree_status_t RunFunction(iree_benchmark_state_t* benchmark_state,
                                 iree_string_view_t function_name,
                                 std::vector<int32_t> i32_args,
                                 int result_count, int64_t batch_size = 1) {
  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        iree_allocator_system(), &instance));

  iree_vm_module_t* import_module = NULL;
  IREE_CHECK_OK(native_import_module_create(instance, iree_allocator_system(),
                                            &import_module));

  const auto* module_file_toc =
      iree_vm_bytecode_module_benchmark_module_create();
  iree_vm_module_t* bytecode_module = nullptr;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      instance,
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          static_cast<iree_host_size_t>(module_file_toc->size)},
      iree_allocator_null(), iree_allocator_system(), &bytecode_module));

  std::array<iree_vm_module_t*, 2> modules = {import_module, bytecode_module};
  iree_vm_context_t* context = NULL;
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
      iree_allocator_system(), &context));

  iree_vm_function_t function;
  IREE_CHECK_OK(
      iree_vm_context_resolve_function(context, function_name, &function));

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments =
      iree_make_byte_span(iree_alloca(i32_args.size() * sizeof(int32_t)),
                          i32_args.size() * sizeof(int32_t));
  call.results =
      iree_make_byte_span(iree_alloca(result_count * sizeof(int32_t)),
                          result_count * sizeof(int32_t));

  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_INVOCATION_FLAG_NONE,
                                  iree_vm_context_state_resolver(context),
                                  iree_allocator_system());
  while (iree_benchmark_keep_running(benchmark_state, batch_size)) {
    for (iree_host_size_t i = 0; i < i32_args.size(); ++i) {
      reinterpret_cast<int32_t*>(call.arguments.data)[i] = i32_args[i];
    }
    IREE_CHECK_OK(
        bytecode_module->begin_call(bytecode_module->self, stack, call));
  }
  iree_vm_stack_deinitialize(stack);

  iree_vm_module_release(import_module);
  iree_vm_module_release(bytecode_module);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);

  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_ModuleCreate) {
  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        iree_allocator_system(), &instance));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    const auto* module_file_toc =
        iree_vm_bytecode_module_benchmark_module_create();
    iree_vm_module_t* module = nullptr;
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance,
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            static_cast<iree_host_size_t>(module_file_toc->size)},
        iree_allocator_null(), iree_allocator_system(), &module));

    // Just testing creation and verification here!
    iree_optimization_barrier(module);

    iree_vm_module_release(module);
  }

  iree_vm_instance_release(instance);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_ModuleCreate);

IREE_BENCHMARK_FN(BM_ModuleCreateState) {
  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        iree_allocator_system(), &instance));

  const auto* module_file_toc =
      iree_vm_bytecode_module_benchmark_module_create();
  iree_vm_module_t* module = nullptr;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      instance,
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          static_cast<iree_host_size_t>(module_file_toc->size)},
      iree_allocator_null(), iree_allocator_system(), &module));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_vm_module_state_t* module_state;
    module->alloc_state(module->self, iree_allocator_system(), &module_state);

    // Really just testing malloc overhead, though it'll be module-dependent
    // and if we do anything heavyweight on state init it'll show here.
    iree_optimization_barrier(module_state);

    module->free_state(module->self, module_state);
  }

  iree_vm_module_release(module);
  iree_vm_instance_release(instance);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_ModuleCreateState);

IREE_BENCHMARK_FN(BM_FullModuleInit) {
  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        iree_allocator_system(), &instance));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    const auto* module_file_toc =
        iree_vm_bytecode_module_benchmark_module_create();
    iree_vm_module_t* module = nullptr;
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance,
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            static_cast<iree_host_size_t>(module_file_toc->size)},
        iree_allocator_null(), iree_allocator_system(), &module));

    iree_vm_module_state_t* module_state;
    module->alloc_state(module->self, iree_allocator_system(), &module_state);

    iree_optimization_barrier(module_state);

    module->free_state(module->self, module_state);
    iree_vm_module_release(module);
  }

  iree_vm_instance_release(instance);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_FullModuleInit);

IREE_ATTRIBUTE_NOINLINE static int empty_fn(void) {
  int ret = 1;
  iree_optimization_barrier(ret);
  return ret;
}

IREE_BENCHMARK_FN(BM_EmptyFuncReference) {
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    int ret = empty_fn();
    iree_optimization_barrier(ret);
    iree_benchmark_clobber();
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_EmptyFuncReference);

IREE_BENCHMARK_FN(BM_EmptyFuncBytecode) {
  IREE_CHECK_OK(RunFunction(
      benchmark_state,
      iree_make_cstring_view("bytecode_module_benchmark.empty_func"), {},
      /*result_count=*/0));
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_EmptyFuncBytecode);

IREE_ATTRIBUTE_NOINLINE static int add_fn(int value) {
  iree_optimization_barrier(value += value);
  return value;
}

IREE_BENCHMARK_FN(BM_CallInternalFuncReference) {
  while (iree_benchmark_keep_running(benchmark_state, 10)) {
    int value = 1;
    value = add_fn(value);
    iree_optimization_barrier(value);
    value = add_fn(value);
    iree_optimization_barrier(value);
    value = add_fn(value);
    iree_optimization_barrier(value);
    value = add_fn(value);
    iree_optimization_barrier(value);
    value = add_fn(value);
    iree_optimization_barrier(value);
    value = add_fn(value);
    iree_optimization_barrier(value);
    value = add_fn(value);
    iree_optimization_barrier(value);
    value = add_fn(value);
    iree_optimization_barrier(value);
    value = add_fn(value);
    iree_optimization_barrier(value);
    value = add_fn(value);
    iree_optimization_barrier(value);
    iree_benchmark_clobber();
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_CallInternalFuncReference);

IREE_BENCHMARK_FN(BM_CallInternalFuncBytecode) {
  static const int batch = 100;
  return RunFunction(
      benchmark_state,
      iree_make_cstring_view("bytecode_module_benchmark.call_internal_func"),
      {batch},
      /*result_count=*/1,
      /*batch_size=*/batch);
}
IREE_BENCHMARK_REGISTER(BM_CallInternalFuncBytecode);

IREE_BENCHMARK_FN(BM_CallImportedFuncBytecode) {
  static const int batch = 100;
  return RunFunction(
      benchmark_state,
      iree_make_cstring_view("bytecode_module_benchmark.call_imported_func"),
      {batch},
      /*result_count=*/1,
      /*batch_size=*/batch);
}
IREE_BENCHMARK_REGISTER(BM_CallImportedFuncBytecode);

IREE_BENCHMARK_FN(BM_LoopSumReference) {
  static const int batch = 100000;
  static auto work = +[](int x) {
    iree_optimization_barrier(x);
    return x;
  };
  static auto loop = +[](int count) {
    int i = 0;
    for (; i < count; ++i) {
      iree_optimization_barrier(i = work(i));
    }
    return i;
  };
  while (iree_benchmark_keep_running(benchmark_state, batch)) {
    int ret = loop(batch);
    iree_optimization_barrier(ret);
    iree_benchmark_clobber();
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_LoopSumReference);

IREE_BENCHMARK_FN(BM_LoopSumBytecode) {
  static const int batch = 100000;
  return RunFunction(
      benchmark_state,
      iree_make_cstring_view("bytecode_module_benchmark.loop_sum"), {batch},
      /*result_count=*/1,
      /*batch_size=*/batch);
}
IREE_BENCHMARK_REGISTER(BM_LoopSumBytecode);

IREE_BENCHMARK_FN(BM_BufferReduceReference) {
  static const int batch = 100000;
  static auto work = +[](int32_t* buffer, int i, int sum) {
    int new_sum = buffer[i] + sum;
    iree_optimization_barrier(new_sum);
    return new_sum;
  };
  static auto loop = +[](int32_t* buffer, int count) {
    int sum = 0;
    for (int i = 0; i < count; ++i) {
      iree_optimization_barrier(sum = work(buffer, i, sum));
    }
    return sum;
  };
  while (iree_benchmark_keep_running(benchmark_state, batch)) {
    int32_t* buffer = (int32_t*)malloc(batch * 4);
    for (int i = 0; i < batch; ++i) {
      buffer[i] = 1;
    }
    int ret = loop(buffer, batch);
    iree_optimization_barrier(ret);
    iree_benchmark_clobber();
    free(buffer);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BufferReduceReference);

IREE_BENCHMARK_FN(BM_BufferReduceBytecode) {
  static const int batch = 100000;
  return RunFunction(
      benchmark_state,
      iree_make_cstring_view("bytecode_module_benchmark.buffer_reduce"),
      {batch},
      /*result_count=*/1,
      /*batch_size=*/batch);
}
IREE_BENCHMARK_REGISTER(BM_BufferReduceBytecode);

// NOTE: unrolled 8x, requires %count to be % 8 = 0.
IREE_BENCHMARK_FN(BM_BufferReduceBytecodeUnrolled) {
  static const int batch = 100000;
  return RunFunction(benchmark_state,
                     iree_make_cstring_view(
                         "bytecode_module_benchmark.buffer_reduce_unrolled"),
                     {batch},
                     /*result_count=*/1,
                     /*batch_size=*/batch);
}
IREE_BENCHMARK_REGISTER(BM_BufferReduceBytecodeUnrolled);

}  // namespace
