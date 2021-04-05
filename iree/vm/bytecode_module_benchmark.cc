// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/bytecode_module_benchmark_module.h"

namespace {

struct native_import_module_s;
struct native_import_module_state_s;
typedef struct native_import_module_s native_import_module_t;
typedef struct native_import_module_state_s native_import_module_state_t;

// vm.import @native_import_module.add_1(%arg0 : i32) -> i32
static iree_status_t native_import_module_add_1(
    iree_vm_stack_t* stack, const iree_vm_function_call_t* call,
    iree_vm_native_function_target_t target_fn, void* module,
    void* module_state, iree_vm_execution_result_t* out_result) {
  // Add 1 to arg0 and return.
  int32_t arg0 = *reinterpret_cast<int32_t*>(call->arguments.data);
  int32_t ret0 = arg0 + 1;
  *reinterpret_cast<int32_t*>(call->results.data) = ret0;
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
        iree_make_cstring_view("native_import_module"),
        0,
        NULL,
        IREE_ARRAYSIZE(native_import_module_exports_),
        native_import_module_exports_,
        IREE_ARRAYSIZE(native_import_module_funcs_),
        native_import_module_funcs_,
        0,
        NULL,
};

static iree_status_t native_import_module_create(
    iree_allocator_t allocator, iree_vm_module_t** out_module) {
  iree_vm_module_t interface;
  IREE_RETURN_IF_ERROR(iree_vm_module_initialize(&interface, NULL));
  return iree_vm_native_module_create(
      &interface, &native_import_module_descriptor_, allocator, out_module);
}

// Benchmarks the given exported function, optionally passing in arguments.
static iree_status_t RunFunction(benchmark::State& state,
                                 absl::string_view function_name,
                                 absl::Span<const int32_t> i32_args,
                                 int result_count, int64_t batch_size = 1) {
  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance));

  iree_vm_module_t* import_module = NULL;
  IREE_CHECK_OK(
      native_import_module_create(iree_allocator_system(), &import_module));

  const auto* module_file_toc =
      iree::vm::bytecode_module_benchmark_module_create();
  iree_vm_module_t* bytecode_module = nullptr;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          module_file_toc->size},
      iree_allocator_null(), iree_allocator_system(), &bytecode_module));

  std::array<iree_vm_module_t*, 2> modules = {import_module, bytecode_module};
  iree_vm_context_t* context = NULL;
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      instance, modules.data(), modules.size(), iree_allocator_system(),
      &context));

  iree_vm_function_t function;
  IREE_CHECK_OK(iree_vm_context_resolve_function(
      context,
      iree_make_string_view(function_name.data(), function_name.size()),
      &function));

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments =
      iree_make_byte_span(iree_alloca(i32_args.size() * sizeof(int32_t)),
                          i32_args.size() * sizeof(int32_t));
  call.results =
      iree_make_byte_span(iree_alloca(result_count * sizeof(int32_t)),
                          result_count * sizeof(int32_t));

  IREE_VM_INLINE_STACK_INITIALIZE(
      stack, iree_vm_context_state_resolver(context), iree_allocator_system());
  while (state.KeepRunningBatch(batch_size)) {
    for (iree_host_size_t i = 0; i < i32_args.size(); ++i) {
      reinterpret_cast<int32_t*>(call.arguments.data)[i] = i32_args[i];
    }

    iree_vm_execution_result_t result;
    IREE_CHECK_OK(bytecode_module->begin_call(bytecode_module->self, stack,
                                              &call, &result));
  }
  iree_vm_stack_deinitialize(stack);

  iree_vm_module_release(import_module);
  iree_vm_module_release(bytecode_module);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);

  return iree_ok_status();
}

static void BM_ModuleCreate(benchmark::State& state) {
  while (state.KeepRunning()) {
    const auto* module_file_toc =
        iree::vm::bytecode_module_benchmark_module_create();
    iree_vm_module_t* module = nullptr;
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            module_file_toc->size},
        iree_allocator_null(), iree_allocator_system(), &module))
        << "Bytecode module failed to load";

    // Just testing creation and verification here!
    benchmark::DoNotOptimize(module);

    iree_vm_module_release(module);
  }
}
BENCHMARK(BM_ModuleCreate);

static void BM_ModuleCreateState(benchmark::State& state) {
  const auto* module_file_toc =
      iree::vm::bytecode_module_benchmark_module_create();
  iree_vm_module_t* module = nullptr;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          module_file_toc->size},
      iree_allocator_null(), iree_allocator_system(), &module))
      << "Bytecode module failed to load";

  while (state.KeepRunning()) {
    iree_vm_module_state_t* module_state;
    module->alloc_state(module->self, iree_allocator_system(), &module_state);

    // Really just testing malloc overhead, though it'll be module-dependent
    // and if we do anything heavyweight on state init it'll show here.
    benchmark::DoNotOptimize(module_state);

    module->free_state(module->self, module_state);
  }

  iree_vm_module_release(module);
}
BENCHMARK(BM_ModuleCreateState);

static void BM_FullModuleInit(benchmark::State& state) {
  while (state.KeepRunning()) {
    const auto* module_file_toc =
        iree::vm::bytecode_module_benchmark_module_create();
    iree_vm_module_t* module = nullptr;
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            module_file_toc->size},
        iree_allocator_null(), iree_allocator_system(), &module))
        << "Bytecode module failed to load";

    iree_vm_module_state_t* module_state;
    module->alloc_state(module->self, iree_allocator_system(), &module_state);

    benchmark::DoNotOptimize(module_state);

    module->free_state(module->self, module_state);
    iree_vm_module_release(module);
  }
}
BENCHMARK(BM_FullModuleInit);

IREE_ATTRIBUTE_NOINLINE static int empty_fn() {
  int ret = 1;
  benchmark::DoNotOptimize(ret);
  return ret;
}

static void BM_EmptyFuncReference(benchmark::State& state) {
  while (state.KeepRunning()) {
    int ret = empty_fn();
    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_EmptyFuncReference);

static void BM_EmptyFuncBytecode(benchmark::State& state) {
  IREE_CHECK_OK(RunFunction(state, "bytecode_module_benchmark.empty_func", {},
                            /*result_count=*/0));
}
BENCHMARK(BM_EmptyFuncBytecode);

IREE_ATTRIBUTE_NOINLINE static int add_fn(int value) {
  benchmark::DoNotOptimize(value += value);
  return value;
}

static void BM_CallInternalFuncReference(benchmark::State& state) {
  while (state.KeepRunningBatch(10)) {
    int value = 1;
    value = add_fn(value);
    benchmark::DoNotOptimize(value);
    value = add_fn(value);
    benchmark::DoNotOptimize(value);
    value = add_fn(value);
    benchmark::DoNotOptimize(value);
    value = add_fn(value);
    benchmark::DoNotOptimize(value);
    value = add_fn(value);
    benchmark::DoNotOptimize(value);
    value = add_fn(value);
    benchmark::DoNotOptimize(value);
    value = add_fn(value);
    benchmark::DoNotOptimize(value);
    value = add_fn(value);
    benchmark::DoNotOptimize(value);
    value = add_fn(value);
    benchmark::DoNotOptimize(value);
    value = add_fn(value);
    benchmark::DoNotOptimize(value);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_CallInternalFuncReference);

static void BM_CallInternalFuncBytecode(benchmark::State& state) {
  IREE_CHECK_OK(
      RunFunction(state, "bytecode_module_benchmark.call_internal_func", {100},
                  /*result_count=*/1,
                  /*batch_size=*/20));
}
BENCHMARK(BM_CallInternalFuncBytecode);

static void BM_CallImportedFuncBytecode(benchmark::State& state) {
  IREE_CHECK_OK(
      RunFunction(state, "bytecode_module_benchmark.call_imported_func", {100},
                  /*result_count=*/1,
                  /*batch_size=*/20));
}
BENCHMARK(BM_CallImportedFuncBytecode);

static void BM_LoopSumReference(benchmark::State& state) {
  static auto work = +[](int x) {
    benchmark::DoNotOptimize(x);
    return x;
  };
  static auto loop = +[](int count) {
    int i = 0;
    for (; i < count; ++i) {
      benchmark::DoNotOptimize(i = work(i));
    }
    return i;
  };
  while (state.KeepRunningBatch(state.range(0))) {
    int ret = loop(static_cast<int>(state.range(0)));
    benchmark::DoNotOptimize(ret);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_LoopSumReference)->Arg(100000);

static void BM_LoopSumBytecode(benchmark::State& state) {
  IREE_CHECK_OK(RunFunction(state, "bytecode_module_benchmark.loop_sum",
                            {static_cast<int32_t>(state.range(0))},
                            /*result_count=*/1,
                            /*batch_size=*/state.range(0)));
}
BENCHMARK(BM_LoopSumBytecode)->Arg(100000);

}  // namespace
