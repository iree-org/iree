// Copyright 2020 Google LLC
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

#include "iree/hal/llvmjit/llvmjit_executable.h"

#include <iostream>
#include <memory>

#include "flatbuffers/flatbuffers.h"
#include "iree/base/tracing.h"
#include "iree/hal/buffer.h"
#include "iree/hal/executable.h"
#include "iree/hal/llvmjit/memref_runtime.h"
#include "iree/schemas/llvmir_executable_def_generated.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

namespace iree {
namespace hal {
namespace llvmjit {

// static
StatusOr<ref_ptr<LLVMJITExecutable>> LLVMJITExecutable::Load(
    ExecutableSpec spec, bool allow_aliasing_data) {
  IREE_TRACE_SCOPE0("LLVMJITExecutable::Load");

  auto module_def =
      ::flatbuffers::GetRoot<LLVMIRExecutableDef>(spec.executable_data.data());
  auto data =
      reinterpret_cast<const char*>(module_def->llvmir_module()->data());
  const int size = module_def->llvmir_module()->size();
  auto mem_buffer = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(data, size), "llvm-ir");
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  llvm::SMDiagnostic sm_diagnostic;
  auto module = llvm::parseAssembly(*mem_buffer, sm_diagnostic, *llvm_context);
  if (!module) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "Can't parse LLVMIR Module";
  }
  auto dataLayout = module->getDataLayout();
  const auto entry_points = module_def->entry_points();
  llvm::orc::ThreadSafeModule thread_safe_module(std::move(module),
                                                 std::move(llvm_context));
  auto ll_jit = llvm::cantFail(llvm::orc::LLJITBuilder().create());

  llvm::Error err = ll_jit->addIRModule(std::move(thread_safe_module));
  if (err) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Can't add executable module to executable LLJIT"
           << llvm::toString(std::move(err));
  }

  auto dylib_serarch_generator =
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix());
  if (!dylib_serarch_generator) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Can't resolve symbols in current process";
  }

  auto& main_jitdylib = ll_jit->getMainJITDylib();
  main_jitdylib.addGenerator(std::move(dylib_serarch_generator.get()));

  auto executable =
      make_ref<LLVMJITExecutable>(spec, std::move(ll_jit), allow_aliasing_data);

  for (const auto func_name : *entry_points) {
    auto func_symbol =
        executable->ll_jit_->lookup("invoke_" + func_name->str());
    if (!func_symbol) {
      return NotFoundErrorBuilder(IREE_LOC)
             << "Can't JIT compile function : " << func_name;
    }
    // Map function to its invoke_ symbol.
    executable->symbols_.push_back(func_symbol.get());
  }

  return executable;
}

LLVMJITExecutable::LLVMJITExecutable(ExecutableSpec spec,
                                     std::unique_ptr<llvm::orc::LLJIT> ll_jit,
                                     bool allow_aliasing_data)
    : spec_(spec), ll_jit_(std::move(ll_jit)) {
  if (!allow_aliasing_data) {
    // Clone data.
    cloned_executable_data_ = {spec.executable_data.begin(),
                               spec.executable_data.end()};
    spec_.executable_data = absl::MakeConstSpan(cloned_executable_data_);
  }
}

LLVMJITExecutable::~LLVMJITExecutable() = default;

struct LLVMJITDispatchState : public HostExecutable::DispatchState {
  LLVMJITDispatchState() = default;
  ~LLVMJITDispatchState() override {
    for (int i = 0; i < descriptors.size(); ++i) {
      freeUnrankedDescriptor(descriptors[i]);
    }
  }

  llvm::JITEvaluatedSymbol symbol;
  llvm::SmallVector<UnrankedMemRefType<uint32_t>*, 4> descriptors;
  llvm::SmallVector<void*, 4> args;
};

StatusOr<ref_ptr<HostExecutable::DispatchState>>
LLVMJITExecutable::PrepareDispatch(const DispatchParams& params) {
  IREE_TRACE_SCOPE0("LLVMJITExecutable::PrepareDispatch");

  if (params.entry_point >= symbols_.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Invalid entry point ordinal " << params.entry_point;
  }

  auto dispatch_state = make_ref<LLVMJITDispatchState>();
  dispatch_state->symbol = symbols_[params.entry_point];

  for (size_t set = 0; set < params.set_bindings.size(); ++set) {
    for (size_t binding = 0; binding < params.set_bindings[set].size();
         ++binding) {
      const auto& io_binding = params.set_bindings[set][binding];
      ASSIGN_OR_RETURN(auto memory, io_binding.buffer->MapMemory<uint8_t>(
                                        MemoryAccessBitfield::kWrite,
                                        io_binding.offset, io_binding.length));
      auto data = memory.mutable_data();
      auto descriptor = allocUnrankedDescriptor<uint32_t>(data);
      dispatch_state->descriptors.push_back(descriptor);
      dispatch_state->args.push_back(&descriptor->descriptor);
    }
  }

  auto push_constants_descriptor = allocUnrankedDescriptor<uint32_t>(
      const_cast<uint32_t*>(params.push_constants->values.data()),
      {static_cast<int64_t>(params.push_constants->values.size())});
  dispatch_state->descriptors.push_back(push_constants_descriptor);
  dispatch_state->args.push_back(&push_constants_descriptor->descriptor);

  return std::move(dispatch_state);
}

Status LLVMJITExecutable::DispatchTile(DispatchState* state,
                                       std::array<uint32_t, 3> workgroup_xyz) {
  IREE_TRACE_SCOPE0("LLVMJITExecutable::DispatchTile");
  auto* dispatch_state = static_cast<LLVMJITDispatchState*>(state);

  auto func_ptr = (void (*)(void**))dispatch_state->symbol.getAddress();
  func_ptr(dispatch_state->args.data());

  return OkStatus();
}

}  // namespace llvmjit
}  // namespace hal
}  // namespace iree
