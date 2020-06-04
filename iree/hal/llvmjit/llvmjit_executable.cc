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
#include "iree/hal/executable.h"
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
    executable->InsertSymbol(func_symbol.get());
  }

  return executable;
}

void LLVMJITExecutable::InsertSymbol(llvm::JITEvaluatedSymbol symbol) {
  symbols_.push_back(symbol);
}

Status LLVMJITExecutable::Invoke(int func_id,
                                 llvm::MutableArrayRef<void*> args) {
  const auto func_symbol = symbols_[func_id];
  auto exe_func = (void (*)(void**))func_symbol.getAddress();
  exe_func(args.data());
  return OkStatus();
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

}  // namespace llvmjit
}  // namespace hal
}  // namespace iree
