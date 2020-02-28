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
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

namespace iree {
namespace hal {
namespace llvmjit {
// TODO(ataei): Move this to compile time HAL/Target/LLVM.
static void CreateInvocationFunc(const std::string& name,
                                 llvm::Module* module) {
  auto& ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  auto var_func = module->getFunction(name);

  auto new_type = llvm::FunctionType::get(
      builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
      /*isVarArg=*/false);

  auto new_name = "invoke_" + name;
  auto func_cst = module->getOrInsertFunction(new_name, new_type);
  llvm::Function* interface_func =
      llvm::cast<llvm::Function>(func_cst.getCallee());

  auto bb = llvm::BasicBlock::Create(ctx);
  bb->insertInto(interface_func);
  builder.SetInsertPoint(bb);
  llvm::Value* argList = interface_func->arg_begin();
  llvm::SmallVector<llvm::Value*, 8> args;
  args.reserve(llvm::size(var_func->args()));
  for (auto& indexedArg : llvm::enumerate(var_func->args())) {
    llvm::Value* arg_index = llvm::Constant::getIntegerValue(
        builder.getInt64Ty(), llvm::APInt(64, indexedArg.index()));
    llvm::Value* arg_ptr_ptr = builder.CreateGEP(argList, arg_index);
    llvm::Value* arg_ptr = builder.CreateLoad(arg_ptr_ptr);
    arg_ptr = builder.CreateBitCast(
        arg_ptr, indexedArg.value().getType()->getPointerTo());
    llvm::Value* arg = builder.CreateLoad(arg_ptr);
    args.push_back(arg);
  }
  builder.CreateCall(var_func, args);
  builder.CreateRetVoid();
}

// static
StatusOr<ref_ptr<LLVMJITExecutable>> LLVMJITExecutable::Load(
    hal::Allocator* allocator, ExecutableSpec spec,
    llvm::orc::LLJIT* execution_engine, bool allow_aliasing_data) {
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

  const auto entry_points = module_def->entry_points();
  // Create an invocation function for each entry point.
  for (const auto func_name : *entry_points) {
    CreateInvocationFunc(func_name->str(), module.get());
  }
  llvm::orc::ThreadSafeModule thread_safe_module(std::move(module),
                                                 std::move(llvm_context));
  llvm::Error err =
      execution_engine->addIRModule(std::move(thread_safe_module));
  if (err)
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Can't add executable module to execution engine";

  auto executable =
      make_ref<LLVMJITExecutable>(allocator, spec, allow_aliasing_data);

  for (const auto func_name : *entry_points) {
    auto func_symbol = execution_engine->lookup("invoke_" + func_name->str());
    if (!func_symbol.get()) {
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

LLVMJITExecutable::LLVMJITExecutable(hal::Allocator* allocator,
                                     ExecutableSpec spec,
                                     bool allow_aliasing_data)
    : spec_(spec) {
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
