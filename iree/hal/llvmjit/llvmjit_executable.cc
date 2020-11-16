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

#include "iree/base/tracing.h"
#include "iree/hal/buffer.h"
#include "iree/hal/executable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

// flatcc schemas:
#include "iree/base/flatcc.h"
#include "iree/schemas/llvmir_executable_def_reader.h"
#include "iree/schemas/llvmir_executable_def_verifier.h"

// NOTE: starting to port this to C.

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_llvmir_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_LLVMIRExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_LLVMIRExecutableDef_table_t executable_def =
      iree_LLVMIRExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_LLVMIRExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
  for (size_t i = 0; i < entry_point_count; ++i) {
    if (!flatbuffers_string_len(
            flatbuffers_string_vec_at(entry_points_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  if (!flatbuffers_uint8_vec_len(
          iree_LLVMIRExecutableDef_bitcode_module_get(executable_def))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable bitcode_module is missing/empty");
  }

  return iree_ok_status();
}

namespace iree {
namespace hal {
namespace llvmjit {

// static
StatusOr<ref_ptr<LLVMJITExecutable>> LLVMJITExecutable::Load(
    ExecutableSpec spec, bool allow_aliasing_data) {
  IREE_TRACE_SCOPE0("LLVMJITExecutable::Load");

  // Verify and fetch the executable flatbuffer wrapper.
  iree_const_byte_span_t executable_data = iree_make_const_byte_span(
      spec.executable_data.data(), spec.executable_data.size());
  IREE_RETURN_IF_ERROR(
      iree_hal_llvmir_executable_flatbuffer_verify(executable_data));
  iree_LLVMIRExecutableDef_table_t executable_def =
      iree_LLVMIRExecutableDef_as_root(executable_data.data);

  flatbuffers_uint8_vec_t bitcode_module_vec =
      iree_LLVMIRExecutableDef_bitcode_module_get(executable_def);
  auto mem_buffer = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(reinterpret_cast<const char*>(bitcode_module_vec),
                      flatbuffers_uint8_vec_len(bitcode_module_vec)),
      "llvm-ir");
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  llvm::SMDiagnostic sm_diagnostic;
  auto module = llvm::parseAssembly(*mem_buffer, sm_diagnostic, *llvm_context);
  if (!module) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Can't parse LLVMIR Module: " << sm_diagnostic.getMessage().str();
  }
  auto dataLayout = module->getDataLayout();
  llvm::orc::ThreadSafeModule thread_safe_module(std::move(module),
                                                 std::move(llvm_context));
  auto ll_jit = llvm::cantFail(llvm::orc::LLJITBuilder().create());

  llvm::Error err = ll_jit->addIRModule(std::move(thread_safe_module));
  if (err) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Can't add executable module to executable LLJIT"
           << llvm::toString(std::move(err));
  }

  auto llvmjit_serarch_generator =
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix());
  if (!llvmjit_serarch_generator) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Can't resolve symbols in current process: "
           << llvm::toString(llvmjit_serarch_generator.takeError());
  }

  auto& main_jitllvmjit = ll_jit->getMainJITDylib();
  main_jitllvmjit.addGenerator(std::move(llvmjit_serarch_generator.get()));

  auto executable =
      make_ref<LLVMJITExecutable>(spec, std::move(ll_jit), allow_aliasing_data);

  flatbuffers_string_vec_t entry_points =
      iree_LLVMIRExecutableDef_entry_points_get(executable_def);
  executable->symbols_.resize(flatbuffers_string_vec_len(entry_points));
  for (size_t i = 0; i < flatbuffers_string_vec_len(entry_points); ++i) {
    flatbuffers_string_t entry_point =
        flatbuffers_string_vec_at(entry_points, i);
    auto func_symbol = executable->ll_jit_->lookup(
        llvm::StringRef(entry_point, flatbuffers_string_len(entry_point)));
    if (!func_symbol) {
      return NotFoundErrorBuilder(IREE_LOC)
             << "Can't JIT compile function '" << entry_point
             << "': " << llvm::toString(func_symbol.takeError());
    }
    executable->symbols_[i] = func_symbol.get();
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

  llvm::JITEvaluatedSymbol symbol;
  llvm::SmallVector<void*, 4> args;
  llvm::SmallVector<int32_t, 4> push_constant;
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
      IREE_ASSIGN_OR_RETURN(auto memory,
                            io_binding.buffer->MapMemory<uint8_t>(
                                MemoryAccessBitfield::kWrite, io_binding.offset,
                                io_binding.length));
      auto data = memory.mutable_data();
      dispatch_state->args.push_back(data);
    }
  }
  // TODO(ataei): Consider moving this casting to codegen side ?!
  for (int i = 0; i < params.push_constants->values.size(); ++i) {
    dispatch_state->push_constant.push_back(params.push_constants->values[i]);
  }

  return std::move(dispatch_state);
}

Status LLVMJITExecutable::DispatchTile(DispatchState* state,
                                       std::array<uint32_t, 3> workgroup_xyz) {
  IREE_TRACE_SCOPE0("LLVMJITExecutable::DispatchTile");
  auto* dispatch_state = static_cast<LLVMJITDispatchState*>(state);

  auto func_ptr = (void (*)(void**, int32_t*, int32_t, int32_t,
                            int32_t))dispatch_state->symbol.getAddress();
  func_ptr(dispatch_state->args.data(), dispatch_state->push_constant.data(),
           workgroup_xyz[0], workgroup_xyz[1], workgroup_xyz[2]);

  return OkStatus();
}

}  // namespace llvmjit
}  // namespace hal
}  // namespace iree
