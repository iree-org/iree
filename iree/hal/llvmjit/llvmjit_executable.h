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

#ifndef IREE_HAL_LLVMJIT_LLVMJIT_EXECUTABLE_H_
#define IREE_HAL_LLVMJIT_LLVMJIT_EXECUTABLE_H_

#include <vector>

#include "iree/base/status.h"
#include "iree/hal/executable_spec.h"
#include "iree/hal/host/host_executable.h"
#include "iree/schemas/llvmir_executable_def_generated.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace iree {
namespace hal {
namespace llvmjit {

struct MemrefType;

class LLVMJITExecutable final : public HostExecutable {
 public:
  static StatusOr<ref_ptr<LLVMJITExecutable>> Load(ExecutableSpec spec,
                                                   bool allow_aliasing_data);

  LLVMJITExecutable(ExecutableSpec spec,
                    std::unique_ptr<llvm::orc::LLJIT> ll_jit,
                    bool allow_aliasing_data);
  ~LLVMJITExecutable() override;

  bool supports_debugging() const override { return false; }

  StatusOr<ref_ptr<DispatchState>> PrepareDispatch(
      const DispatchParams& params) override;
  Status DispatchTile(DispatchState* state,
                      std::array<uint32_t, 3> workgroup_xyz) override;

 private:
  ExecutableSpec spec_;
  std::vector<uint8_t> cloned_executable_data_;
  std::unique_ptr<llvm::orc::LLJIT> ll_jit_;
  llvm::SmallVector<llvm::JITEvaluatedSymbol, 4> symbols_;
};

}  // namespace llvmjit
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_LLVMJIT_LLVMJIT_EXECUTABLE_H_
