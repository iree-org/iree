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

#ifndef IREE_HAL_INTERPRETER_BYTECODE_CACHE_H_
#define IREE_HAL_INTERPRETER_BYTECODE_CACHE_H_

#include "iree/hal/allocator.h"
#include "iree/hal/executable.h"
#include "iree/hal/executable_cache.h"
#include "iree/rt/instance.h"

namespace iree {
namespace hal {

class BytecodeCache final : public ExecutableCache {
 public:
  BytecodeCache(ref_ptr<rt::Instance> instance, hal::Allocator* allocator);
  ~BytecodeCache() override;

  bool CanPrepareFormat(ExecutableFormat format) const override;

  StatusOr<ref_ptr<Executable>> PrepareExecutable(
      ExecutableCachingModeBitfield mode, const ExecutableSpec& spec) override;

 private:
  ref_ptr<rt::Instance> instance_;
  hal::Allocator* allocator_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_INTERPRETER_BYTECODE_CACHE_H_
