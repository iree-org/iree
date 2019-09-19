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

#include "iree/vm/stack_frame.h"

#include "iree/base/status.h"

namespace iree {
namespace vm {

StackFrame::StackFrame(Function function) : function_(function) {
  const auto* bytecode_def = function_.def().bytecode();
  if (bytecode_def) {
    offset_limit_ = bytecode_def->contents()->Length();
    locals_.resize(bytecode_def->local_count());
  } else {
    locals_.resize(function_.input_count() + function_.result_count());
  }
}

StackFrame::StackFrame(const ImportFunction& function)
    : function_(function), import_function_(&function) {}

Status StackFrame::set_offset(int offset) {
  if (offset < 0 || offset > offset_limit_) {
    return OutOfRangeErrorBuilder(ABSL_LOC)
           << "Offset " << offset
           << " is outside of the bytecode body limit of " << offset_limit_;
  }
  offset_ = offset;
  return OkStatus();
}

}  // namespace vm
}  // namespace iree
