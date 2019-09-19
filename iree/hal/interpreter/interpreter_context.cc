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

#include "iree/hal/interpreter/interpreter_context.h"

#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"
#include "iree/hal/interpreter/bytecode_dispatch.h"

namespace iree {
namespace hal {

namespace {

using ::iree::vm::Function;

}  // namespace

Status InterpreterContext::Invoke(vm::Stack* stack, Function function,
                                  absl::Span<BufferView> args,
                                  absl::Span<BufferView> results) const {
  // Verify arg/result counts.
  if (args.size() != function.input_count()) {
    return InvalidArgumentErrorBuilder(ABSL_LOC)
           << "Function " << function.name() << " requires "
           << function.input_count() << " inputs but only " << args.size()
           << " provided";
  }
  if (results.size() != function.result_count()) {
    return InvalidArgumentErrorBuilder(ABSL_LOC)
           << "Function " << function.name() << " requires "
           << function.result_count() << " outputs but only " << results.size()
           << " provided";
  }

  // Push stack frame for the function we are calling.
  ASSIGN_OR_RETURN(auto* callee_stack_frame, stack->PushFrame(function));

  // Marshal input arguments.
  for (int i = 0; i < args.size(); ++i) {
    *callee_stack_frame->mutable_local(i) = std::move(args[i]);
  }

  // Run main dispatch loop until it exits (or errors).
  RETURN_IF_ERROR(Dispatch(allocator_, &kernel_runtime_state_, stack,
                           callee_stack_frame, results));

  // Pop the callee frame to balance out the stack.
  RETURN_IF_ERROR(stack->PopFrame());

  return OkStatus();
}

}  // namespace hal
}  // namespace iree
