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

#include "iree/vm/sequencer_context.h"

#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"
#include "iree/hal/buffer_view.h"
#include "iree/vm/fiber_state.h"
#include "iree/vm/sequencer_dispatch.h"

namespace iree {
namespace vm {

namespace {

using ::iree::hal::BufferView;

Status ValidateElementSize(int element_bit_width,
                           const ElementTypeDef& expected_element_type) {
  switch (expected_element_type.type_union_type()) {
    case ElementTypeDefUnion::FloatTypeDef: {
      auto expected_bit_width =
          expected_element_type.type_union_as_FloatTypeDef()->width();
      if (element_bit_width != expected_bit_width) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Has element bit width " << element_bit_width
               << " but expected " << expected_bit_width;
      }
      return OkStatus();
    }
    case ElementTypeDefUnion::IntegerTypeDef: {
      auto expected_bit_width =
          expected_element_type.type_union_as_IntegerTypeDef()->width();
      if (element_bit_width != expected_bit_width) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Has element bit width " << element_bit_width
               << " but expected " << expected_bit_width;
      }
      return OkStatus();
    }
    case ElementTypeDefUnion::UnknownTypeDef:
    case ElementTypeDefUnion::NONE: {
    }
  }
  return InvalidArgumentErrorBuilder(IREE_LOC)
         << "Defined type has unsupported element type "
         << EnumNameElementTypeDefUnion(
                expected_element_type.type_union_type());
}

Status ValidateArgType(const BufferView& arg,
                       const MemRefTypeDef& expected_type) {
  RETURN_IF_ERROR(
      ValidateElementSize(arg.element_size * 8, *expected_type.element_type()));

  auto expected_shape = expected_type.shape();
  if (arg.shape.size() != expected_shape->size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Argument should have rank " << expected_shape->size()
           << " but has rank " << arg.shape.size();
  }
  for (int i = 0; i < expected_shape->size(); ++i) {
    auto dim_size = arg.shape[i];
    auto expected_dim_size = expected_shape->Get(i);
    if (dim_size != expected_dim_size) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Argument dimension " << i << " should have size "
             << expected_dim_size << " but has size " << dim_size;
    }
  }
  return OkStatus();
}

}  // namespace

SequencerContext::SequencerContext(std::shared_ptr<Instance> instance)
    : instance_(std::move(instance)) {}

SequencerContext::~SequencerContext() = default;

Status SequencerContext::RegisterNativeFunction(
    std::string name, NativeFunction native_function) {
  // TODO(benvanik): provide to debugger.
  return Context::RegisterNativeFunction(std::move(name),
                                         std::move(native_function));
}

Status SequencerContext::RegisterModule(std::unique_ptr<Module> module) {
  RETURN_IF_ERROR(Context::RegisterModule(std::move(module)));
  return OkStatus();
}

Status SequencerContext::Invoke(FiberState* fiber_state, Function function,
                                absl::Span<BufferView> args,
                                absl::Span<BufferView> results) const {
  // Verify arg/result counts.
  if (args.size() != function.input_count()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Function " << function.name() << " requires "
           << function.input_count() << " inputs but " << args.size()
           << " provided";
  }
  if (results.size() != function.result_count()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Function " << function.name() << " requires "
           << function.result_count() << " outputs but " << results.size()
           << " provided";
  }

  // Push stack frame for the function we are calling.
  auto* stack = fiber_state->mutable_stack();
  ASSIGN_OR_RETURN(auto* callee_stack_frame, stack->PushFrame(function));

  // Marshal input arguments.
  for (int i = 0; i < args.size(); ++i) {
    auto arg = args[i];
    auto expected_arg_type = function.type_def().inputs()->Get(i);
    RETURN_IF_ERROR(
        ValidateArgType(arg, *expected_arg_type->type_union_as_MemRefTypeDef()))
        << "Function " << function.name() << " argument " << i;
    *callee_stack_frame->mutable_local(i) = std::move(arg);
  }

  // TODO(benvanik): change to:
  //   get command queue (any command queue)
  //   make command buffer
  //   record dispatch
  //   submit
  //   wait on fence
  ASSIGN_OR_RETURN(auto placement,
                   instance_->device_manager()->ResolvePlacement({}));
  RETURN_IF_ERROR(
      DispatchSequence(placement, stack, callee_stack_frame, results));

  // Pop the callee frame to balance out the stack.
  RETURN_IF_ERROR(stack->PopFrame());

  return OkStatus();
}

}  // namespace vm
}  // namespace iree
