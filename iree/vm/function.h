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

#ifndef IREE_VM_FUNCTION_H_
#define IREE_VM_FUNCTION_H_

#include <functional>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"
#include "iree/hal/buffer_view.h"
#include "iree/schemas/function_def_generated.h"
#include "iree/schemas/type_def_generated.h"

namespace iree {
namespace vm {

class Stack;
class Module;

// TODO(benvanik): reorganize this; I don't like it. maybe ImportFunction
// shouldn't derive from Function at all?

// A function defined within a Module.
// Imported functions may be of the ImportFunction type and contain additional
// runtime linkage information.
class Function {
 public:
  Function() = default;
  Function(const Module& module, const FunctionDef& function_def)
      : module_(&module), function_def_(&function_def) {}

  absl::string_view name() const { return WrapString(function_def_->name()); }

  const Module& module() const { return *module_; }
  const FunctionDef& def() const { return *function_def_; }
  const FunctionTypeDef& type_def() const { return *def().type(); }

  int input_count() const {
    return type_def().inputs() ? type_def().inputs()->size() : 0;
  }
  int result_count() const {
    return type_def().results() ? type_def().results()->size() : 0;
  }

  std::string DebugStringShort() const;

 private:
  const Module* module_ = nullptr;
  const FunctionDef* function_def_ = nullptr;
};

inline std::ostream& operator<<(std::ostream& stream,
                                const Function& function) {
  stream << function.DebugStringShort();
  return stream;
}

// TODO(benvanik): make an interface as well.
// TODO(benvanik): pass through additional attributes.
using NativeFunction =
    std::function<Status(Stack* stack, absl::Span<hal::BufferView> args,
                         absl::Span<hal::BufferView> results)>;

// A function imported into a Module from either a native function or other
// module.
class ImportFunction : public Function {
 public:
  enum class LinkType {
    kNativeFunction,
    kModule,
  };

  ImportFunction() = default;
  ImportFunction(const Module& module, const FunctionDef& function_def,
                 NativeFunction native_function)
      : Function(module, function_def),
        link_type_(LinkType::kNativeFunction),
        native_function_(std::move(native_function)) {}
  ImportFunction(const Module& module, const FunctionDef& function_def,
                 Function linked_function)
      : Function(module, function_def),
        link_type_(LinkType::kModule),
        linked_function_(std::move(linked_function)) {}
  ImportFunction(const ImportFunction&) = delete;
  ImportFunction& operator=(const ImportFunction&) = delete;
  ImportFunction(ImportFunction&&) = default;
  ImportFunction& operator=(ImportFunction&&) = default;

  LinkType link_type() const { return link_type_; }
  const NativeFunction& native_function() const { return native_function_; }
  const Function& linked_function() const { return linked_function_; }

  std::string DebugStringShort() const;

 private:
  LinkType link_type_;
  NativeFunction native_function_;
  Function linked_function_;
};

inline std::ostream& operator<<(std::ostream& stream,
                                const ImportFunction& function) {
  stream << function.DebugStringShort();
  return stream;
}

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_FUNCTION_H_
