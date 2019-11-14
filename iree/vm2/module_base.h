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

#ifndef IREE_VM2_MODULE_BASE_H_
#define IREE_VM2_MODULE_BASE_H_

#include <cstring>

#include "absl/strings/string_view.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/vm2/module.h"

namespace iree {
namespace vm {

// TODO(benvanik): iree_vm_module_signature_t wrapper.
// TODO(benvanik): iree_vm_function_signature_t wrapper.

// A reference to a function within a module.
class Function {
 public:
  // Describes the type of a function reference.
  enum class Linkage {
    // Function is internal to the module and may not be reflectable.
    kInternal = IREE_VM_FUNCTION_LINKAGE_INTERNAL,
    // Function is an import from another module.
    kImport = IREE_VM_FUNCTION_LINKAGE_IMPORT,
    // Function is an export from the module.
    kExport = IREE_VM_FUNCTION_LINKAGE_EXPORT,
  };

  Function() = default;
  explicit Function(iree_vm_function_t function) : function_(function) {}

  iree_vm_function_t value() const { return function_; }

  // Module the function is contained within.
  iree_vm_module_t* module() const { return function_.module; }

  // Linkage of the function. Note that Linkage::kInternal functions may be
  // missing reflection information.
  Linkage linkage() const { return static_cast<Linkage>(function_.linkage); }

  // Ordinal within the module in the linkage scope.
  int ordinal() const { return function_.ordinal; }

  // Function name, or empty string if unavailable.
  absl::string_view name() const;

  // Function signature defining arguments and results.
  iree_vm_function_signature_t signature() const;

 private:
  iree_vm_function_t function_;
};

// Defines an interface that can be used to reflect and execute functions on a
// module.
//
// Module implementations must be thread-safe as lookups and executions may
// occur in any order from any thread.
class ModuleBase : public RefObject<ModuleBase> {
 public:
  // Internal storage for the module state.
  // Thread-compatible; it's expected that only one thread at a time is
  // executing VM functions and accessing this state.
  class State {
   public:
    virtual ~State() = default;

    // Resolves the import with the given ordinal to |function|.
    // The function is guaranteed to remain valid for the lifetime of the module
    // state.
    virtual Status ResolveImport(int ordinal, Function function) = 0;

   protected:
    State() = default;
  };

  virtual ~ModuleBase();

  // Interface of the module that can be passed to C APIs.
  iree_vm_module_t* interface() const { return &interface_; }

  // The name of the module (used during resolution).
  virtual absl::string_view name() const = 0;

  // The reflected signature of the module.
  virtual iree_vm_module_signature_t signature() const = 0;

  // Gets a function descriptor.
  virtual StatusOr<Function> GetFunction(Function::Linkage linkage,
                                         int ordinal) const = 0;

  // Looks up a function with the given name and linkage in the module.
  // This may perform a linear scan and results should be cached.
  virtual StatusOr<Function> LookupFunction(Function::Linkage linkage,
                                            absl::string_view name) const = 0;

  // Allocates module state data.
  virtual StatusOr<std::unique_ptr<State>> CreateState() const = 0;

  // Asynchronously executes the function specified in the |frame|.
  // This may be called repeatedly for the same frame if the execution
  // previously yielded. The offset within the frame is preserved across calls.
  virtual StatusOr<iree_vm_execution_result_t> Execute(
      iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame) const = 0;

 protected:
  ModuleBase();

  mutable iree_vm_module_t interface_;
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM2_MODULE_BASE_H_
