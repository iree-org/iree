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

#ifndef THIRD_PARTY_MLIR_EDGE_IREE_VM_CONTEXT_H_
#define THIRD_PARTY_MLIR_EDGE_IREE_VM_CONTEXT_H_

#include <memory>
#include <vector>

#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/mlir_edge/iree/base/status.h"
#include "third_party/mlir_edge/iree/vm/function.h"
#include "third_party/mlir_edge/iree/vm/module.h"

namespace iree {
namespace vm {

// An isolated execution context.
// Effectively a sandbox where modules can be loaded and run with restricted
// visibility. Each context may have its own set of imports that modules can
// access and its own resource constraints.
//
// The function namespace is shared within a context, meaning that an import of
// function 'a' from a module will resolve to an export of function 'a' from
// another. Functions internal to a module are not resolved through the
// namespace and may share names (or have no names at all).
//
// Modules have imports resolved automatically when loaded by searching existing
// modules. This means that load order is important to ensure overrides are
// respected. For example, target-specific modules should be loaded prior to
// generic modules that may import functions defined there and if a function is
// not available in the target-specific modules the fallback provided by the
// generic module will be used.
//
// TODO(benvanik): evaluate if worth making thread-safe (epochs/generational).
// Contexts are thread-compatible; const methods may be called concurrently from
// any thread (including Invoke), however no threads must be using a shared
// Context while new native functions or modules are registered.
class Context {
 public:
  Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = default;
  Context& operator=(Context&&) = default;
  virtual ~Context();

  // A process-unique ID for the context.
  int id() const { return id_; }

  // TODO(benvanik): make immutable by moving to a static Create fn.
  virtual Status RegisterNativeFunction(std::string name,
                                        NativeFunction native_function);

  virtual Status RegisterModule(std::unique_ptr<Module> module);

  const std::vector<std::pair<std::string, NativeFunction>>& native_functions()
      const {
    return native_functions_;
  }

  const std::vector<std::unique_ptr<Module>>& modules() const {
    return modules_;
  }

  StatusOr<const Module*> LookupModule(absl::string_view module_name) const;
  StatusOr<Module*> LookupModule(absl::string_view module_name);
  StatusOr<const Function> LookupExport(absl::string_view export_name) const;

 private:
  int id_;
  std::vector<std::pair<std::string, NativeFunction>> native_functions_;
  std::vector<std::unique_ptr<Module>> modules_;
};

}  // namespace vm
}  // namespace iree

#endif  // THIRD_PARTY_MLIR_EDGE_IREE_VM_CONTEXT_H_
