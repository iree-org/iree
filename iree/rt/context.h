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

#ifndef IREE_RT_CONTEXT_H_
#define IREE_RT_CONTEXT_H_

#include <ostream>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "iree/base/intrusive_list.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/buffer_view.h"
#include "iree/rt/invocation.h"
#include "iree/rt/module.h"
#include "iree/rt/policy.h"

namespace iree {
namespace rt {

class Instance;

using ModuleImportTable = std::pair<Module*, std::vector<Function>>;

// An isolated execution context.
// Effectively a sandbox where modules can be loaded and run with restricted
// visibility and where they can maintain state.
//
// Modules have imports resolved automatically when registered by searching
// existing modules registered within the context and load order is used for
// resolution. For example, target-specific modules should be loaded prior to
// generic modules that may import functions defined there and if a function is
// not available in the target-specific modules the fallback provided by the
// generic module will be used.
//
// Thread-compatible and must be externally synchronized.
class Context final : public RefObject<Context> {
 public:
  Context(ref_ptr<Instance> instance, ref_ptr<Policy> policy);
  ~Context();

  // A process-unique ID for the context.
  int32_t id() const { return id_; }

  // Instance this context uses for shared resources.
  const ref_ptr<Instance>& instance() const { return instance_; }

  // A short human-readable name for the context.
  std::string DebugStringShort() const;

  // A list of modules registered with the context.
  absl::Span<const ref_ptr<Module>> modules() const {
    return absl::MakeConstSpan(modules_);
  }

  // Registers a new module with the context.
  // Imports from the module will be resolved using the existing modules in the
  // context. The module will be retained by the context until destruction.
  Status RegisterModule(ref_ptr<Module> module);

  // Looks up a module by name.
  StatusOr<Module*> LookupModuleByName(absl::string_view module_name) const;

  // Resolves an exported function by fully-qualified name. The function
  // reference is valid for the lifetime of the context.
  StatusOr<const Function> ResolveFunction(absl::string_view full_name) const;

  // Resolves an imported function by import ordinal. The function reference is
  // valid for the lifetime of the context.
  StatusOr<const Function> ResolveImport(const Module* module,
                                         int32_t ordinal) const;

 private:
  // Resolves imports for the given module.
  StatusOr<ModuleImportTable> ResolveImports(Module* module);

  friend class Invocation;
  void RegisterInvocation(Invocation* invocation);
  void UnregisterInvocation(Invocation* invocation);

  int32_t id_;
  ref_ptr<Instance> instance_;
  ref_ptr<Policy> policy_;

  absl::InlinedVector<ref_ptr<Module>, 4> modules_;
  absl::InlinedVector<ModuleImportTable, 4> module_import_tables_;

  absl::Mutex invocations_mutex_;
  IntrusiveList<Invocation, offsetof(Invocation, context_list_link_)>
      invocations_ ABSL_GUARDED_BY(invocations_mutex_);

  friend class Instance;
  IntrusiveListLink instance_list_link_;
};

inline std::ostream& operator<<(std::ostream& stream, const Context& context) {
  stream << context.DebugStringShort();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_CONTEXT_H_
