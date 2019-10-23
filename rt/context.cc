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

#include "rt/context.h"

#include <atomic>

#include "absl/strings/str_cat.h"
#include "base/status.h"
#include "base/tracing.h"
#include "rt/debug/debug_server.h"
#include "rt/instance.h"
#include "rt/invocation.h"

namespace iree {
namespace rt {

namespace {

int32_t NextUniqueContextId() {
  static std::atomic<int32_t> next_id = {0};
  return ++next_id;
}

}  // namespace

Context::Context(ref_ptr<Instance> instance, ref_ptr<Policy> policy)
    : id_(NextUniqueContextId()),
      instance_(std::move(instance)),
      policy_(std::move(policy)) {
  IREE_TRACE_SCOPE("Context::ctor", int32_t)(id_);
  instance_->RegisterContext(this);
}

Context::~Context() {
  IREE_TRACE_SCOPE("Context::dtor", int32_t)(id_);
  instance_->UnregisterContext(this);
}

std::string Context::DebugStringShort() const {
  return absl::StrCat("context_", id_);
}

Status Context::RegisterModule(ref_ptr<Module> module) {
  IREE_TRACE_SCOPE0("Context::RegisterModule");

  // Ensure no conflicts in naming - we don't support shadowing.
  for (const auto& existing_module : modules_) {
    if (existing_module->name() == module->name()) {
      return FailedPreconditionErrorBuilder(IREE_LOC)
             << "Module '" << module->name()
             << "' has already been registered in the context";
    }
  }

  // Try resolving prior to actually registering; if we can't resolve an import
  // then we want to fail the entire registration.
  ASSIGN_OR_RETURN(auto import_table, ResolveImports(module.get()));

  auto* debug_server = instance_->debug_server();
  if (debug_server) {
    CHECK_OK(debug_server->RegisterContextModule(this, module.get()));
  }

  modules_.push_back(std::move(module));
  module_import_tables_.push_back(std::move(import_table));
  return OkStatus();
}

StatusOr<ModuleImportTable> Context::ResolveImports(Module* module) {
  IREE_TRACE_SCOPE0("Context::ResolveImports");

  int32_t import_count = module->signature().import_function_count();
  ModuleImportTable import_table;
  import_table.first = module;
  import_table.second.resize(import_count);

  for (int32_t i = 0; i < import_count; ++i) {
    ASSIGN_OR_RETURN(auto import_function_name,
                     module->GetFunctionName(Function::Linkage::kImport, i));
    ASSIGN_OR_RETURN(import_table.second[i],
                     ResolveFunction(import_function_name));
  }

  return import_table;
}

StatusOr<Module*> Context::LookupModuleByName(
    absl::string_view module_name) const {
  for (const auto& module : modules_) {
    if (module->name() == module_name) {
      return module.get();
    }
  }
  return NotFoundErrorBuilder(IREE_LOC)
         << "No module with the name '" << module_name
         << "' has been registered";
}

StatusOr<const Function> Context::ResolveFunction(
    absl::string_view full_name) const {
  size_t last_dot = full_name.rfind('.');
  if (last_dot == absl::string_view::npos) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "'" << full_name
           << "' is not fully qualified (expected 'module.function')";
  }
  auto module_name = full_name.substr(0, last_dot);
  auto function_name = full_name.substr(last_dot + 1);
  ASSIGN_OR_RETURN(auto* module, LookupModuleByName(module_name));
  return module->LookupFunctionByName(Function::Linkage::kExport,
                                      function_name);
}

StatusOr<const Function> Context::ResolveImport(const Module* module,
                                                int32_t ordinal) const {
  for (const auto& import_table_ref : module_import_tables_) {
    if (import_table_ref.first == module) {
      const auto& import_table = import_table_ref.second;
      if (ordinal >= import_table.size()) {
        return NotFoundErrorBuilder(IREE_LOC)
               << "Import ordinal " << ordinal
               << " out of bounds of import table (" << import_table.size()
               << ")";
      }
      return import_table[ordinal];
    }
  }
  return NotFoundErrorBuilder(IREE_LOC)
         << "Import ordinal " << ordinal << " not found";
}

void Context::RegisterInvocation(Invocation* invocation) {
  {
    absl::MutexLock lock(&invocations_mutex_);
    invocations_.push_back(invocation);
  }
  auto* debug_server = instance_->debug_server();
  if (debug_server) {
    CHECK_OK(debug_server->RegisterInvocation(invocation));
  }
}

void Context::UnregisterInvocation(Invocation* invocation) {
  auto* debug_server = instance_->debug_server();
  if (debug_server) {
    CHECK_OK(debug_server->UnregisterInvocation(invocation));
  }
  {
    absl::MutexLock lock(&invocations_mutex_);
    invocations_.erase(invocation);
  }
}

}  // namespace rt
}  // namespace iree
