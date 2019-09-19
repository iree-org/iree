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

#include "iree/vm/context.h"

#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"

namespace iree {
namespace vm {

namespace {

int NextUniqueId() {
  static int next_id = 0;
  return ++next_id;
}

}  // namespace

Context::Context() : id_(NextUniqueId()) {}

Context::~Context() = default;

Status Context::RegisterNativeFunction(std::string name,
                                       NativeFunction native_function) {
  native_functions_.emplace_back(std::move(name), std::move(native_function));
  return OkStatus();
}

Status Context::RegisterModule(std::unique_ptr<Module> module) {
  // Attempt to link the module.
  RETURN_IF_ERROR(module->mutable_function_table()->ResolveImports(
      [&](const Module& importing_module,
          const FunctionDef& import_function_def) -> StatusOr<ImportFunction> {
        absl::string_view export_name = WrapString(import_function_def.name());

        // Try to find a native function (we prefer these).
        for (const auto& native_function : native_functions_) {
          if (native_function.first == export_name) {
            LOG(INFO) << "Resolved import '" << export_name
                      << "' to native function";
            return ImportFunction(importing_module, import_function_def,
                                  native_function.second);
          }
        }

        // Try to find an export in an existing module.
        // We prefer the more recently registered modules.
        // NOTE: slow O(n*m) search through all modules * exports.
        for (auto it = modules_.rbegin(); it != modules_.rend(); ++it) {
          const auto& module = *it;
          auto export_or = module->function_table().LookupExport(export_name);
          if (export_or.ok()) {
            LOG(INFO) << "Resolved import '" << export_name << "' to module "
                      << module->name();
            return ImportFunction(importing_module, import_function_def,
                                  export_or.ValueOrDie());
          }
        }

        return NotFoundErrorBuilder(ABSL_LOC)
               << "Import '" << export_name << "' could not be resolved";
      }));

  modules_.push_back(std::move(module));
  return OkStatus();
}

StatusOr<const Module*> Context::LookupModule(
    absl::string_view module_name) const {
  return const_cast<Context*>(this)->LookupModule(module_name);
}

StatusOr<Module*> Context::LookupModule(absl::string_view module_name) {
  for (const auto& module : modules_) {
    if (module->name() == module_name) {
      return module.get();
    }
  }
  return NotFoundErrorBuilder(ABSL_LOC)
         << "No module with the name '" << module_name
         << "' has been registered";
}

StatusOr<const Function> Context::LookupExport(
    absl::string_view export_name) const {
  for (const auto& module : modules_) {
    auto export_or = module->function_table().LookupExport(export_name);
    if (export_or.ok()) {
      return export_or.ValueOrDie();
    }
  }
  return NotFoundErrorBuilder(ABSL_LOC)
         << "No export with the name '" << export_name
         << "' is present in the context";
}

}  // namespace vm
}  // namespace iree
