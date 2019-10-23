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

#ifndef IREE_RT_FUNCTION_H_
#define IREE_RT_FUNCTION_H_

#include <cstdint>
#include <ostream>

#include "absl/strings/string_view.h"
#include "rt/function_signature.h"

namespace iree {
namespace rt {

class Module;

// Reference to a function within a module.
// Functions are either visible or hidden from the module interface and may be
// of one Linkage type. Imports and exports are always visible (as they are
// required for dynamic linking) however functions with internal linkage may be
// hidden in optimized builds to reduce the amount of reflection metadata
// required.
class Function final {
 public:
  enum class Linkage {
    // Function is internal to the module and may not be reflectable.
    kInternal = 0,
    // Function is an import from another module.
    kImport = 1,
    // Function is an export from the module.
    kExport = 2,
  };

  Function() = default;
  Function(const Module* module, Linkage linkage, int32_t ordinal)
      : module_(module), linkage_(linkage), ordinal_(ordinal) {}

  // Module the function is contained within.
  const Module* module() const { return module_; }

  // Linkage of the function. Note that Linkage::kInternal functions may be
  // missing reflection information.
  Linkage linkage() const { return linkage_; }

  // Ordinal within the module in the linkage scope.
  int32_t ordinal() const { return ordinal_; }

  // Returns the original name of the function.
  // Internal functions may return empty if debugging info has been stripped.
  absl::string_view name() const;

  // Returns the signature of the function.
  // Always present for imports and exports but may be empty for internal
  // functions if debugging info has been stripped.
  const FunctionSignature signature() const;

 private:
  const Module* module_ = nullptr;
  Linkage linkage_ = Linkage::kInternal;
  int32_t ordinal_ = -1;
};

inline std::ostream& operator<<(std::ostream& stream,
                                const Function& function) {
  stream << '@' << function.name() << '#' << function.ordinal();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_FUNCTION_H_
