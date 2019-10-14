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

#ifndef IREE_RT_MODULE_SIGNATURE_H_
#define IREE_RT_MODULE_SIGNATURE_H_

#include <ostream>
#include <string>

namespace iree {
namespace rt {

// Describes the imports, exports, and capabilities of a module.
class ModuleSignature final {
 public:
  ModuleSignature(int32_t import_function_count, int32_t export_function_count,
                  int32_t internal_function_count, int32_t state_slot_count)
      : import_function_count_(import_function_count),
        export_function_count_(export_function_count),
        internal_function_count_(internal_function_count),
        state_slot_count_(state_slot_count) {}

  // TODO(benvanik): pretty printing of module signatures.
  std::string DebugString() const { return "<signature>"; }

  // Total number of imported functions.
  int32_t import_function_count() const { return import_function_count_; }

  // Total number of exported functions.
  int32_t export_function_count() const { return export_function_count_; }

  // Total number of internal functions, if debugging info is present and they
  // can be queried.
  int32_t internal_function_count() const { return internal_function_count_; }

  // Total number of state block resource slots consumed.
  int32_t state_slot_count() const { return state_slot_count_; }

 private:
  int32_t import_function_count_;
  int32_t export_function_count_;
  int32_t internal_function_count_;
  int32_t state_slot_count_;
};

inline std::ostream& operator<<(std::ostream& stream,
                                const ModuleSignature& module_signature) {
  stream << module_signature.DebugString();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_MODULE_SIGNATURE_H_
