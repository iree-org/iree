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

#ifndef IREE_HAL_INTERPRETER_TYPE_H_
#define IREE_HAL_INTERPRETER_TYPE_H_

#include "iree/base/status.h"
#include "iree/schemas/bytecode/interpreter_bytecode_v0.h"
#include "iree/schemas/interpreter_module_def_generated.h"

namespace iree {
namespace hal {

class Type {
 public:
  static StatusOr<const Type> FromTypeIndex(uint8_t type_index);
  static const Type FromBuiltin(BuiltinType type);

  std::string DebugString() const;

  uint8_t type_index() const { return type_index_; }

  bool is_opaque() const {
    return type_index_ == static_cast<uint8_t>(BuiltinType::kOpaque);
  }
  bool is_builtin() const { return !is_opaque(); }
  BuiltinType builtin_type() const {
    DCHECK(is_builtin());
    return static_cast<BuiltinType>(type_index_);
  }

  size_t element_size() const;

 private:
  explicit Type(uint8_t type_index) : type_index_(type_index) {}

  uint8_t type_index_;
};

inline bool operator==(const Type& a, const Type& b) {
  return a.type_index() == b.type_index();
}

inline bool operator!=(const Type& a, const Type& b) { return !(a == b); }

inline std::ostream& operator<<(std::ostream& stream, const Type& type) {
  stream << type.DebugString();
  return stream;
}

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_INTERPRETER_TYPE_H_
