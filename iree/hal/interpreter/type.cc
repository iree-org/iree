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

#include "iree/hal/interpreter/type.h"

#include "iree/base/status.h"

namespace iree {
namespace hal {

// static
StatusOr<const Type> Type::FromTypeIndex(uint8_t type_index) {
  // Currently we only support the builtin types.
  if (type_index == static_cast<uint8_t>(BuiltinType::kOpaque)) {
    return Type(type_index);
  } else if (type_index < kBuiltinTypeCount) {
    return Type(type_index);
  }
  return InvalidArgumentErrorBuilder(IREE_LOC)
         << "Type index " << static_cast<int>(type_index) << " not supported";
}

// static
const Type Type::FromBuiltin(BuiltinType type) {
  return Type(static_cast<uint8_t>(type));
}

std::string Type::DebugString() const {
  switch (type_index_) {
#define TYPE_NAME(index, name, str, size) \
  case index:                             \
    return str;
    IREE_TYPE_LIST(TYPE_NAME)
#undef TYPE_NAME
    default:
      return "<invalid>";
  }
}

size_t Type::element_size() const {
  switch (type_index_) {
#define TYPE_SIZE(index, name, str, size) \
  case index:                             \
    return size;
    IREE_TYPE_LIST(TYPE_SIZE)
#undef TYPE_SIZE
    default:
      return 0;
  }
}

}  // namespace hal
}  // namespace iree
