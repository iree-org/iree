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

#include "iree/vm/function.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "iree/base/flatbuffer_util.h"
#include "iree/schemas/type_def_generated.h"

namespace iree {
namespace vm {

namespace {

struct TypeFormatter {
  void operator()(std::string* out, const TypeDef* type_def) const {
    switch (type_def->type_union_type()) {
      case TypeDefUnion::MemRefTypeDef:
        (*this)(out, type_def->type_union_as_MemRefTypeDef());
        return;
      case TypeDefUnion::DeviceTypeDef:
        out->append("device");
        return;
      case TypeDefUnion::CommandBufferTypeDef:
        out->append("command_buffer");
        return;
      case TypeDefUnion::EventTypeDef:
        out->append("event");
        return;
      case TypeDefUnion::SemaphoreTypeDef:
        out->append("semaphore");
        return;
      case TypeDefUnion::FenceTypeDef:
        out->append("fence");
        return;
      default:
        out->append("<invalid>");
        return;
    }
  }

  void operator()(std::string* out,
                  const MemRefTypeDef* mem_ref_type_def) const {
    out->append("memref<");
    if (mem_ref_type_def->shape()) {
      for (int dim : *mem_ref_type_def->shape()) {
        out->append(std::to_string(dim));
        out->append("x");
      }
    } else {
      out->append("?x");
    }
    (*this)(out, mem_ref_type_def->element_type());
    out->append(">");
  }

  void operator()(std::string* out, const ElementTypeDef* type_def) const {
    switch (type_def->type_union_type()) {
      case ElementTypeDefUnion::FloatTypeDef: {
        const auto* float_type_def = type_def->type_union_as_FloatTypeDef();
        out->append("f");
        out->append(std::to_string(float_type_def->width()));
        break;
      }
      case ElementTypeDefUnion::IntegerTypeDef: {
        const auto* int_type_def = type_def->type_union_as_IntegerTypeDef();
        out->append("i");
        out->append(std::to_string(int_type_def->width()));
        break;
      }
      case ElementTypeDefUnion::UnknownTypeDef: {
        const auto* unknown_type_def = type_def->type_union_as_UnknownTypeDef();
        out->append("unknown<");
        auto dialect_str = WrapString(unknown_type_def->dialect());
        out->append(dialect_str.data(), dialect_str.size());
        auto type_data_str = WrapString(unknown_type_def->type_data());
        out->append(type_data_str.data(), type_data_str.size());
        out->append(">");
        break;
      }
      default:
        out->append("<invalid>");
        return;
    }
  }
};

}  // namespace

std::string Function::DebugStringShort() const {
  return absl::StrCat(
      name(), "(",
      type_def().inputs()
          ? absl::StrJoin(*type_def().inputs(), ", ", TypeFormatter())
          : "",
      ") -> (",
      type_def().results()
          ? absl::StrJoin(*type_def().results(), ", ", TypeFormatter())
          : "",
      ")");
}

std::string ImportFunction::DebugStringShort() const {
  // TODO(benvanik): import function strings.
  return "(IMPORT)";
}

}  // namespace vm
}  // namespace iree
