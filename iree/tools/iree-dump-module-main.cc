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

#include <iostream>
#include <string>
#include <utility>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/reflection.h"
#include "flatbuffers/util.h"
#include "iree/base/flatbuffer_util.h"
#include "iree/base/init.h"
#include "iree/schemas/bytecode_module_def_generated.h"

namespace {

using ::flatbuffers::ElementaryType;
using ::flatbuffers::NumToString;
using ::flatbuffers::String;
using ::flatbuffers::TypeTable;

struct TruncatingToStringVisitor : public ::flatbuffers::IterationVisitor {
  std::string s;
  std::string d;

  bool is_truncating_vector = false;
  int vector_depth = 0;

  explicit TruncatingToStringVisitor(std::string delimiter)
      : d(std::move(delimiter)) {}

  void StartSequence() override {
    if (is_truncating_vector) return;
    s += "{";
    s += d;
  }
  void EndSequence() override {
    if (is_truncating_vector) return;
    s += d;
    s += "}";
  }
  void Field(size_t field_idx, size_t set_idx, ElementaryType type,
             bool is_vector, const TypeTable* type_table, const char* name,
             const uint8_t* val) override {
    if (is_truncating_vector) return;
    if (!val) return;
    if (set_idx) {
      s += ",";
      s += d;
    }
    if (name) {
      s += name;
      s += ": ";
    }
  }
  template <typename T>
  void Named(T x, const char* name) {
    if (name) {
      s += name;
    } else {
      s += NumToString(x);
    }
  }
  void UType(uint8_t x, const char* name) override {
    if (is_truncating_vector) return;
    Named(x, name);
  }
  void Bool(bool x) override {
    if (is_truncating_vector) return;
    s += x ? "true" : "false";
  }
  void Char(int8_t x, const char* name) override {
    if (is_truncating_vector) return;
    Named(x, name);
  }
  void UChar(uint8_t x, const char* name) override {
    if (is_truncating_vector) return;
    Named(x, name);
  }
  void Short(int16_t x, const char* name) override {
    if (is_truncating_vector) return;
    Named(x, name);
  }
  void UShort(uint16_t x, const char* name) override {
    if (is_truncating_vector) return;
    Named(x, name);
  }
  void Int(int32_t x, const char* name) override {
    if (is_truncating_vector) return;
    Named(x, name);
  }
  void UInt(uint32_t x, const char* name) override {
    if (is_truncating_vector) return;
    Named(x, name);
  }
  void Long(int64_t x) override {
    if (is_truncating_vector) return;
    s += NumToString(x);
  }
  void ULong(uint64_t x) override {
    if (is_truncating_vector) return;
    s += NumToString(x);
  }
  void Float(float x) override {
    if (is_truncating_vector) return;
    s += NumToString(x);
  }
  void Double(double x) override {
    if (is_truncating_vector) return;
    s += NumToString(x);
  }
  void String(const struct String* str) override {
    if (is_truncating_vector) return;
    ::flatbuffers::EscapeString(str->c_str(), str->size(), &s, true, false);
  }
  void Unknown(const uint8_t*) override {
    if (is_truncating_vector) return;
    s += "(?)";
  }
  void StartVector() override {
    ++vector_depth;
    if (is_truncating_vector) return;
    s += "[ ";
  }
  void EndVector() override {
    --vector_depth;
    if (vector_depth == 0) {
      is_truncating_vector = false;
    }
    if (is_truncating_vector) return;
    s += " ]";
  }
  void Element(size_t i, ElementaryType type, const TypeTable* type_table,
               const uint8_t* val) override {
    if (is_truncating_vector) return;
    if (i > 1024) {
      if (!is_truncating_vector) {
        s += ", ...";
        is_truncating_vector = true;
      }
    } else if (i) {
      s += ", ";
    }
  }
};

}  // namespace

extern "C" int main(int argc, char** argv) {
  iree::InitializeEnvironment(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Syntax: iree-dump-module filename\n";
    return 1;
  }
  std::string module_path = argv[1];
  auto module_fb = iree::FlatBufferFile<iree::vm::BytecodeModuleDef>::LoadFile(
                       iree::vm::BytecodeModuleDefIdentifier(), module_path)
                       .value();
  TruncatingToStringVisitor tos_visitor("\n");
  auto object = reinterpret_cast<const uint8_t*>(module_fb->root());
  flatbuffers::IterateObject(object, module_fb->root()->MiniReflectTypeTable(),
                             &tos_visitor);
  std::cout << tos_visitor.s << std::endl;
  return 0;
}
