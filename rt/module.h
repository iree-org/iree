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

#ifndef IREE_RT_MODULE_H_
#define IREE_RT_MODULE_H_

#include <ostream>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "base/ref_ptr.h"
#include "base/status.h"
#include "hal/buffer_view.h"
#include "rt/function.h"
#include "rt/module_signature.h"

namespace iree {
namespace rt {

class Disassembler;
class SourceResolver;
class Stack;

// Abstract compiled module interface for resolving functions.
//
// Modules are (generally) stateless, immutable, and may exist in multiple
// contexts at the same time.
class Module : public RefObject<Module> {
 public:
  virtual ~Module() = default;

  // Name of the module used to resolve fully-qualified references.
  // The lifetime of the returned reference is not guaranteed beyond the current
  // calling scope and callers must clone it if they want to retain it.
  virtual absl::string_view name() const = 0;

  // A description of the module imports, exports, and other metadata.
  virtual const ModuleSignature signature() const = 0;

  // Returns a resolver capable of resolving functions to source and performing
  // basic debugging logic (such as offset calculation).
  // May be nullptr if debugging info has been stripped.
  virtual SourceResolver* source_resolver() const = 0;

  // Returns a disassembler that can be used to disassemble functions in the
  // module. May be nullptr if debugging info has been stripped or disassembly
  // has been disabled as a compile option.
  virtual Disassembler* disassembler() const = 0;

  // A short human-readable string that matches the compiler formatting.
  virtual std::string DebugStringShort() const = 0;

  // Looks up a visible function by ordinal.
  // Internal functions may not be found if debugging info has been stripped.
  virtual StatusOr<const Function> LookupFunctionByOrdinal(
      Function::Linkage linkage, int32_t ordinal) const = 0;

  // Looks up a visible function by name.
  // Internal functions may not be found if debugging info has been stripped.
  virtual StatusOr<const Function> LookupFunctionByName(
      Function::Linkage linkage, absl::string_view name) const = 0;

  // Returns the name of the visible function as a string reference.
  //
  // May return empty for functions with internal linkage if debugging info has
  // been stripped.
  //
  // The lifetime of the returned reference is not guaranteed beyond the current
  // calling scope and callers must clone it if they want to retain it.
  virtual StatusOr<absl::string_view> GetFunctionName(
      Function::Linkage linkage, int32_t ordinal) const = 0;

  // Returns the full function signature for the given |ordinal|.
  //
  // May return empty for functions with internal linkage if the debugging info
  // has been stripped.
  virtual StatusOr<const FunctionSignature> GetFunctionSignature(
      Function::Linkage linkage, int32_t ordinal) const = 0;

  // Temporary until scheduler is built.
  virtual Status Execute(
      Stack* stack, const Function function,
      absl::InlinedVector<hal::BufferView, 8> arguments,
      absl::InlinedVector<hal::BufferView, 8>* results) const = 0;

 protected:
  Module() = default;
};

inline std::ostream& operator<<(std::ostream& stream, const Module& module) {
  stream << module.DebugStringShort();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_MODULE_H_
