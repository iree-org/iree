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

#ifndef IREE_RT_FUNCTION_SIGNATURE_H_
#define IREE_RT_FUNCTION_SIGNATURE_H_

#include <ostream>
#include <string>

namespace iree {
namespace rt {

// Describes the expected calling convention and arguments/results of a
// function.
class FunctionSignature final {
 public:
  FunctionSignature() = default;
  FunctionSignature(int32_t argument_count, int32_t result_count)
      : argument_count_(argument_count), result_count_(result_count) {}

  // TODO(benvanik): pretty printing of function signatures.
  std::string DebugString() const { return "<signature>"; }

  // Total number of arguments to the function.
  int32_t argument_count() const { return argument_count_; }

  // Total number of results from the function.
  int32_t result_count() const { return result_count_; }

 private:
  int32_t argument_count_ = 0;
  int32_t result_count_ = 0;
};

inline std::ostream& operator<<(std::ostream& stream,
                                const FunctionSignature& function_signature) {
  stream << function_signature.DebugString();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_FUNCTION_SIGNATURE_H_
