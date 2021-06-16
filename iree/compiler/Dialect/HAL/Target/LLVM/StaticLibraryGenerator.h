// Copyright 2021 Google LLC
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
//

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_STATICLIBRARYGENERATOR_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_STATICLIBRARYGENERATOR_H_

#include <string>

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Produces a static executable library and generated '.h'.
// The temporary object file is copied to the library_output_path. The '.h' file
// with the query_function_name is placed beside it (using the same base
// filename of the library). Returns true if successful.
bool outputStaticLibrary(const std::string &library_name,
                         const std::string &query_function_name,
                         const std::string &library_output_path,
                         const std::string &temp_object_path);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_STATICLIBRARYGENERATOR_H_
