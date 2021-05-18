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

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

void GenerateIntroComment(std::ofstream& f) {
  f << "// Auto-generated static library header file.\n\n";
}

void GenerateIfDefOpen(std::ofstream& f, const std::string& library_name) {
  std::string uppercase = library_name;
  std::transform(uppercase.begin(), uppercase.end(), uppercase.begin(),
                 ::toupper);
  f << "#ifndef IREE_GENERATED_STATIC_EXECUTABLE_LIBRARY_" << uppercase
    << "_\n";
  f << "#define IREE_GENERATED_STATIC_EXECUTABLE_LIBRARY_" << uppercase
    << "_\n";
}

void GenerateIfDefClose(std::ofstream& f, const std::string& library_name) {
  std::string uppercase = library_name;
  std::transform(uppercase.begin(), uppercase.end(), uppercase.begin(),
                 ::toupper);
  f << "#endif // IREE_GENERATED_STATIC_EXECUTABLE_LIBRARY_" << uppercase
    << "_\n";
}

void GenerateExecutableLibraryInclude(std::ofstream& f) {
  f << "#include \"iree/hal/local/executable_library.h\"\n";
}

void GenerateExternCOpen(std::ofstream& f) {
  f << "\n#if __cplusplus\n";
  f << "extern \"C\" {\n";
  f << "#endif // __cplusplus\n\n";
}

void GenerateExternCClose(std::ofstream& f) {
  f << "\n#if __cplusplus\n";
  f << "}\n";
  f << "#endif // __cplusplus\n\n";
}

void GenerateQueryFunction(std::ofstream& f,
                           const std::string& query_function_name) {
  f << "const iree_hal_executable_library_header_t**\n";
  f << query_function_name << "(\n";
  f << "iree_hal_executable_library_version_t max_version, void* reserved);\n";
}

}  // namespace

bool GenerateExecutableLibraryHeader(const std::string& library_name,
                                     const std::string& query_function_name,
                                     const std::string& header_file_path) {
  std::ofstream f(header_file_path, std::ios::out | std::ios::trunc);

  GenerateIntroComment(f);

  GenerateIfDefOpen(f, library_name);

  GenerateExecutableLibraryInclude(f);

  GenerateExternCOpen(f);

  GenerateQueryFunction(f, query_function_name);

  GenerateExternCClose(f);

  GenerateIfDefClose(f, library_name);

  f.close();
  return f.good();
}

bool OutputStaticLibrary(const std::string& library_name,
                         const std::string& query_function_name,
                         const std::string& library_output_path,
                         const std::string& temp_object_path) {
  llvm::SmallString<32> object_file_path(library_output_path);
  llvm::sys::path::replace_extension(object_file_path, ".o");
  llvm::SmallString<32> header_file_path(library_output_path);
  llvm::sys::path::replace_extension(header_file_path, ".h");

  // Copy the object file
  llvm::SmallString<32> copy_from_path(temp_object_path);
  if (llvm::sys::fs::copy_file(copy_from_path, object_file_path)) {
    return false;
  }

  // Generate the header file
  return GenerateExecutableLibraryHeader(library_name, query_function_name,
                                         header_file_path.c_str());
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
