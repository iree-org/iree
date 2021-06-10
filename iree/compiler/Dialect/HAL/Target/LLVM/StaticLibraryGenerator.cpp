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
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

void generateIntroComment(llvm::raw_ostream &os,
  const std::string &library_name,
  const std::string &query_function_name) {
  os << "// === [" << library_name <<  "] static library ===\n";
  os << "//\n";
  os << "// To use:\n";
  os << "//  - Include this header and generated object into your program \n";
  os << "//  - At runtime: retrieve library name from host binary \n";
  os << "//  - Query library from " << query_function_name << "()<< \n";
  os << "//  - Feed library into static_library_loader \n";
  os << "//\n";
  os << "// === Automatically generated file. DO NOT EDIT! === \n\n";
}

void generateIfDefOpen(llvm::raw_ostream &os, const std::string &library_name) {
  llvm::StringRef ref(library_name);
  os << "#ifndef IREE_GENERATED_STATIC_EXECUTABLE_LIBRARY_" << ref.upper()
    << "_\n";
  os << "#define IREE_GENERATED_STATIC_EXECUTABLE_LIBRARY_" << ref.upper()
    << "_\n";
}

void generateIfDefClose(llvm::raw_ostream &os, const std::string &library_name) {
  std::string uppercase = library_name;
  std::transform(uppercase.begin(), uppercase.end(), uppercase.begin(),
                 ::toupper);
  os << "#endif // IREE_GENERATED_STATIC_EXECUTABLE_LIBRARY_" << uppercase
    << "_\n";
}

void generateExecutableLibraryInclude(llvm::raw_ostream &os) {
  os << "\n#include \"iree/hal/local/executable_library.h\"\n";
}

void generateExternCOpen(llvm::raw_ostream &os) {
  os << "\n#if __cplusplus\n";
  os << "extern \"C\" {\n";
  os << "#endif // __cplusplus\n\n";
}

void generateExternCClose(llvm::raw_ostream &os) {
  os << "\n#if __cplusplus\n";
  os << "}\n";
  os << "#endif // __cplusplus\n\n";
}

void generateQueryFunction(llvm::raw_ostream &os,
                           const std::string &query_function_name) {
  os << "const iree_hal_executable_library_header_t**\n";
  os << query_function_name << "(\n";
  os << "iree_hal_executable_library_version_t max_version, void* reserved);\n";
}

}  // namespace

bool generateExecutableLibraryHeader(const std::string &library_name,
                                     const std::string &query_function_name,
                                     const std::string &header_file_path) {
  std::error_code ec;
  llvm::raw_fd_ostream os(header_file_path, ec);

  generateIntroComment(os, library_name, query_function_name);
  generateIfDefOpen(os, library_name);
  generateExecutableLibraryInclude(os);
  generateExternCOpen(os);
  generateQueryFunction(os, query_function_name);
  generateExternCClose(os);
  generateIfDefClose(os, library_name);

  os.close();
  return !os.has_error();
}

bool outputStaticLibrary(const std::string &library_name,
                         const std::string &query_function_name,
                         const std::string &library_output_path,
                         const std::string &temp_object_path) {
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
  return generateExecutableLibraryHeader(library_name, query_function_name,
                                         header_file_path.c_str());
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
