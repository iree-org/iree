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

static void generatePrefix(llvm::raw_ostream &os,
                           const std::string &library_name,
                           const std::string &query_function_name) {
  // Intro Comment:
  os << "// === [" << library_name << "] static library ===\n"
     << "//\n"
     << "// To use:\n"
     << "//  - Include this header and generated object into your program \n"
     << "//  - At runtime: retrieve library name from host binary \n"
     << "//  - Query library from " << query_function_name << "()<< \n"
     << "//  - Feed library into static_library_loader \n"
     << "//\n"
     << "// === Automatically generated file. DO NOT EDIT! === \n\n";

  // IfDef Open:
  llvm::StringRef ref(library_name);
  os << "#ifndef IREE_GENERATED_STATIC_EXECUTABLE_LIBRARY_" << ref.upper()
     << "_\n"
     << "#define IREE_GENERATED_STATIC_EXECUTABLE_LIBRARY_" << ref.upper()
     << "_\n";

  // Executable Library Include:
  os << "\n#include \"iree/hal/local/executable_library.h\"\n";

  // ExternC Open:
  os << "\n#if __cplusplus\n"
     << "extern \"C\" {\n"
     << "#endif // __cplusplus\n\n";
}

static void generateQueryFunction(llvm::raw_ostream &os,
                                  const std::string &library_name,
                                  const std::string &query_function_name) {
  os << "const iree_hal_executable_library_header_t**\n"
     << query_function_name << "(\n"
     << "iree_hal_executable_library_version_t max_version, void* reserved);\n";
}

static void generateSuffix(llvm::raw_ostream &os,
                           const std::string &library_name,
                           const std::string &query_function_name) {
  // ExternC Close:
  os << "\n#if __cplusplus\n"
     << "}\n"
     << "#endif // __cplusplus\n\n";

  // IfDef Close:
  llvm::StringRef ref(library_name);
  os << "#endif // IREE_GENERATED_STATIC_EXECUTABLE_LIBRARY_" << ref.upper()
     << "_\n";
}

static bool generateExecutableLibraryHeader(
    const std::string &library_name, const std::string &query_function_name,
    const std::string &header_file_path) {
  std::error_code ec;
  llvm::raw_fd_ostream os(header_file_path, ec);

  generatePrefix(os, library_name, query_function_name);
  generateQueryFunction(os, library_name, query_function_name);
  generateSuffix(os, library_name, query_function_name);

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

  // Copy the object file.
  llvm::SmallString<32> copy_from_path(temp_object_path);
  if (llvm::sys::fs::copy_file(copy_from_path, object_file_path)) {
    return false;
  }

  // Generate the header file.
  return generateExecutableLibraryHeader(library_name, query_function_name,
                                         header_file_path.c_str());
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
