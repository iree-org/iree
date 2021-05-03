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

#include <fstream>
#include <iomanip>
#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"

ABSL_FLAG(std::string, identifier, "resources",
          "name of the resources function");
ABSL_FLAG(std::string, output_header, "", "output header file");
ABSL_FLAG(std::string, output_impl, "", "output impl file");
ABSL_FLAG(std::string, cpp_namespace, "", "generate in a c++ namespace");
ABSL_FLAG(bool, c_output, false, "generate a c output");
ABSL_FLAG(std::string, strip_prefix, "", "strip prefix from filenames");
ABSL_FLAG(bool, flatten, false,
          "whether to flatten the directory structure (only include basename)");

void GenerateExternCOpen(std::ofstream& f) {
  f << "\n#if __cplusplus\n";
  f << "extern \"C\" {\n";
  f << "#endif // __cplusplus\n";
}

void GenerateExternCClose(std::ofstream& f) {
  f << "#if __cplusplus\n";
  f << "}\n";
  f << "#endif // __cplusplus\n\n";
}

void GenerateNamespaceOpen(std::ofstream& f) {
  const auto& ns = absl::GetFlag(FLAGS_cpp_namespace);
  if (ns.empty()) return;

  std::vector<std::string> ns_comps =
      absl::StrSplit(absl::GetFlag(FLAGS_cpp_namespace), absl::ByString("::"));
  for (const auto& ns_comp : ns_comps) {
    f << "namespace " << ns_comp << " {\n";
  }
}

void GenerateNamespaceClose(std::ofstream& f) {
  const auto& ns = absl::GetFlag(FLAGS_cpp_namespace);
  if (ns.empty()) return;

  std::vector<std::string> ns_comps =
      absl::StrSplit(absl::GetFlag(FLAGS_cpp_namespace), absl::ByString("::"));
  for (size_t i = 0, e = ns_comps.size(); i < e; ++i) {
    f << "}\n";
  }
}

void GenerateTocStruct(std::ofstream& f) {
  const auto& c_output = absl::GetFlag(FLAGS_c_output);
  f << "#ifndef IREE_FILE_TOC\n";
  f << "#define IREE_FILE_TOC\n";
  if (c_output) {
    GenerateExternCOpen(f);
  } else {
    f << "namespace iree {\n";
  }
  f << "struct iree_file_toc_t {\n";
  f << "  const char* name;             // the file's original name\n";
  f << "  const char* data;             // beginning of the file\n";
  if (c_output) {
    f << "  size_t size;                  // length of the file\n";
    f << "};\n";
    GenerateExternCClose(f);
  } else {
    f << "  std::size_t size;             // length of the file\n";
    f << "};\n";
    f << "}  // namespace iree\n";
  }
  f << "#endif  // IREE_FILE_TOC\n";
}

bool GenerateHeader(const std::string& header_file,
                    const std::vector<std::string>& toc_files) {
  std::ofstream f(header_file, std::ios::out | std::ios::trunc);
  const auto& c_output = absl::GetFlag(FLAGS_c_output);

  f << "#pragma once\n";  // Pragma once isn't great but is the best we can do.
  if (c_output) {
    f << "#include <stddef.h>\n";
    GenerateTocStruct(f);
    GenerateExternCOpen(f);
    f << "const struct iree_file_toc_t* " << absl::GetFlag(FLAGS_identifier)
      << "_create();\n";
    f << "static inline size_t " << absl::GetFlag(FLAGS_identifier)
      << "_size() {\n";
    f << "  return " << toc_files.size() << ";\n";
    f << "}\n";
    GenerateExternCClose(f);
  } else {
    f << "#include <cstddef>\n";
    GenerateTocStruct(f);
    GenerateNamespaceOpen(f);
    f << "extern const struct ::iree::iree_file_toc_t* "
      << absl::GetFlag(FLAGS_identifier) << "_create();\n";
    f << "static inline std::size_t " << absl::GetFlag(FLAGS_identifier)
      << "_size() { \n";
    f << "  return " << toc_files.size() << ";\n";
    f << "}\n";
    GenerateNamespaceClose(f);
  }
  f.close();
  return f.good();
}

bool SlurpFile(const std::string& file_name, std::string* contents) {
  constexpr std::streamoff kMaxSize = 100000000;
  std::ifstream f(file_name, std::ios::in | std::ios::binary);
  // get length of file:
  f.seekg(0, f.end);
  std::streamoff length = f.tellg();
  f.seekg(0, f.beg);
  if (!f.good()) return false;

  if (length > kMaxSize) {
    std::cerr << "File " << file_name << " is too large\n";
    return false;
  }

  size_t mem_length = static_cast<size_t>(length);
  contents->resize(mem_length);
  f.read(&(*contents)[0], mem_length);
  f.close();
  return f.good();
}

bool GenerateImpl(const std::string& impl_file,
                  const std::vector<std::string>& input_files,
                  const std::vector<std::string>& toc_files) {
  std::ofstream f(impl_file, std::ios::out | std::ios::trunc);
  const auto& c_output = absl::GetFlag(FLAGS_c_output);
  if (c_output) {
    f << "#include <stddef.h>\n";
    f << R"(
#if !defined(IREE_DATA_ALIGNAS_PTR)
#if defined(_MSVC)
#define IREE_DATA_ALIGNAS_PTR __declspec(align(__alignof(void*)))
#else
#include <stdalign.h>
#define IREE_DATA_ALIGNAS_PTR alignas(alignof(void*))
#endif  // _MSVC
#endif  // !IREE_DATA_ALIGNAS_PTR
    )";
    GenerateTocStruct(f);
  } else {
    f << "#include <cstddef>\n";
    f << "#define IREE_DATA_ALIGNAS_PTR alignas(alignof(void*))\n";
    GenerateTocStruct(f);
    GenerateNamespaceOpen(f);
  }
  for (size_t i = 0, e = input_files.size(); i < e; ++i) {
    f << "IREE_DATA_ALIGNAS_PTR static char const file_" << i << "[] = {\n";
    std::string contents;
    if (!SlurpFile(input_files[i], &contents)) {
      std::cerr << "Error reading file " << input_files[i] << "\n";
      return false;
    }
    absl::string_view remaining_contents = contents;
    constexpr int kMaxBytesPerLine = 1024;
    while (!remaining_contents.empty()) {
      auto line = remaining_contents.substr(0, kMaxBytesPerLine);
      f << "\"" << absl::CHexEscape(line) << "\"\n";
      remaining_contents = remaining_contents.substr(line.size());
    }
    f << "};\n";
  }
  if (c_output) {
    f << "static const struct iree_file_toc_t toc[] = {\n";
  } else {
    f << "static const struct ::iree::iree_file_toc_t toc[] = {\n";
  }
  assert(input_files.size() == toc_files.size());
  for (size_t i = 0, e = input_files.size(); i < e; ++i) {
    f << "  {\n";
    f << "    \"" << absl::CEscape(toc_files[i]) << "\",\n";
    f << "    file_" << i << ",\n";
    f << "    sizeof(file_" << i << ") - 1\n";
    f << "  },\n";
  }
  if (c_output) {
    f << "  {NULL, NULL, 0},\n";
    f << "};\n";
    f << "const struct iree_file_toc_t* " << absl::GetFlag(FLAGS_identifier)
      << "_create() {\n";
  } else {
    f << "  {nullptr, nullptr, 0},\n";
    f << "};\n";
    f << "const struct ::iree::iree_file_toc_t* "
      << absl::GetFlag(FLAGS_identifier) << "_create() {\n";
  }
  f << "  return &toc[0];\n";
  f << "}\n";
  if (!c_output) {
    GenerateNamespaceClose(f);
  }
  f.close();
  return f.good();
}

int main(int argc, char** argv) {
  // Parse flags.
  std::vector<char*> raw_positional_args = absl::ParseCommandLine(argc, argv);
  std::vector<std::string> input_files;
  input_files.reserve(raw_positional_args.size() - 1);
  // Skip program name.
  for (size_t i = 1, e = raw_positional_args.size(); i < e; ++i) {
    input_files.push_back(std::string(raw_positional_args[i]));
  }

  // Generate TOC files by optionally removing a prefix.
  std::vector<std::string> toc_files;
  toc_files.reserve(input_files.size());
  const std::string& strip_prefix = absl::GetFlag(FLAGS_strip_prefix);
  for (const auto& input_file : input_files) {
    std::string toc_file = input_file;
    if (!strip_prefix.empty()) {
      toc_file = std::string(absl::StripPrefix(toc_file, strip_prefix));
    }
    if (absl::GetFlag(FLAGS_flatten)) {
      std::vector<std::string> comps =
          absl::StrSplit(toc_file, absl::ByAnyChar("/\\"));
      toc_file = comps.back();
    }
    toc_files.push_back(toc_file);
  }
  // Can either generate the c or c++ output.
  if (!absl::GetFlag(FLAGS_cpp_namespace).empty() &&
      absl::GetFlag(FLAGS_c_output)) {
    std::cerr << "Can only generate either c or c++ output.\n";
    return 1;
  }
  if (!absl::GetFlag(FLAGS_output_header).empty()) {
    if (!GenerateHeader(absl::GetFlag(FLAGS_output_header), toc_files)) {
      std::cerr << "Error generating headers.\n";
      return 1;
    }
  }

  if (!absl::GetFlag(FLAGS_output_impl).empty()) {
    if (!GenerateImpl(absl::GetFlag(FLAGS_output_impl), input_files,
                      toc_files)) {
      std::cerr << "Error generating impl.\n";
      return 2;
    }
  }

  return 0;
}
