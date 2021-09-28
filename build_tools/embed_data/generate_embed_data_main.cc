// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "iree/base/internal/flags.h"

IREE_FLAG(string, identifier, "resources", "name of the resources function");
IREE_FLAG(string, output_header, "", "output header file");
IREE_FLAG(string, output_impl, "", "output impl file");
IREE_FLAG(string, strip_prefix, "", "strip prefix from filenames");
IREE_FLAG(bool, flatten, false,
          "whether to flatten the directory structure (only include basename)");

static std::string CEscape(const std::string& src) {
  static const char kHexChar[] = "0123456789ABCDEF";
  std::string dest;
  bool last_hex_escape = false;  // true if last output char was \xNN.
  for (unsigned char c : src) {
    bool is_hex_escape = false;
    switch (c) {
      case '\n':
        dest.append("\\n");
        break;
      case '\r':
        dest.append("\\r");
        break;
      case '\t':
        dest.append("\\t");
        break;
      case '\"':
        dest.append("\\\"");
        break;
      case '\'':
        dest.append("\\'");
        break;
      case '\\':
        dest.append("\\\\");
        break;
      default:
        // Note that if we emit \xNN and the src character after that is a hex
        // digit then that digit must be escaped too to prevent it being
        // interpreted as part of the character code by C.
        if ((!isprint(c) || (last_hex_escape && isxdigit(c)))) {
          dest.append(
              "\\"
              "x");
          dest.push_back(kHexChar[c / 16]);
          dest.push_back(kHexChar[c % 16]);
          is_hex_escape = true;
        } else {
          dest.push_back(c);
          break;
        }
    }
    last_hex_escape = is_hex_escape;
  }
  return dest;
}

static void GenerateExternCOpen(std::ofstream& f) {
  f << "\n#if __cplusplus\n";
  f << "extern \"C\" {\n";
  f << "#endif // __cplusplus\n";
}

static void GenerateExternCClose(std::ofstream& f) {
  f << "#if __cplusplus\n";
  f << "}\n";
  f << "#endif // __cplusplus\n\n";
}

static void GenerateTocStruct(std::ofstream& f) {
  f << "#ifndef IREE_FILE_TOC\n";
  f << "#define IREE_FILE_TOC\n";
  GenerateExternCOpen(f);
  f << "typedef struct iree_file_toc_t {\n";
  f << "  const char* name;             // the file's original name\n";
  f << "  const char* data;             // beginning of the file\n";
  f << "  size_t size;                  // length of the file\n";
  f << "} iree_file_toc_t;\n";
  GenerateExternCClose(f);
  f << "#endif  // IREE_FILE_TOC\n";
}

static bool GenerateHeader(const std::string& header_file,
                           const std::vector<std::string>& toc_files) {
  std::ofstream f(header_file, std::ios::out | std::ios::trunc);
  f << "#pragma once\n";  // Pragma once isn't great but is the best we can do.
  f << "#include <stddef.h>\n";
  GenerateTocStruct(f);
  GenerateExternCOpen(f);
  f << "const iree_file_toc_t* " << FLAG_identifier << "_create();\n";
  f << "static inline size_t " << FLAG_identifier << "_size() {\n";
  f << "  return " << toc_files.size() << ";\n";
  f << "}\n";
  GenerateExternCClose(f);
  f.close();
  return f.good();
}

static bool SlurpFile(const std::string& file_name, std::string* contents) {
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

static bool GenerateImpl(const std::string& impl_file,
                         const std::vector<std::string>& input_files,
                         const std::vector<std::string>& toc_files) {
  std::ofstream f(impl_file, std::ios::out | std::ios::trunc);
  f << "#include <stddef.h>\n";
  f << "#include <stdint.h>\n";
  f << R"(
#if !defined(IREE_DATA_ALIGNAS_PTR)
#if defined(_MSC_VER)
#define IREE_DATA_ALIGNAS_PTR __declspec(align(8))
#else
#include <stdalign.h>
#define IREE_DATA_ALIGNAS_PTR alignas(alignof(void*))
#endif  // _MSC_VER
#endif  // !IREE_DATA_ALIGNAS_PTR
  )";
  GenerateTocStruct(f);
  for (size_t i = 0, e = input_files.size(); i < e; ++i) {
    f << "IREE_DATA_ALIGNAS_PTR static uint8_t const file_" << i << "[] = {\n";
    std::string contents;
    if (!SlurpFile(input_files[i], &contents)) {
      std::cerr << "Error reading file " << input_files[i] << "\n";
      return false;
    }
    size_t remaining_offset = 0;
    size_t remaining_length = contents.size();
    constexpr size_t kMaxBytesPerLine = 1024;
    while (remaining_length > 0) {
      size_t line_length = std::min(remaining_length, kMaxBytesPerLine);
      for (size_t j = 0; j < line_length; ++j) {
        char c = contents[remaining_offset + j];
        f << std::to_string((uint8_t)c) << ",";
      }
      f << "\n";
      remaining_offset += line_length;
      remaining_length -= line_length;
    }
    f << "0,\n";  // NUL termination
    f << "};\n";
  }
  f << "static const struct iree_file_toc_t toc[] = {\n";
  assert(input_files.size() == toc_files.size());
  for (size_t i = 0, e = input_files.size(); i < e; ++i) {
    f << "  {\n";
    f << "    \"" << CEscape(toc_files[i]) << "\",\n";
    f << "    file_" << i << ",\n";
    f << "    sizeof(file_" << i << ") - 1\n";
    f << "  },\n";
  }
  f << "  {NULL, NULL, 0},\n";
  f << "};\n";
  f << "const struct iree_file_toc_t* " << FLAG_identifier << "_create() {\n";
  f << "  return &toc[0];\n";
  f << "}\n";
  f.close();
  return f.good();
}

int main(int argc, char** argv) {
  // Parse flags, updating argc/argv with position arguments.
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  std::vector<std::string> input_files;
  input_files.reserve(argc - 1);
  for (size_t i = 1, e = argc; i < e; ++i) {  // Skip program name.
    input_files.push_back(std::string(argv[i]));
  }

  // Generate TOC files by optionally removing a prefix.
  std::vector<std::string> toc_files;
  toc_files.reserve(input_files.size());
  const std::string& strip_prefix = FLAG_strip_prefix;
  for (const auto& input_file : input_files) {
    std::string toc_file = input_file;
    if (!strip_prefix.empty()) {
      if (toc_file.find(strip_prefix) == 0) {
        toc_file = toc_file.substr(strip_prefix.size());
      }
    }
    if (FLAG_flatten) {
      size_t slash_pos = toc_file.find_last_of("/\\");
      if (slash_pos != std::string::npos) {
        toc_file = toc_file.substr(slash_pos + 1);
      }
    }
    toc_files.push_back(toc_file);
  }
  if (strlen(FLAG_output_header) != 0) {
    if (!GenerateHeader(FLAG_output_header, toc_files)) {
      std::cerr << "Error generating headers.\n";
      return 1;
    }
  }

  if (strlen(FLAG_output_impl) != 0) {
    if (!GenerateImpl(FLAG_output_impl, input_files, toc_files)) {
      std::cerr << "Error generating impl.\n";
      return 2;
    }
  }

  return 0;
}
