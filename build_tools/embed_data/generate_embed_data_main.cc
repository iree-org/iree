// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

static bool GenerateHeader(const std::string& identifier,
                           const std::string& header_file,
                           const std::vector<std::string>& toc_files) {
  std::ofstream f(header_file, std::ios::out | std::ios::trunc);
  if (!f) {
    fprintf(stderr, "Failed to open '%s' for write.\n", header_file.c_str());
    exit(EXIT_FAILURE);
  }

  f << "#pragma once\n";  // Pragma once isn't great but is the best we can do.
  f << "#include <stddef.h>\n";
  GenerateTocStruct(f);
  GenerateExternCOpen(f);
  f << "const iree_file_toc_t* " << identifier << "_create();\n";
  f << "static inline size_t " << identifier << "_size() {\n";
  f << "  return " << toc_files.size() << ";\n";
  f << "}\n";
  GenerateExternCClose(f);
  f.close();
  return f.good();
}

static bool SlurpFile(const std::string& file_name, std::string* contents) {
  constexpr std::streamoff kMaxSize = 100000000;
  std::ifstream f(file_name, std::ios::in | std::ios::binary);
  if (!f) {
    fprintf(stderr, "Failed to open '%s' for read.\n", file_name.c_str());
    exit(EXIT_FAILURE);
  }
  // get length of file:
  f.seekg(0, f.end);
  std::streamoff length = f.tellg();
  f.seekg(0, f.beg);
  if (!f.good()) return false;

  if (length > kMaxSize) {
    fprintf(stderr,
            "File '%s' is too large to embed into a C file (%lld bytes > %lld "
            "bytes). Consider other methods for packaging and loading on your "
            "platform, such as using traditional file I/O\n",
            file_name.c_str(), (long long)length, (long long)kMaxSize);
    return false;
  }

  size_t mem_length = static_cast<size_t>(length);
  contents->resize(mem_length);
  f.read(&(*contents)[0], mem_length);
  f.close();
  return f.good();
}

static bool GenerateImpl(const std::string& identifier,
                         const std::string& impl_file,
                         const std::vector<std::string>& input_files,
                         const std::vector<std::string>& toc_files) {
  std::ofstream f(impl_file, std::ios::out | std::ios::trunc);
  if (!f) {
    fprintf(stderr, "Failed to open '%s' for write.\n", impl_file.c_str());
    exit(EXIT_FAILURE);
  }

  f << "#include <stddef.h>\n";
  f << "#include <stdint.h>\n";
  f << R"(
#if !defined(IREE_DATA_ALIGNAS_PTR)
// Default set to 512b alignment.
#if defined(_MSC_VER)
#define IREE_DATA_ALIGNAS_PTR __declspec(align(64))
#else
#define IREE_DATA_ALIGNAS_PTR _Alignas(64)
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
    f << "    (const char*)file_" << i << ",\n";
    f << "    sizeof(file_" << i << ") - 1\n";
    f << "  },\n";
  }
  f << "  {NULL, NULL, 0},\n";
  f << "};\n";
  GenerateExternCOpen(f);
  f << "const struct iree_file_toc_t* " << identifier << "_create() {\n";
  f << "  return &toc[0];\n";
  f << "}\n";
  GenerateExternCClose(f);
  f.close();
  return f.good();
}

static void SplitArgument(const char* arg, bool& is_option, std::string& value,
                          std::string& key) {
  // Handle non-option.
  if (*arg != '-') {
    is_option = false;
    key.clear();
    value = std::string(arg);
    return;
  }

  // Eat leading hyphens.
  is_option = true;
  while (*arg == '-') arg++;

  // Parse key=value.
  key = std::string(arg);
  value.clear();
  auto eqPos = key.find('=');
  if (eqPos == std::string::npos) {
    // No '='.
    return;
  }

  // Split.
  value.append(key.begin() + eqPos + 1, key.end());
  key.resize(eqPos);
}

int main(int argc, char** argv) {
  // Parse command line options. As part of the build which needs to not depend
  // on anything, we do this the manual way vs using a flag library.
  std::vector<std::string> input_files;
  std::string identifier("resources");
  std::string output_header;
  std::string output_impl;
  std::string strip_prefix;
  bool flatten = false;

  for (size_t i = 1, e = argc; i < e; ++i) {
    const char* arg = argv[i];
    bool is_option;
    std::string value;
    std::string key;
    SplitArgument(arg, is_option, value, key);

    if (!is_option) {
      input_files.push_back(std::move(value));
      continue;
    }

    if (key == "identifier") {
      identifier = value;
    } else if (key == "output_header") {
      output_header = value;
    } else if (key == "output_impl") {
      output_impl = value;
    } else if (key == "strip_prefix") {
      strip_prefix = value;
    } else if (key == "flatten") {
      flatten = true;
    } else {
      std::cerr << "Unrecognized command line argument: " << arg << "\n";
      return 100;
    }
  }

  // Generate TOC files by optionally removing a prefix.
  std::vector<std::string> toc_files;
  toc_files.reserve(input_files.size());
  for (const auto& input_file : input_files) {
    std::string toc_file = input_file;
    if (!strip_prefix.empty()) {
      if (toc_file.find(strip_prefix) == 0) {
        toc_file = toc_file.substr(strip_prefix.size());
      }
    }
    if (flatten) {
      size_t slash_pos = toc_file.find_last_of("/\\");
      if (slash_pos != std::string::npos) {
        toc_file = toc_file.substr(slash_pos + 1);
      }
    }
    toc_files.push_back(toc_file);
  }
  if (!output_header.empty()) {
    if (!GenerateHeader(identifier, output_header, toc_files)) {
      std::cerr << "Error generating headers.\n";
      return 1;
    }
  }

  if (!output_impl.empty()) {
    if (!GenerateImpl(identifier, output_impl, input_files, toc_files)) {
      std::cerr << "Error generating impl.\n";
      return 2;
    }
  }

  return 0;
}
