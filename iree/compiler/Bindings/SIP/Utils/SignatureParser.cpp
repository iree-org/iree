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

#include "iree/compiler/Bindings/SIP/Utils/SignatureParser.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace SIP {

// -----------------------------------------------------------------------------
// SignatureParser
// -----------------------------------------------------------------------------

SignatureParser::Type SignatureParser::Next() {
  next_type_ = Type::kError;
  next_tag_ = 0;
  next_ival_ = 0;
  next_sval_ = StringRef();
  if (cursor_ == encoded_.end()) {
    next_type_ = Type::kEnd;
    return next_type_;
  }

  next_tag_ = *cursor_;
  StringRef::const_iterator ival_begin = cursor_ + 1;
  StringRef::const_iterator ival_end = ival_begin;
  while (ival_end != encoded_.end() &&
         ((*ival_end >= '0' && *ival_end <= '9') ||
          (*ival_end == '-' && ival_end == ival_begin))) {
    ++ival_end;
  }

  // No numeric value.
  if (ival_end == ival_begin) {
    return next_type_;
  }

  // Parse ival.
  if (StringRef(&(*ival_begin), ival_end - ival_begin)
          .consumeInteger(10, next_ival_)) {
    // Should not be possible.
    return next_type_;
  }

  // For integer components ('_', 'a'..'z'), that is all.
  if (next_tag_ == '_' || (next_tag_ >= 'a' && next_tag_ <= 'z')) {
    next_type_ = Type::kInteger;
    cursor_ = ival_end;
    return next_type_;
  }

  // For string components ('A'..'Z'), extract the string.
  if (next_tag_ >= 'A' && next_tag_ <= 'Z') {
    if (next_ival_ < 0) return next_type_;  // Negative size error.
    StringRef::const_iterator sval_begin = ival_end;
    StringRef::const_iterator sval_end = sval_begin + next_ival_;
    if (sval_end > encoded_.end()) return next_type_;  // Underrun.

    // Remove escape char if escaped.
    if (next_ival_ == 0 || *sval_begin != '!') {
      next_type_ = Type::kError;
      return next_type_;
    }
    next_ival_ -= 1;
    ++sval_begin;

    next_sval_ = StringRef(&(*sval_begin), sval_end - sval_begin);
    cursor_ = sval_end;
    next_type_ = Type::kSpan;
    return next_type_;
  }

  // Otherwise, error.
  return next_type_;
}

bool SignatureParser::SeekTag(char tag) {
  while (next_tag_ != tag && next_type_ != Type::kEnd) {
    Next();
  }
  return next_type_ != Type::kEnd;
}

// -----------------------------------------------------------------------------
// RawSignatureParser
// -----------------------------------------------------------------------------

void RawSignatureParser::Description::ToString(std::string& s) const {
  switch (type) {
    case Type::kBuffer: {
      const char* scalar_type_name = "!BADTYPE!";
      unsigned scalar_type_u = static_cast<unsigned>(buffer.scalar_type);
      if (scalar_type_u >= 0 &&
          scalar_type_u <= AbiConstants::kScalarTypeNames.size()) {
        scalar_type_name = AbiConstants::kScalarTypeNames[static_cast<unsigned>(
            scalar_type_u)];
      }
      s.append("Buffer<");
      s.append(scalar_type_name);
      s.append("[");
      for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) s.push_back('x');
        if (dims[i] >= 0) {
          s.append(std::to_string(dims[i]));
        } else {
          s.push_back('?');
        }
      }
      s.append("]>");
      break;
    }
    case Type::kRefObject: {
      s.append("RefObject<?>");
      break;
    }
    case Type::kScalar: {
      const char* type_name = "!BADTYPE!";
      unsigned type_u = static_cast<unsigned>(scalar.type);
      if (type_u >= 0 && type_u <= AbiConstants::kScalarTypeNames.size()) {
        type_name =
            AbiConstants::kScalarTypeNames[static_cast<unsigned>(type_u)];
      }
      s.append(type_name);
      break;
    }
    default:
      s.append("!UNKNOWN!");
  }
}

llvm::Optional<std::string> RawSignatureParser::FunctionSignatureToString(
    StringRef signature) {
  std::string s;

  bool print_sep = false;
  auto visitor = [&print_sep, &s](const Description& d) {
    if (print_sep) {
      s.append(", ");
    }
    d.ToString(s);
    print_sep = true;
  };
  s.push_back('(');
  VisitInputs(signature, visitor);
  s.append(") -> (");
  print_sep = false;
  VisitResults(signature, visitor);
  s.push_back(')');

  if (!GetError()) {
    return s;
  } else {
    return llvm::None;
  }
}

// -----------------------------------------------------------------------------
// SipSignatureParser
// -----------------------------------------------------------------------------

void SipSignatureParser::ToStringVisitor::IntegerKey(SipSignatureParser& p,
                                                     int k) {
  s_.append(indent_);
  s_.append(std::to_string(k));
}

void SipSignatureParser::ToStringVisitor::StringKey(SipSignatureParser& p,
                                                    StringRef k) {
  s_.append(indent_);
  s_.append(k.data(), k.size());
}

void SipSignatureParser::ToStringVisitor::OpenStruct(SipSignatureParser& p,
                                                     StructType struct_type) {
  indent_.append("  ");
  switch (struct_type) {
    case StructType::kDict:
      close_char_.push_back('}');
      s_.append(":{");
      break;
    case StructType::kSequence:
      close_char_.push_back(']');
      s_.append(":[");
      break;
    default:
      close_char_.push_back('?');
      s_.append(":?");
  }
  s_.append("\n");
}

void SipSignatureParser::ToStringVisitor::CloseStruct(SipSignatureParser& p) {
  if (indent_.size() >= 2) {
    indent_.resize(indent_.size() - 2);
  }
  s_.append(indent_);
  s_.push_back(close_char_.back());
  close_char_.pop_back();
  s_.append(",\n");
}

void SipSignatureParser::ToStringVisitor::MapToRawSignatureIndex(
    SipSignatureParser& p, int index) {
  s_.append("=raw(");
  s_.append(std::to_string(index));
  s_.append("),\n");
}

}  // namespace SIP
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
