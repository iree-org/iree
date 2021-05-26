// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/signature_parser.h"

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

namespace iree {

// -----------------------------------------------------------------------------
// AbiConstants
// -----------------------------------------------------------------------------

const std::array<size_t, 12> AbiConstants::kScalarTypeSize = {
    4,  // kIeeeFloat32 = 0,
    2,  // kIeeeFloat16 = 1,
    8,  // kIeeeFloat64 = 2,
    2,  // kGoogleBfloat16 = 3,
    1,  // kSint8 = 4,
    2,  // kSint16 = 5,
    4,  // kSint32 = 6,
    8,  // kSint64 = 7,
    1,  // kUint8 = 8,
    2,  // kUint16 = 9,
    4,  // kUint32 = 10,
    8,  // kUint64 = 11,
};

const std::array<const char*, 12> AbiConstants::kScalarTypeNames = {
    "float32", "float16", "float64", "bfloat16", "sint8",  "sint16",
    "sint32",  "sint64",  "uint8",   "uint16",   "uint32", "uint64",
};

// -----------------------------------------------------------------------------
// SignatureParser
// -----------------------------------------------------------------------------

SignatureParser::Type SignatureParser::Next() {
  next_type_ = Type::kError;
  next_tag_ = 0;
  next_ival_ = 0;
  next_sval_ = absl::string_view();
  if (cursor_ == encoded_.end()) {
    next_type_ = Type::kEnd;
    return next_type_;
  }

  next_tag_ = *cursor_;
  absl::string_view::const_iterator ival_begin = cursor_ + 1;
  absl::string_view::const_iterator ival_end = ival_begin;
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
  if (!absl::SimpleAtoi(
          absl::string_view(&(*ival_begin), ival_end - ival_begin),
          &next_ival_)) {
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
    absl::string_view::const_iterator sval_begin = ival_end;
    absl::string_view::const_iterator sval_end = sval_begin + next_ival_;
    if (sval_end > encoded_.end()) return next_type_;  // Underrun.

    // Remove escape char if escaped.
    if (next_ival_ == 0 || *sval_begin != '!') {
      next_type_ = Type::kError;
      return next_type_;
    }
    next_ival_ -= 1;
    ++sval_begin;

    next_sval_ = absl::string_view(&(*sval_begin), sval_end - sval_begin);
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
      absl::StrAppend(&s, "Buffer<", scalar_type_name, "[");
      for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) s.push_back('x');
        if (dims[i] >= 0) {
          absl::StrAppend(&s, dims[i]);
        } else {
          s.push_back('?');
        }
      }
      absl::StrAppend(&s, "]>");
      break;
    }
    case Type::kRefObject: {
      absl::StrAppend(&s, "RefObject<?>");
      break;
    }
    case Type::kScalar: {
      const char* type_name = "!BADTYPE!";
      unsigned type_u = static_cast<unsigned>(scalar.type);
      if (type_u >= 0 && type_u <= AbiConstants::kScalarTypeNames.size()) {
        type_name =
            AbiConstants::kScalarTypeNames[static_cast<unsigned>(type_u)];
      }
      absl::StrAppend(&s, type_name);
      break;
    }
    default:
      absl::StrAppend(&s, "!UNKNOWN!");
  }
}

absl::optional<std::string> RawSignatureParser::FunctionSignatureToString(
    absl::string_view signature) {
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
    return absl::nullopt;
  }
}

// -----------------------------------------------------------------------------
// SipSignatureParser
// -----------------------------------------------------------------------------

void SipSignatureParser::ToStringVisitor::IntegerKey(SipSignatureParser& p,
                                                     int k) {
  absl::StrAppend(&s_, indent_, k);
}

void SipSignatureParser::ToStringVisitor::StringKey(SipSignatureParser& p,
                                                    absl::string_view k) {
  absl::StrAppend(&s_, indent_, k);
}

void SipSignatureParser::ToStringVisitor::OpenStruct(SipSignatureParser& p,
                                                     StructType struct_type) {
  absl::StrAppend(&indent_, "  ");
  switch (struct_type) {
    case StructType::kDict:
      close_char_.push_back('}');
      absl::StrAppend(&s_, ":{");
      break;
    case StructType::kSequence:
      close_char_.push_back(']');
      absl::StrAppend(&s_, ":[");
      break;
    default:
      close_char_.push_back('?');
      absl::StrAppend(&s_, ":?");
  }
  absl::StrAppend(&s_, "\n");
}

void SipSignatureParser::ToStringVisitor::CloseStruct(SipSignatureParser& p) {
  if (indent_.size() >= 2) {
    indent_.resize(indent_.size() - 2);
  }
  absl::StrAppend(&s_, indent_);
  s_.push_back(close_char_.back());
  close_char_.pop_back();
  absl::StrAppend(&s_, ",\n");
}

void SipSignatureParser::ToStringVisitor::MapToRawSignatureIndex(
    SipSignatureParser& p, int index) {
  absl::StrAppend(&s_, "=raw(", index, "),\n");
}

}  // namespace iree
