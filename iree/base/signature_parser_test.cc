// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/signature_parser.h"

#include "iree/testing/gtest.h"

namespace iree {
namespace {

class SipSignatureTest : public ::testing::Test {
 protected:
  std::string PrintInputSignature(const char* encoded) {
    SipSignatureParser parser;
    SipSignatureParser::ToStringVisitor printer;
    parser.VisitInputs(printer, encoded);
    EXPECT_FALSE(parser.GetError()) << "Parse error: " << *parser.GetError();
    return std::move(printer.s());
  }

  std::string PrintResultsSignature(const char* encoded) {
    SipSignatureParser parser;
    SipSignatureParser::ToStringVisitor printer;
    parser.VisitResults(printer, encoded);
    EXPECT_FALSE(parser.GetError()) << "Parse error: " << *parser.GetError();
    return std::move(printer.s());
  }
};

TEST(SignatureBuilderTest, TestInteger) {
  SignatureParser sp1("_5a1z10x-5991");

  // Expect 5.
  ASSERT_EQ(SignatureParser::Type::kInteger, sp1.type());
  EXPECT_EQ('_', sp1.tag());
  EXPECT_EQ(5, sp1.ival());
  EXPECT_TRUE(sp1.sval().empty());

  // Expect 1.
  ASSERT_EQ(SignatureParser::Type::kInteger, sp1.Next());
  EXPECT_EQ('a', sp1.tag());
  EXPECT_EQ(1, sp1.ival());
  EXPECT_TRUE(sp1.sval().empty());

  // Expect 10.
  ASSERT_EQ(SignatureParser::Type::kInteger, sp1.Next());
  EXPECT_EQ('z', sp1.tag());
  EXPECT_EQ(10, sp1.ival());
  EXPECT_TRUE(sp1.sval().empty());

  // Expect -5991.
  ASSERT_EQ(SignatureParser::Type::kInteger, sp1.Next());
  EXPECT_EQ('x', sp1.tag());
  EXPECT_EQ(-5991, sp1.ival());
  EXPECT_TRUE(sp1.sval().empty());

  // Expect end.
  ASSERT_EQ(SignatureParser::Type::kEnd, sp1.Next());
  ASSERT_EQ(SignatureParser::Type::kEnd, sp1.Next());
}

TEST(SignatureBuilderTest, TestSpan) {
  SignatureParser sp1("A7!foobarZ17!FOOBAR_23_FOOBAR");

  // Expect "foobar".
  ASSERT_EQ(SignatureParser::Type::kSpan, sp1.type());
  EXPECT_EQ('A', sp1.tag());
  EXPECT_EQ("foobar", sp1.sval());
  EXPECT_EQ(6, sp1.ival());  // Length.

  // Expect "FOOBAR_23_FOOBAR"
  ASSERT_EQ(SignatureParser::Type::kSpan, sp1.Next());
  EXPECT_EQ('Z', sp1.tag());
  EXPECT_EQ("FOOBAR_23_FOOBAR", sp1.sval());
  EXPECT_EQ(16, sp1.ival());  // Length.

  // Expect end.
  ASSERT_EQ(SignatureParser::Type::kEnd, sp1.Next());
  ASSERT_EQ(SignatureParser::Type::kEnd, sp1.Next());
}

TEST(SignatureBuilderTest, TestEscapedNumericSpan) {
  SignatureParser sp1("A6!12345Z4!-23");

  // Expect "foobar".
  ASSERT_EQ(SignatureParser::Type::kSpan, sp1.type());
  EXPECT_EQ('A', sp1.tag());
  EXPECT_EQ("12345", sp1.sval());
  EXPECT_EQ(5, sp1.ival());  // Length.

  // Expect "FOOBAR_23_FOOBAR"
  ASSERT_EQ(SignatureParser::Type::kSpan, sp1.Next());
  EXPECT_EQ('Z', sp1.tag());
  EXPECT_EQ("-23", sp1.sval());
  EXPECT_EQ(3, sp1.ival());  // Length.

  // Expect end.
  ASSERT_EQ(SignatureParser::Type::kEnd, sp1.Next());
  ASSERT_EQ(SignatureParser::Type::kEnd, sp1.Next());
}

TEST(SignatureBuilderTest, TestEscapedEscapeChar) {
  SignatureParser sp1("A6!!2345Z4!-23");

  // Expect "foobar".
  ASSERT_EQ(SignatureParser::Type::kSpan, sp1.type());
  EXPECT_EQ('A', sp1.tag());
  EXPECT_EQ("!2345", sp1.sval());
  EXPECT_EQ(5, sp1.ival());  // Length.

  // Expect "FOOBAR_23_FOOBAR"
  ASSERT_EQ(SignatureParser::Type::kSpan, sp1.Next());
  EXPECT_EQ('Z', sp1.tag());
  EXPECT_EQ("-23", sp1.sval());
  EXPECT_EQ(3, sp1.ival());  // Length.

  // Expect end.
  ASSERT_EQ(SignatureParser::Type::kEnd, sp1.Next());
  ASSERT_EQ(SignatureParser::Type::kEnd, sp1.Next());
}

TEST(SignatureBuilderTest, TestNested) {
  SignatureParser sp1("_5X3!_6");
  ASSERT_EQ(SignatureParser::Type::kInteger, sp1.type());
  EXPECT_EQ('_', sp1.tag());
  EXPECT_EQ(5, sp1.ival());
  ASSERT_EQ(SignatureParser::Type::kSpan, sp1.Next());
  EXPECT_EQ('X', sp1.tag());
  auto sp2 = sp1.nested();
  ASSERT_EQ(SignatureParser::Type::kEnd, sp1.Next());
  ASSERT_EQ(SignatureParser::Type::kInteger, sp2.type());
  EXPECT_EQ(6, sp2.ival());
  EXPECT_EQ('_', sp2.tag());
  ASSERT_EQ(SignatureParser::Type::kEnd, sp2.Next());
}

TEST(SignatureParserTest, Empty) {
  SignatureParser sp1("");
  EXPECT_EQ(SignatureParser::Type::kEnd, sp1.type());
  ASSERT_EQ(SignatureParser::Type::kEnd, sp1.Next());
}

TEST(SignatureParserTest, IllegalTag) {
  SignatureParser sp1("\0011 ");
  EXPECT_EQ(SignatureParser::Type::kError, sp1.type());
  ASSERT_EQ(SignatureParser::Type::kError, sp1.Next());
}

TEST(SignatureParserTest, ShortLength) {
  SignatureParser sp1("Z4abc");
  EXPECT_EQ(SignatureParser::Type::kError, sp1.type());
  ASSERT_EQ(SignatureParser::Type::kError, sp1.Next());
}

TEST(SignatureParserTest, NonNumeric) {
  SignatureParser sp1("_+12");
  EXPECT_EQ(SignatureParser::Type::kError, sp1.type());
  ASSERT_EQ(SignatureParser::Type::kError, sp1.Next());
}

TEST(SignatureParserTest, NegativeLength) {
  SignatureParser sp1("Z-3abc");
  EXPECT_EQ(SignatureParser::Type::kError, sp1.type());
  ASSERT_EQ(SignatureParser::Type::kError, sp1.Next());
}

TEST(SignatureParserTest, ZeroLengthSpan) {
  SignatureParser sp1("Z1!");
  EXPECT_EQ(SignatureParser::Type::kSpan, sp1.type());
  EXPECT_EQ(0, sp1.ival());
  EXPECT_EQ("", sp1.sval());
  EXPECT_EQ(SignatureParser::Type::kEnd, sp1.Next());
}

// -----------------------------------------------------------------------------
// Raw signatures
// -----------------------------------------------------------------------------

TEST(RawSignatureParserTest, EmptySignature) {
  RawSignatureParser p;
  auto s = p.FunctionSignatureToString("I1!R1!");
  ASSERT_TRUE(s) << *p.GetError();
  EXPECT_EQ("() -> ()", *s);
}

TEST(RawSignatureParserTest, StaticNdArrayBuffer) {
  RawSignatureParser p;
  auto s = p.FunctionSignatureToString("I15!B11!d10d128d64R15!B11!t6d32d8d64");
  ASSERT_TRUE(s) << *p.GetError();
  EXPECT_EQ("(Buffer<float32[10x128x64]>) -> (Buffer<sint32[32x8x64]>)", *s);
}

TEST(RawSignatureParserTest, DynamicNdArrayBuffer) {
  RawSignatureParser p;
  auto s = p.FunctionSignatureToString("I15!B11!d-1d128d64R15!B11!t6d-1d8d64");
  ASSERT_TRUE(s) << *p.GetError();
  EXPECT_EQ("(Buffer<float32[?x128x64]>) -> (Buffer<sint32[?x8x64]>)", *s);
}

TEST(RawSignatureParserTest, Scalar) {
  RawSignatureParser p;
  auto s = p.FunctionSignatureToString("I6!S3!t6R6!S3!t2");
  ASSERT_TRUE(s) << *p.GetError();
  EXPECT_EQ("(sint32) -> (float64)", *s);
}

TEST(RawSignatureParserTest, AllTypes) {
  RawSignatureParser p;
  auto s = p.FunctionSignatureToString(
      "I23!O1!B11!d-1d128d64S3!t6R17!B13!t11d32d-1d64");
  ASSERT_TRUE(s) << *p.GetError();
  EXPECT_EQ(
      "(RefObject<?>, Buffer<float32[?x128x64]>, sint32) -> "
      "(Buffer<uint64[32x?x64]>)",
      *s);
}

// -----------------------------------------------------------------------------
// Sip signatures
// -----------------------------------------------------------------------------

TEST_F(SipSignatureTest, NoInputsResults) {
  const char kEncodedSignature[] = "I1!R1!";
  const char kExpectedInputs[] = R"()";
  const char kExpectedResults[] = R"()";

  auto inputs_string = PrintInputSignature(kEncodedSignature);
  EXPECT_EQ(kExpectedInputs, inputs_string) << inputs_string;

  auto results_string = PrintResultsSignature(kEncodedSignature);
  EXPECT_EQ(kExpectedResults, results_string) << results_string;
}

TEST_F(SipSignatureTest, SequentialInputSingleResult) {
  const char kEncodedSignature[] = "I12!S9!k0_0k1_1R3!_0";
  const char kExpectedInputs[] = R"(:[
  0=raw(0),
  1=raw(1),
],
)";
  const char kExpectedResults[] = R"(=raw(0),
)";

  auto inputs_string = PrintInputSignature(kEncodedSignature);
  EXPECT_EQ(kExpectedInputs, inputs_string) << inputs_string;

  auto results_string = PrintResultsSignature(kEncodedSignature);
  EXPECT_EQ(kExpectedResults, results_string) << results_string;
}

TEST_F(SipSignatureTest, NestedInputMultiResult) {
  const char kEncodedSignature[] =
      "I31!S27!k0D17!K4!bar_1K4!foo_0k1_2R12!S9!k0_0k1_1";
  const char kExpectedInputs[] = R"(:[
  0:{
    bar=raw(1),
    foo=raw(0),
  },
  1=raw(2),
],
)";
  const char kExpectedResults[] = R"(:[
  0=raw(0),
  1=raw(1),
],
)";

  auto inputs_string = PrintInputSignature(kEncodedSignature);
  EXPECT_EQ(kExpectedInputs, inputs_string) << inputs_string;

  auto results_string = PrintResultsSignature(kEncodedSignature);
  EXPECT_EQ(kExpectedResults, results_string) << results_string;
}

}  // namespace
}  // namespace iree
