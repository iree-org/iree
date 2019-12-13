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

#include "iree/base/signature_mangle.h"

#include "iree/testing/gtest.h"

namespace iree {
namespace {

class SipSignatureTest : public ::testing::Test {
 protected:
  std::string PrintInputSignature(absl::optional<SignatureBuilder> signature) {
    EXPECT_TRUE(signature);

    SipSignatureParser parser;
    SipSignatureParser::ToStringVisitor printer;
    parser.VisitInputs(printer, signature->encoded());
    EXPECT_FALSE(parser.GetError()) << "Parse error: " << *parser.GetError();
    return std::move(printer.s());
  }

  std::string PrintResultsSignature(
      absl::optional<SignatureBuilder> signature) {
    EXPECT_TRUE(signature);

    SipSignatureParser parser;
    SipSignatureParser::ToStringVisitor printer;
    parser.VisitResults(printer, signature->encoded());
    EXPECT_FALSE(parser.GetError()) << "Parse error: " << *parser.GetError();
    return std::move(printer.s());
  }
};

TEST(SignatureBuilderTest, TestInteger) {
  SignatureBuilder sb1;
  sb1.Integer(5).Integer(1, 'a').Integer(10, 'z').Integer(-5991, 'x');
  EXPECT_EQ("_5a1z10x-5991", sb1.encoded());

  SignatureParser sp1(sb1.encoded());

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
  SignatureBuilder sb1;
  sb1.Span("foobar", 'A').Span("FOOBAR_23_FOOBAR", 'Z');
  EXPECT_EQ("A7!foobarZ17!FOOBAR_23_FOOBAR", sb1.encoded());

  SignatureParser sp1(sb1.encoded());

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
  SignatureBuilder sb1;
  sb1.Span("12345", 'A').Span("-23", 'Z');
  EXPECT_EQ("A6!12345Z4!-23", sb1.encoded());

  SignatureParser sp1(sb1.encoded());

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
  SignatureBuilder sb1;
  sb1.Span("!2345", 'A').Span("-23", 'Z');
  EXPECT_EQ("A6!!2345Z4!-23", sb1.encoded());

  SignatureParser sp1(sb1.encoded());

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
  SignatureBuilder sb1;
  sb1.Integer(5);
  SignatureBuilder().Integer(6).AppendTo(sb1, 'X');
  EXPECT_EQ("_5X3!_6", sb1.encoded());

  SignatureParser sp1(sb1.encoded());
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

TEST_F(SipSignatureTest, NoInputsResults) {
  const char kExpectedInputs[] = R"()";
  const char kExpectedResults[] = R"()";

  SipSignatureMangler inputs;
  SipSignatureMangler results;

  auto signature = SipSignatureMangler::ToFunctionSignature(inputs, results);
  LOG(INFO) << "SIG: " << signature->encoded();
  EXPECT_EQ("I1!R1!", signature->encoded());

  auto inputs_string = PrintInputSignature(signature);
  EXPECT_EQ(kExpectedInputs, inputs_string) << inputs_string;

  auto results_string = PrintResultsSignature(signature);
  EXPECT_EQ(kExpectedResults, results_string) << results_string;
}

TEST_F(SipSignatureTest, SequentialInputSingleResult) {
  const char kExpectedInputs[] = R"(:[
  0=raw(0),
  1=raw(1),
],
)";
  const char kExpectedResults[] = R"(=raw(0),
)";

  SipSignatureMangler inputs;
  inputs.SetRawSignatureIndex(0, {{0}});
  inputs.SetRawSignatureIndex(1, {{1}});

  SipSignatureMangler results;
  results.SetRawSignatureIndex(0, {});

  auto signature = SipSignatureMangler::ToFunctionSignature(inputs, results);
  LOG(INFO) << "SIG: " << signature->encoded();
  auto inputs_string = PrintInputSignature(signature);
  EXPECT_EQ(kExpectedInputs, inputs_string) << inputs_string;

  auto results_string = PrintResultsSignature(signature);
  EXPECT_EQ(kExpectedResults, results_string) << results_string;
}

TEST_F(SipSignatureTest, NestedInputMultiResult) {
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

  SipSignatureMangler inputs;
  inputs.SetRawSignatureIndex(0, {{0, "foo"}});
  inputs.SetRawSignatureIndex(1, {{0, "bar"}});
  inputs.SetRawSignatureIndex(2, {{1}});

  SipSignatureMangler results;
  results.SetRawSignatureIndex(0, {{0}});
  results.SetRawSignatureIndex(1, {{1}});

  auto signature = SipSignatureMangler::ToFunctionSignature(inputs, results);
  LOG(INFO) << "SIG: " << signature->encoded();
  auto inputs_string = PrintInputSignature(signature);
  EXPECT_EQ(kExpectedInputs, inputs_string) << inputs_string;

  auto results_string = PrintResultsSignature(signature);
  EXPECT_EQ(kExpectedResults, results_string) << results_string;
}

}  // namespace
}  // namespace iree
