// RUN: iree-opt %s | FileCheck %s

// Test case 1: Single quotes in comments (common in English text).
// CHECK-LABEL: @single_quotes_in_comments
func.func @single_quotes_in_comments() {
  // This function doesn't use single quotes in attributes.
  // They're only valid in comments like "can't" or "won't".
  return
}

// -----

// Test case 2: Double quotes in string literals.
// CHECK-LABEL: @double_quotes
util.global private @str = "string with \"escaped\" quotes" : !util.buffer
func.func @double_quotes() {
  return
}

// -----

// Test case 3: Backticks in comments (common in markdown/code examples).
// CHECK-LABEL: @backticks_in_comments
func.func @backticks_in_comments() {
  // Example code: `some_function(arg1, arg2)`
  // Template literal: `value is ${expr}`
  return
}

// -----

// Test case 4: Dollar signs in comments (common in cost analysis, bash vars).
// CHECK-LABEL: @dollar_signs
func.func @dollar_signs() {
  // Cost analysis: $100 per operation
  // Environment: $HOME, $PATH
  // Math: cost = $O(n^2)$
  return
}

// -----

// Test case 5: Unicode characters in comments and names.
// CHECK-LABEL: @unicode_test
func.func @unicode_test() {
  // Japanese: 日本語テスト
  // Chinese: 测试
  // Emoji: 🔧 🚀
  // Math: ∑ ∫ ∂
  return
}
