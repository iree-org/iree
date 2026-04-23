// clang-format off

// Test with default configuration.
// RUN: clang-tidy \
// RUN:   %s \
// RUN:   --load $IREE_BINARY_DIR/lib/IREEClangTidyModule.so \
// RUN:   --checks="-*,iree-test" \
// RUN:   -p $IREE_BINARY_DIR | \
// RUN:   FileCheck %s --check-prefix=DEFAULT
// DEFAULT: warning: found test function 'test_iree_check'
// DEFAULT-SAME: [TestBool=false]

// Test with primitive values (numbers, strings, etc).
// RUN: clang-tidy \
// RUN:   %s \
// RUN:   --load $IREE_BINARY_DIR/lib/IREEClangTidyModule.so \
// RUN:   --checks="-*,iree-test" \
// RUN:   --config="{CheckOptions: {iree-test.TestString: 'hello', iree-test.TestNumber: '42', iree-test.TestBool: true}}" \
// RUN:   -p $IREE_BINARY_DIR | \
// RUN:   FileCheck %s --check-prefix=PRIMITIVES
// PRIMITIVES: warning: found test function 'test_iree_check'
// PRIMITIVES-SAME: [TestString=hello] [TestNumber=42] [TestBool=true]

// Test with list configuration.
// RUN: clang-tidy \
// RUN:   %s \
// RUN:   --load $IREE_BINARY_DIR/lib/IREEClangTidyModule.so \
// RUN:   --checks="-*,iree-test" \
// RUN:   --config="{CheckOptions: {iree-test.TestList: 'foo,bar,baz'}}" \
// RUN:   -p $IREE_BINARY_DIR | \
// RUN:   FileCheck %s --check-prefix=LIST
// LIST: warning: found test function 'test_iree_check'
// LIST-SAME: [TestBool=false] [TestList=foo,bar,baz]

// Test with map configuration.
// RUN: clang-tidy \
// RUN:   %s \
// RUN:   --load $IREE_BINARY_DIR/lib/IREEClangTidyModule.so \
// RUN:   --checks="-*,iree-test" \
// RUN:   --config="{CheckOptions: {iree-test.TestMap: 'key1:value1,key2:value2'}}" \
// RUN:   -p $IREE_BINARY_DIR | \
// RUN:   FileCheck %s --check-prefix=MAP
// MAP: warning: found test function 'test_iree_check'
// MAP-SAME: [TestBool=false] [TestMap=key1:value1,key2:value2]

// Test with remark severity.
// RUN: clang-tidy \
// RUN:   %s \
// RUN:   --load $IREE_BINARY_DIR/lib/IREEClangTidyModule.so \
// RUN:   --checks="-*,iree-test" \
// RUN:   --config="{CheckOptions: {iree-test.Severity: 'remark'}}" \
// RUN:   -p $IREE_BINARY_DIR | \
// RUN:   FileCheck %s --check-prefix=REMARK
// REMARK: remark: found test function 'test_iree_check'
// REMARK-SAME: [TestBool=false]

// Test with boolean true.
// RUN: clang-tidy \
// RUN:   %s \
// RUN:   --load $IREE_BINARY_DIR/lib/IREEClangTidyModule.so \
// RUN:   --checks="-*,iree-test" \
// RUN:   --config="{CheckOptions: {iree-test.TestBool: 'true'}}" \
// RUN:   -p $IREE_BINARY_DIR | \
// RUN:   FileCheck %s --check-prefix=BOOL
// BOOL: warning: found test function 'test_iree_check'
// BOOL-SAME: [TestBool=true]

// This should trigger the test check with various configurations.
void test_iree_check(void) {
  // Function that will be detected.
}
