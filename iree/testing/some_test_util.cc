#include "iree/testing/some_test_util.h"

#include "iree/testing/gtest.h"

namespace iree {
namespace testing {

void ExpectTrue(int32_t val) {
  EXPECT_TRUE(val);
}
}
}
