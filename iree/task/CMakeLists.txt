# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

iree_add_all_subdirs()

iree_cc_library(
  NAME
    task
  HDRS
    "affinity_set.h"
    "executor.h"
    "list.h"
    "pool.h"
    "queue.h"
    "scope.h"
    "submission.h"
    "task.h"
    "topology.h"
    "tuning.h"
  SRCS
    "executor.c"
    "executor_impl.h"
    "list.c"
    "pool.c"
    "post_batch.c"
    "post_batch.h"
    "queue.c"
    "scope.c"
    "submission.c"
    "task.c"
    "task_impl.h"
    "topology.c"
    "worker.c"
    "worker.h"
  DEPS
    cpuinfo
    iree::base::api
    iree::base::core_headers
    iree::base::internal::atomic_slist
    iree::base::internal::prng
    iree::base::internal::wait_handle
    iree::base::synchronization
    iree::base::threading
    iree::base::tracing
  PUBLIC
)

iree_cc_test(
  NAME
    executor_test
  SRCS
    "executor_test.cc"
  DEPS
    ::task
    iree::base::api
    iree::base::core_headers
    iree::base::internal::prng
    iree::task::testing::test_util
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    list_test
  SRCS
    "list_test.cc"
  DEPS
    ::task
    iree::base::api
    iree::task::testing::test_util
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    pool_test
  SRCS
    "pool_test.cc"
  DEPS
    ::task
    iree::base::api
    iree::task::testing::test_util
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    queue_test
  SRCS
    "queue_test.cc"
  DEPS
    ::task
    iree::base::api
    iree::task::testing::test_util
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    scope_test
  SRCS
    "scope_test.cc"
    "task_impl.h"
  DEPS
    ::task
    iree::base::api
    iree::task::testing::test_util
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    task_tests
  SRCS
    "task_test_barrier.cc"
    "task_test_call.cc"
    "task_test_dispatch.cc"
    "task_test_fence.cc"
    "task_test_nop.cc"
    "task_test_wait.cc"
  DEPS
    ::task
    iree::base::api
    iree::task::testing::task_test
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    topology_test
  SRCS
    "topology_test.cc"
  DEPS
    ::task
    cpuinfo
    iree::base::api
    iree::testing::gtest
    iree::testing::gtest_main
)
