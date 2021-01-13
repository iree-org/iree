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

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "task",
    srcs = [
        "executor.c",
        "executor_impl.h",
        "list.c",
        "pool.c",
        "post_batch.c",
        "post_batch.h",
        "queue.c",
        "scope.c",
        "submission.c",
        "task.c",
        "task_impl.h",
        "topology.c",
        "worker.c",
        "worker.h",
    ],
    hdrs = [
        "affinity_set.h",
        "executor.h",
        "list.h",
        "pool.h",
        "queue.h",
        "scope.h",
        "submission.h",
        "task.h",
        "topology.h",
        "tuning.h",
    ],
    deps = [
        "//iree/base:api",
        "//iree/base:atomic_slist",
        "//iree/base:core_headers",
        "//iree/base:synchronization",
        "//iree/base:threading",
        "//iree/base:tracing",
        "//iree/base:wait_handle",
        "@cpuinfo",
    ],
)

cc_test(
    name = "executor_test",
    srcs = ["executor_test.cc"],
    deps = [
        ":task",
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/task/testing:test_util",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "list_test",
    srcs = ["list_test.cc"],
    deps = [
        ":task",
        "//iree/base:api",
        "//iree/task/testing:test_util",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "pool_test",
    srcs = ["pool_test.cc"],
    deps = [
        ":task",
        "//iree/base:api",
        "//iree/task/testing:test_util",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "queue_test",
    srcs = ["queue_test.cc"],
    deps = [
        ":task",
        "//iree/base:api",
        "//iree/task/testing:test_util",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "scope_test",
    srcs = ["scope_test.cc"],
    deps = [
        ":task",
        "//iree/base:api",
        "//iree/task/testing:test_util",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "topology_test",
    srcs = ["topology_test.cc"],
    deps = [
        ":task",
        "//iree/base:api",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
