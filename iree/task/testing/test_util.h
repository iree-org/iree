// Copyright 2020 Google LLC
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

// NOTE: the best kind of synchronization is no synchronization; always try to
// design your algorithm so that you don't need anything from this file :)
// See https://travisdowns.github.io/blog/2020/07/06/concurrency-costs.html

#ifndef IREE_TASK_TESTING_TEST_UTIL_H_
#define IREE_TASK_TESTING_TEST_UTIL_H_

#include <memory>

#include "iree/task/list.h"
#include "iree/task/pool.h"
#include "iree/task/scope.h"
#include "iree/testing/status_matchers.h"

using TaskPoolPtr =
    std::unique_ptr<iree_task_pool_t, void (*)(iree_task_pool_t*)>;
static inline TaskPoolPtr AllocateNopPool() {
  iree_task_pool_t* pool = new iree_task_pool_t();
  IREE_CHECK_OK(iree_task_pool_initialize(iree_allocator_system(),
                                          sizeof(iree_task_nop_t), 1024, pool));
  return {pool, [](iree_task_pool_t* pool) {
            iree_task_pool_deinitialize(pool);
            delete pool;
          }};
}

using TaskScopePtr =
    std::unique_ptr<iree_task_scope_t, void (*)(iree_task_scope_t*)>;
static inline TaskScopePtr AllocateScope(const char* name) {
  iree_task_scope_t* scope = new iree_task_scope_t();
  iree_task_scope_initialize(iree_make_cstring_view(name), scope);
  return {scope, [](iree_task_scope_t* scope) {
            iree_task_scope_deinitialize(scope);
            delete scope;
          }};
}

static inline iree_task_t* AcquireNopTask(TaskPoolPtr& pool,
                                          TaskScopePtr& scope, uint16_t value) {
  iree_task_t* task = NULL;
  IREE_CHECK_OK(iree_task_pool_acquire(pool.get(), &task));
  iree_task_initialize(IREE_TASK_TYPE_NOP, scope.get(), task);
  task->flags = value;
  return task;
}

static inline bool CheckListOrderFIFO(iree_task_list_t* list) {
  iree_task_t* p = list->head;
  if (!p) return true;
  uint16_t value = p->flags;
  p = p->next_task;
  while (p) {
    if (p->flags <= value) return false;
    p = p->next_task;
  }
  return true;
}

static inline bool CheckListOrderLIFO(iree_task_list_t* list) {
  iree_task_t* p = list->head;
  if (!p) return true;
  uint16_t value = p->flags;
  p = p->next_task;
  while (p) {
    if (p->flags >= value) return false;
    p = p->next_task;
  }
  return true;
}

#endif  // IREE_TASK_TESTING_TEST_UTIL_H_
