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

#ifndef IREE_TASK_LIST_H_
#define IREE_TASK_LIST_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/task/task.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// iree_atomic_task_slist_t, an atomic approximately LIFO singly-linked list.
// iree_task_list_t should be preferred when working with
// uncontended/thread-local lists as it has no overhead, while the
// iree_atomic_task_slist_t should be used when multiple threads may need to
// share lists of tasks (free lists, mailboxes, etc).
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_atomic_task, iree_task_t,
                                offsetof(iree_task_t, next_task));

// Discards a task list; should be used for failure cleanup during list
// construction to ensure intrusive pointers are reset.
void iree_atomic_task_slist_discard(iree_atomic_task_slist_t* slist);

// A singly-linked list of tasks using the embedded task next_task pointer.
//
// Thread-compatible; designed to be used from a single thread manipulating a
// list for passing to an API that accepts lists.
typedef struct iree_task_list_s {
  iree_task_t* head;
  iree_task_t* tail;
} iree_task_list_t;

// Initializes an empty task list.
void iree_task_list_initialize(iree_task_list_t* out_list);

// Moves |list| into |out_list|, leaving |list| empty.
void iree_task_list_move(iree_task_list_t* list, iree_task_list_t* out_list);

// Discards a task list; should be used for failure cleanup during list
// construction to ensure intrusive pointers are reset. List is immediately
// reusable as if it had been initialized.
void iree_task_list_discard(iree_task_list_t* list);

// Returns true if the list is empty.
bool iree_task_list_is_empty(const iree_task_list_t* list);

// Counts the total number of tasks in the list.
// WARNING: this requires an O(n) walk of the entire list; use this only for
// debugging or when the list is known to be small and hot in cache.
iree_host_size_t iree_task_list_calculate_size(const iree_task_list_t* list);

// Returns the first task in the list, if any.
iree_task_t* iree_task_list_front(iree_task_list_t* list);

// Returns the last task in the list, if any.
iree_task_t* iree_task_list_back(iree_task_list_t* list);

// Pushes a task onto the back of the task list. The task list takes ownership
// of |task|.
void iree_task_list_push_back(iree_task_list_t* list, iree_task_t* task);

// Pushes a task onto the front of the task list. The task list takes ownership
// of |task|.
void iree_task_list_push_front(iree_task_list_t* list, iree_task_t* task);

// Pops a task from the front of the task list or returns NULL if the list is
// empty. Caller takes ownership of the returned task.
iree_task_t* iree_task_list_pop_front(iree_task_list_t* list);

// Erases |task| from the list.
// |prev_task| must point to the task immediately prior to |task| in the list
// or NULL if the task was at the head.
void iree_task_list_erase(iree_task_list_t* list, iree_task_t* prev_task,
                          iree_task_t* task);

// Prepends |prefix| onto the beginning of |list|. |prefix| will be reset.
void iree_task_list_prepend(iree_task_list_t* list, iree_task_list_t* prefix);

// Appends |suffix| onto the end of |list|. |suffix| will be reset.
void iree_task_list_append(iree_task_list_t* list, iree_task_list_t* suffix);

// Flushes the given |slist| and appends all tasks to the list in FIFO order.
void iree_task_list_append_from_fifo_slist(iree_task_list_t* list,
                                           iree_atomic_task_slist_t* slist);

// Reverses the list in-place.
// Requires a full O(n) traversal.
void iree_task_list_reverse(iree_task_list_t* list);

// Splits |head_list| in half (up to |max_tasks|) and retains the first half
// in |head_list| and the second half in |tail_list|.
void iree_task_list_split(iree_task_list_t* head_list,
                          iree_host_size_t max_tasks,
                          iree_task_list_t* out_tail_list);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_LIST_H_
