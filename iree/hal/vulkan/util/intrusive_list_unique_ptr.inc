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

// IWYU pragma: private, include "iree/hal/vulkan/util/intrusive_list.h"

#ifndef IREE_HAL_VULKAN_UTIL_INTRUSIVE_LIST_UNIQUE_PTR_H_
#define IREE_HAL_VULKAN_UTIL_INTRUSIVE_LIST_UNIQUE_PTR_H_

#include <cstddef>
#include <memory>

#include "iree/base/logging.h"
#include "iree/hal/vulkan/util/intrusive_list.h"

namespace iree {

// Specialized IntrusiveListBase for std::unique_ptr types.
// This makes the list methods accept std::unique_ptrs and contains a special
// take() method that takes ownership of a list item.
template <typename T, size_t kOffset>
class IntrusiveListUniquePtrBase
    : private IntrusiveListBase<T, IntrusiveListIterator<T, kOffset, true>,
                                IntrusiveListIterator<T, kOffset, false>,
                                kOffset> {
 public:
  using IteratorT = IntrusiveListIterator<T, kOffset, true>;
  using ReverseIteratorT = IntrusiveListIterator<T, kOffset, false>;
  using base_list = IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>;
  using self_type = IntrusiveListUniquePtrBase<T, kOffset>;

  IntrusiveListUniquePtrBase() = default;

  using base_list::empty;
  using base_list::size;

  using base_list::contains;

  inline void merge_from(self_type* other_list) {
    return base_list::merge_from(static_cast<base_list*>(other_list));
  }

  using base_list::clear;

  using base_list::begin;
  using base_list::end;
  using base_list::rbegin;
  using base_list::rend;

  using base_list::next;

  using base_list::previous;

  using base_list::front;

  void push_front(std::unique_ptr<T> value) {
    base_list::push_front(value.release());
  }

  using base_list::pop_front;

  using base_list::back;

  void push_back(std::unique_ptr<T> value) {
    base_list::push_back(value.release());
  }

  using base_list::pop_back;

  void insert(const IteratorT& it, std::unique_ptr<T> value) {
    base_list::insert(it, value.release());
  }

  using base_list::erase;

  // Removes an item from the list at the given iterator and transfers ownership
  // to the caller.
  // Performance: O(1)
  std::unique_ptr<T> take(IteratorT& it) {  // NOLINT(runtime/references)
    return take(*it);
  }

  // Removes an item from the list and transfers ownership to the caller.
  // Performance: O(1)
  std::unique_ptr<T> take(T* value) {
    if (!value) {
      return {nullptr};
    }
    auto* link = impl::TToLink<T, kOffset>(value);
    if (link->prev) {
      IREE_DCHECK_NE(link, head_);
      link->prev->next = link->next;
    } else {
      IREE_DCHECK_EQ(link, head_);
      head_ = link->next;
    }
    if (link->next) {
      IREE_DCHECK_NE(link, tail_);
      link->next->prev = link->prev;
    } else {
      IREE_DCHECK_EQ(link, tail_);
      tail_ = link->prev;
    }
    link->next = link->prev = nullptr;
    --count_;
    base_list::OnRemove(value);
    base_list::CheckCorrectness();
    return std::unique_ptr<T>(value);
  }

  void replace(T* old_value, std::unique_ptr<T> new_value) {
    base_list::replace(old_value, new_value.release());
  }

  using base_list::sort;

 private:
  void OnDeallocate(T* value) override { delete value; }

  using base_list::count_;
  using base_list::head_;
  using base_list::tail_;
};

template <typename U, size_t kOffset>
class IntrusiveList<std::unique_ptr<U>, kOffset>
    : public IntrusiveListUniquePtrBase<U, kOffset> {};

template <typename U>
class IntrusiveList<std::unique_ptr<U>, kUseDefaultLinkOffset>
    : public IntrusiveListUniquePtrBase<U, offsetof(U, link)> {};

}  // namespace iree

#endif  // IREE_HAL_VULKAN_UTIL_INTRUSIVE_LIST_UNIQUE_PTR_H_
