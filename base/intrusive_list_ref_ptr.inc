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

// IWYU pragma: private, include "base/intrusive_list.h"

#ifndef IREE_BASE_INTRUSIVE_LIST_REF_PTR_H_
#define IREE_BASE_INTRUSIVE_LIST_REF_PTR_H_

#include <cstddef>
#include <iterator>

#include "base/intrusive_list.h"
#include "base/ref_ptr.h"

namespace iree {

// Iterator for an IntrusiveList specialized to ref_ptr.
template <typename T, size_t kOffset, bool kForward>
class IntrusiveListRefPtrIterator
    : public std::iterator<std::input_iterator_tag, int> {
 public:
  using self_type = IntrusiveListRefPtrIterator<T, kOffset, kForward>;

  explicit IntrusiveListRefPtrIterator(IntrusiveListLink* current)
      : current_(current) {}
  self_type& operator++() {
    if (current_) {
      current_ = kForward ? current_->next : current_->prev;
    }
    return *this;
  }
  self_type operator++(int) {
    self_type tmp(current_);
    operator++();
    return tmp;
  }
  self_type& operator--() {
    if (current_) {
      current_ = kForward ? current_->prev : current_->next;
    }
    return *this;
  }
  self_type operator--(int) {
    self_type tmp(current_);
    operator--();
    return tmp;
  }
  bool operator==(const self_type& rhs) const {
    return rhs.current_ == current_;
  }
  bool operator!=(const self_type& rhs) const { return !operator==(rhs); }
  ref_ptr<T> operator*() const {
    return add_ref(impl::LinkToT<T, kOffset>(current_));
  }

 protected:
  IntrusiveListLink* current_;
};

// Specialized IntrusiveListBase for ref_ptr types.
// This makes the list methods accept/return ref_ptrs and iterate with
// a ref_ptr iterator.
template <typename T, size_t kOffset>
class IntrusiveListRefPtrBase
    : private IntrusiveListBase<
          T, IntrusiveListRefPtrIterator<T, kOffset, true>,
          IntrusiveListRefPtrIterator<T, kOffset, false>, kOffset> {
 public:
  using IteratorT = IntrusiveListRefPtrIterator<T, kOffset, true>;
  using ReverseIteratorT = IntrusiveListRefPtrIterator<T, kOffset, false>;
  using base_list = IntrusiveListBase<T, IteratorT, ReverseIteratorT, kOffset>;

  IntrusiveListRefPtrBase() = default;

  using base_list::empty;
  using base_list::size;

  using base_list::contains;
  bool contains(const ref_ptr<T>& value) const {
    return base_list::contains(value.get());
  }

  using base_list::clear;

  using base_list::begin;
  using base_list::end;
  using base_list::rbegin;
  using base_list::rend;

  inline ref_ptr<T> next(const ref_ptr<T>& value) const {
    return add_ref(base_list::next(value.get()));
  }
  inline ref_ptr<T> next(T* value) const {
    return add_ref(base_list::next(value));
  }

  inline ref_ptr<T> previous(const ref_ptr<T>& value) const {
    return add_ref(base_list::previous(value.get()));
  }
  inline ref_ptr<T> previous(T* value) const {
    return add_ref(base_list::previous(value));
  }

  // Performance: O(1)
  inline ref_ptr<T> front() const {
    return add_ref(impl::LinkToT<T, kOffset>(head_));
  }

  void push_front(const ref_ptr<T>& value) {
    base_list::push_front(value.get());
  }

  using base_list::pop_front;

  // Performance: O(1)
  inline ref_ptr<T> back() const {
    return add_ref(impl::LinkToT<T, kOffset>(tail_));
  }

  void push_back(const ref_ptr<T>& value) { base_list::push_back(value.get()); }

  using base_list::pop_back;

  void insert(const IteratorT& it, const ref_ptr<T>& value) {
    base_list::insert(it, value.get());
  }

  using base_list::erase;

  ref_ptr<T> erase(const ref_ptr<T>& value) {
    return add_ref(base_list::erase(value.get()));
  }

  void replace(const ref_ptr<T>& old_value, const ref_ptr<T>& new_value) {
    base_list::replace(old_value.get(), new_value.get());
  }
  void replace(T* old_value, const ref_ptr<T>& new_value) {
    base_list::replace(old_value, new_value.get());
  }

  using base_list::sort;

 private:
  void OnAdd(T* value) override { value->AddReference(); }
  void OnRemove(T* value) override { value->ReleaseReference(); }
  void OnDeallocate(T* value) override { value->ReleaseReference(); }

  using base_list::count_;
  using base_list::head_;
  using base_list::tail_;
};

template <typename U, size_t kOffset>
class IntrusiveList<ref_ptr<U>, kOffset>
    : public IntrusiveListRefPtrBase<U, kOffset> {};

template <typename U>
class IntrusiveList<ref_ptr<U>, kUseDefaultLinkOffset>
    : public IntrusiveListRefPtrBase<U, offsetof(U, link)> {};

}  // namespace iree

#endif  // IREE_BASE_INTRUSIVE_LIST_REF_PTR_H_
