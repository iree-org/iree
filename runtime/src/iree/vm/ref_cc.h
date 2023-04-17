// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_REF_CC_H_
#define IREE_VM_REF_CC_H_

#include <atomic>
#include <memory>
#include <utility>

#include "iree/base/api.h"
#include "iree/base/attributes.h"
#include "iree/vm/ref.h"

#ifndef __cplusplus
#error "This header is meant for use with C++ implementations."
#endif  // __cplusplus

namespace iree {
namespace vm {

//===----------------------------------------------------------------------===//
// iree::vm::RefObject C++ base type equivalent of iree_vm_ref_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): make this automatic for most types, or use type lookup.
// This could be done with SFINAE to detect iree_vm_ref_object_t or RefObject
// types. We may still need the iree_vm_ref_type_t exposed but that's relatively
// simple compared to getting the typed retain/release functions.

template <typename T>
struct ref_type_descriptor {
  static iree_vm_ref_type_t type();
};

// Users may override this with their custom types to allow the packing code to
// access their registered type ID at runtime.
template <typename T>
static inline void ref_type_retain(T* p) {
  iree_vm_ref_object_retain(p, ref_type_descriptor<T>::type());
}

template <typename T>
static inline void ref_type_release(T* p) {
  iree_vm_ref_object_release(p, ref_type_descriptor<T>::type());
}

// Base class for reference counted objects.
// Reference counted objects should be used with the iree::vm::ref<T> pointer
// type. As reference counting can be tricky always prefer to use unique_ptr and
// avoid this type. Only use this when unique_ptr is not possible, such as
// when round-tripping objects through marshaling boundaries (v8/Java) or
// any objects that may have their lifetime tied to a garbage collected
// object.
//
// Subclasses should protect their dtor so that reference counting must
// be used.
//
// This is designed to avoid the need for extra vtable space or for adding
// methods to the vtable of subclasses. This differs from the boost Pointable
// version of this object.
// Inspiration for this comes from Peter Weinert's Dr. Dobb's article:
// http://www.drdobbs.com/cpp/a-base-class-for-intrusively-reference-c/229218807
//
// RefObjects are thread safe and may be used with iree::vm::ref<T>s from
// multiple threads.
//
// Subclasses may implement a custom Delete operator to handle their
// deallocation. It should be thread safe as it may be called from any thread.
//
// Usage:
//   class MyRefObject : public RefObject<MyRefObject> {
//    public:
//     MyRefObject() = default;
//     // Optional; can be used to return to pool/etc - must be public:
//     static void Delete(MyRefObject* ptr) {
//       ::operator delete(ptr);
//     }
//   };
template <class T>
class RefObject {
  static_assert(!std::is_array<T>::value, "T must not be an array");

  // value is true if a static Delete(T*) function is present.
  struct has_custom_deleter {
    template <typename C>
    static auto Test(C* p) -> decltype(C::Delete(nullptr), std::true_type());
    template <typename>
    static std::false_type Test(...);
    static constexpr bool value =
        std::is_same<std::true_type, decltype(Test<T>(nullptr))>::value;
  };

  template <typename V, bool has_custom_deleter>
  struct delete_thunk {
    static void Delete(V* p) {
      auto ref_obj = static_cast<RefObject<V>*>(p);
      int previous_count = ref_obj->counter_.fetch_sub(1);
      if (previous_count == 1) {
        // We delete type T pointer here to avoid the need for a virtual dtor.
        V::Delete(p);
      }
    }
    static void Destroy(V* p) { V::Delete(p); }
  };

  template <typename V>
  struct delete_thunk<V, false> {
    static void Delete(V* p) {
      auto ref_obj = static_cast<RefObject<V>*>(p);
      int previous_count = ref_obj->counter_.fetch_sub(1);
      if (previous_count == 1) {
        // We delete type T pointer here to avoid the need for a virtual dtor.
        delete p;
      }
    }
    static void Destroy(V* p) { delete p; }
  };

 public:
  // Adds a reference; used by ref_ptr.
  friend void ref_ptr_add_ref(T* p) {
    auto ref_obj = static_cast<RefObject*>(p);
    ++ref_obj->counter_;
  }

  // Releases a reference, potentially deleting the object; used by ref_ptr.
  friend void ref_ptr_release_ref(T* p) {
    delete_thunk<T, has_custom_deleter::value>::Delete(p);
  }

  // Deletes the object (precondition: ref count is zero).
  friend void ref_ptr_destroy_ref(T* p) {
    delete_thunk<T, has_custom_deleter::value>::Destroy(p);
  }

  // Deletes the object (precondition: ref count is zero).
  static void DirectDestroy(void* p) {
    ref_ptr_destroy_ref(reinterpret_cast<T*>(p));
  }

  // Adds a reference.
  // ref_ptr should be used instead of this in most cases. This is required
  // for when interoperating with marshaling APIs.
  void AddReference() { ref_ptr_add_ref(static_cast<T*>(this)); }

  // Releases a reference, potentially deleting the object.
  // ref_ptr should be used instead of this in most cases. This is required
  // for when interoperating with marshaling APIs.
  void ReleaseReference() { ref_ptr_release_ref(static_cast<T*>(this)); }

  // Returns the offset of the reference counter field from the start of the
  // type T.
  //
  // This is generally unsafe to use and is here for support of the
  // iree_vm_ref_t glue that allows RefObject-derived types to be round-tripped
  // through the VM.
  //
  // For simple POD types or non-virtual classes we expect this to return 0.
  // If the type has virtual methods (dtors/etc) then it should be 4 or 8
  // (depending on pointer width). It may be other things, and instead of too
  // much crazy magic we just rely on offsetof doing the right thing here.
  static constexpr size_t offsetof_counter() {
    return offsetof(T, counter_) / IREE_VM_REF_COUNTER_ALIGNMENT;
  }

 protected:
  RefObject() { ref_ptr_add_ref(static_cast<T*>(this)); }
  RefObject(const RefObject&) = default;
  RefObject& operator=(const RefObject&) { return *this; }

  // TODO(benvanik): replace this with just iree_vm_ref_object_t.
  // That would allow us to remove a lot of these methods and reuse the C ones.
  std::atomic<int32_t> counter_{0};
};

//===----------------------------------------------------------------------===//
// iree::vm::ref<T> RAII equivalent of iree_vm_ref_t
//===----------------------------------------------------------------------===//

// Reference counted pointer container wrapping iree_vm_ref_t.
// This is modeled on boost::instrusive_ptr in that it requires no
// extra storage over the pointer type and should compile to almost
// no additional code. It also allows us to round-trip object pointers
// through regular pointers, which is critical when having to round-trip
// them through JNI/etc where we can't use things like unique_ptr/shared_ptr.
//
// The ref wrapper calls the iree_vm_ref_* functions and uses the
// iree_vm_ref_type_descriptor_t registered for the type T to manipulate the
// reference counter and, when needed, destroy the object using
// iree_vm_ref_destroy_t. Any iree_vm_ref_t can be used interchangably with
// ref<T> when RAII is needed.
//
// Example:
//   ref<Foo> p1(new Foo());    // ref count 1
//   ref<Foo> p2(p1);           // ref count 2
//   p1.reset();                // ref count 1
//   p2.reset();                // ref count 0, deleted
//
// When round-tripping the pointer through external APIs, use release():
//   ref<Foo> p1(new Foo());    // ref count 1
//   Foo* raw_p = p1.release(); // ref count 1
//   // pass to API
//   ref<Foo> p2(raw_p);        // ref count 1 (don't add ref)
//   p2.reset();                // ref count 0, deleted
//
// See the boost intrusive_ptr docs for details of behavior:
// http://www.boost.org/doc/libs/1_55_0/libs/smart_ptr/intrusive_ptr.html
//
// The retain_ref and assign_ref helpers can be used to make it easier to
// declare and use ref types:
//   ref<Foo> p = assign_ref(new Foo());  // ref count 1
//   PassRefWithRetain(retain_ref(p));
//   PassRefWithMove(std::move(p));       // ala unique_ptr/shared_ptr
//
// ref manages the target objects in a thread-safe way, though you'll want
// to take care with objects that may have pinned threads for deallocation. If
// you release the last reference to an object on a thread other than what it
// was expecting you're gonna have a bad time.
//
// Compatible only with types that implement the following methods:
//   ref_type_retain(T*)
//   ref_type_release(T*)
//   ref_type_descriptor<T>::type()
//
// If you get link errors pertaining to ref_type_registration then ensure that
// you have included the header file containing the
// IREE_VM_DECLARE_TYPE_ADAPTERS for the given type.
//
// TODO(benvanik): reconcile RefObject, iree_vm_ref_t, and this.
template <typename T>
class ref {
 private:
  typedef ref this_type;

 public:
  IREE_ATTRIBUTE_ALWAYS_INLINE iree_vm_ref_type_t type() const noexcept {
    IREE_VM_REF_ASSERT(ref_type_descriptor<T>::type());
    return ref_type_descriptor<T>::type();
  }

  IREE_ATTRIBUTE_ALWAYS_INLINE ref() noexcept : ref_(iree_vm_ref_null()) {}
  IREE_ATTRIBUTE_ALWAYS_INLINE ref(std::nullptr_t) noexcept {}
  IREE_ATTRIBUTE_ALWAYS_INLINE ref(T* p) noexcept {
    if (!p) return;
    ref_.ptr = p;
    ref_.type = ref_type_descriptor<T>::type();
  }
  // TODO(benvanik): use the offsetof_counter we already have locally here and
  // below. In theory the compiler may be able to optimize some of this based on
  // pointer equality but investigation is required.
  IREE_ATTRIBUTE_ALWAYS_INLINE ~ref() noexcept { ref_type_release<T>(get()); }

  // Don't use implicit ref copying; use retain_ref instead to make things more
  // readable. We can't delete the ctor (or, I couldn't find a way not to)
  // because the templated parameter packing magic needs it.
  ref(const ref& rhs) noexcept : ref_(rhs.ref_) { ref_type_retain<T>(get()); }
  ref& operator=(const ref&) noexcept = delete;

  // Move support to transfer ownership from one ref to another.
  ref(ref&& rhs) noexcept : ref_(rhs.ref_) { rhs.release(); }
  ref& operator=(ref&& rhs) noexcept {
    if (get() != rhs.get()) {
      ref_type_release<T>(get());
      ref_ = rhs.ref_;
      rhs.release();
    }
    return *this;
  }

  // Move support from another compatible type.
  template <typename U>
  ref(ref<U>&& rhs) noexcept {  // NOLINT
    ref_.ptr = static_cast<T*>(rhs.release());
    ref_.type = rhs.ref_.type;
  }
  template <typename U>
  ref& operator=(ref<U>&& rhs) noexcept {
    if (get() != rhs.get()) {
      ref_type_release<T>(get());
      ref_.ptr = static_cast<T*>(rhs.release());
    }
    return *this;
  }

  // Resets the object to nullptr and decrements the reference count, possibly
  // deleting it.
  void reset() noexcept {
    ref_type_release<T>(get());
    ref_ = iree_vm_ref_null();
  }

  // Releases a pointer.
  // Returns the current pointer held by this object without having
  // its reference count decremented and resets the ref to empty.
  // Returns nullptr if the ref holds no value.
  // To re-wrap in a ref use either ref<T>(value) or assign().
  IREE_ATTRIBUTE_ALWAYS_INLINE T* release() noexcept {
    T* p = get();
    ref_ = iree_vm_ref_null();
    return p;
  }

  // Assigns a pointer.
  // The pointer will be accepted by the ref and its reference count will
  // not be incremented.
  IREE_ATTRIBUTE_ALWAYS_INLINE void assign(T* value) noexcept {
    reset();
    ref_.ptr = value;
    ref_.type = ref_type_descriptor<T>::type();
  }

  // Gets the pointer referenced by this instance.
  // operator* and operator-> will assert() if there is no current object.
  constexpr T* get() const noexcept { return reinterpret_cast<T*>(ref_.ptr); }
  constexpr T& operator*() const noexcept { return *get(); }
  constexpr T* operator->() const noexcept { return get(); }

  // Returns a pointer to the inner pointer storage.
  // This allows passing a pointer to the ref as an output argument to C-style
  // creation functions.
  constexpr T** operator&() noexcept {  // NOLINT
    return reinterpret_cast<T**>(&ref_.ptr);
  }

  // Support boolean expression evaluation ala unique_ptr/shared_ptr:
  // https://en.cppreference.com/w/cpp/memory/shared_ptr/operator_bool
  constexpr operator bool() const noexcept {  // NOLINT
    return get() != nullptr;
  }
  // Supports unary expression evaluation.
  constexpr bool operator!() const noexcept { return !get(); }

  // Swap support.
  void swap(ref& rhs) { std::swap(ref_, rhs.ref_); }

  // Allows directly passing the ref to a C-API function for creation.
  // Example:
  //    iree::vm::ref<my_type_t> value;
  //    my_type_create(..., &value);
  constexpr operator iree_vm_ref_t*() const noexcept {  // NOLINT
    return &ref_;
  }

 private:
  mutable iree_vm_ref_t ref_ = {0};
};

// Constructs an object of type T and wraps it in a reference.
//
// Usage:
//   ref<MyType> a = make_ref<MyType>(...);
template <typename T, typename... Args>
inline ref<T> make_ref(Args&&... args) {
  return ref<T>(new T(std::forward<Args>(args)...));
}

// Adds a reference to the given ref and returns the same ref.
//
// Usage:
//  ref<MyType> a = AcquireRefFromSomewhere();
//  ref<MyType> b = retain_ref(a);  // ref count + 1
//  retain_ref(b);  // ref count + 1
template <typename T>
inline ref<T> retain_ref(const ref<T>& value) {
  ref_type_retain<T>(value.get());
  return ref<T>(value.get());
}

// Adds a reference to the given raw pointer and returns it wrapped in a ref.
//
// Usage:
//  MyType* raw_ptr = AcquirePointerFromSomewhere();
//  ref<MyType> p = retain_ref(raw_ptr);  // ref count + 1
template <typename T>
inline ref<T> retain_ref(T* value) {
  ref_type_retain<T>(value);
  return ref<T>(value);
}

// Assigns a raw pointer to a ref without adding a reference.
//
// Usage:
//  ref<MyType> p = assign_ref(new MyType());  // ref count untouched
template <typename T>
inline ref<T> assign_ref(T* value) {
  return ref<T>(value);
}

template <class T, class U>
inline bool operator==(ref<T> const& a, ref<U> const& b) {
  return a.get() == b.get();
}

template <class T, class U>
inline bool operator!=(ref<T> const& a, ref<U> const& b) {
  return a.get() != b.get();
}

template <class T, class U>
inline bool operator==(ref<T> const& a, U* b) {
  return a.get() == b;
}

template <class T, class U>
inline bool operator!=(ref<T> const& a, U* b) {
  return a.get() != b;
}

template <class T, class U>
inline bool operator==(T* a, ref<U> const& b) {
  return a == b.get();
}

template <class T, class U>
inline bool operator!=(T* a, ref<U> const& b) {
  return a != b.get();
}

template <class T>
inline bool operator<(ref<T> const& a, ref<T> const& b) {
  return a.get() < b.get();
}

// Swaps the pointers of two refs.
template <class T>
void swap(ref<T>& lhs, ref<T>& rhs) {
  lhs.swap(rhs);
}

//===----------------------------------------------------------------------===//
// iree::opaque_ref utility for type-erased ref values
//===----------------------------------------------------------------------===//

// An opaque reference that does not make any assertions about the type of the
// ref contained. This can be used to accept arbitrary ref objects that are then
// dynamically handled based on type.
class opaque_ref {
 public:
  opaque_ref() = default;
  opaque_ref(const opaque_ref&) = delete;
  opaque_ref& operator=(const opaque_ref&) = delete;
  opaque_ref(opaque_ref&& rhs) noexcept {
    iree_vm_ref_move(&rhs.value_, &value_);
  }
  opaque_ref& operator=(opaque_ref&& rhs) noexcept {
    iree_vm_ref_move(&rhs.value_, &value_);
    return *this;
  }
  ~opaque_ref() { iree_vm_ref_release(&value_); }

  constexpr iree_vm_ref_t* get() const noexcept { return &value_; }
  constexpr operator iree_vm_ref_t*() const noexcept {  // NOLINT
    return &value_;
  }
  constexpr bool operator!() const noexcept { return !value_.ptr; }

  // Returns a pointer to the inner pointer storage.
  // This allows passing a pointer to the ref as an output argument to C-style
  // creation functions.
  constexpr iree_vm_ref_t* operator&() noexcept { return &value_; }  // NOLINT

 private:
  mutable iree_vm_ref_t value_ = {0};
};

}  // namespace vm
}  // namespace iree

//===----------------------------------------------------------------------===//
// ref-type registration and declaration for generic types
//===----------------------------------------------------------------------===//
// This adds vm::ref<T> support for any C type that is registered with the
// dynamic type registration mechanism and that can be wrapped in an
// iree_vm_ref_t.

#define IREE_VM_DECLARE_CC_TYPE_LOOKUP(name, T)                               \
  namespace iree {                                                            \
  namespace vm {                                                              \
  template <>                                                                 \
  struct ref_type_descriptor<T> {                                             \
    static inline const iree_vm_ref_type_descriptor_t* get() {                \
      return reinterpret_cast<iree_vm_ref_type_descriptor_t*>(name##_type()); \
    }                                                                         \
    static inline iree_vm_ref_type_t type() { return name##_type(); }         \
  };                                                                          \
  }                                                                           \
  }

//===----------------------------------------------------------------------===//
// ref-type registration and declaration for core VM types
//===----------------------------------------------------------------------===//
// This adds vm::ref<T> support for arbitrary C types that implement retain and
// release methods and manage their reference count internally. These are not
// registered with the dynamic type registration mechanism.

#define IREE_VM_DECLARE_CC_TYPE_ADAPTERS(name, T)                         \
  namespace iree {                                                        \
  namespace vm {                                                          \
  template <>                                                             \
  inline void ref_type_retain(T* p) {                                     \
    name##_retain(p);                                                     \
  }                                                                       \
  template <>                                                             \
  inline void ref_type_release(T* p) {                                    \
    name##_release(p);                                                    \
  }                                                                       \
  template <>                                                             \
  class ref<T> {                                                          \
   private:                                                               \
    typedef ref this_type;                                                \
                                                                          \
   public:                                                                \
    IREE_ATTRIBUTE_ALWAYS_INLINE ref() noexcept : ptr_(nullptr) {}        \
    IREE_ATTRIBUTE_ALWAYS_INLINE ref(std::nullptr_t) noexcept             \
        : ptr_(nullptr) {}                                                \
    IREE_ATTRIBUTE_ALWAYS_INLINE ref(T* p) noexcept : ptr_(p) {}          \
    IREE_ATTRIBUTE_ALWAYS_INLINE ~ref() noexcept {                        \
      ref_type_release<T>(get());                                         \
    }                                                                     \
    ref(const ref& rhs) noexcept : ptr_(rhs.ptr_) {                       \
      ref_type_retain<T>(get());                                          \
    }                                                                     \
    ref& operator=(const ref&) noexcept = delete;                         \
    ref(ref&& rhs) noexcept : ptr_(rhs.ptr_) { rhs.release(); }           \
    ref& operator=(ref&& rhs) noexcept {                                  \
      if (get() != rhs.get()) {                                           \
        ref_type_release<T>(get());                                       \
        ptr_ = rhs.ptr_;                                                  \
        rhs.release();                                                    \
      }                                                                   \
      return *this;                                                       \
    }                                                                     \
    template <typename U>                                                 \
    ref(ref<U>&& rhs) noexcept {                                          \
      ptr_ = static_cast<T*>(rhs.release());                              \
    }                                                                     \
    template <typename U>                                                 \
    ref& operator=(ref<U>&& rhs) noexcept {                               \
      if (get() != rhs.get()) {                                           \
        ref_type_release<T>(get());                                       \
        ptr_ = static_cast<T*>(rhs.release());                            \
      }                                                                   \
      return *this;                                                       \
    }                                                                     \
    void reset() noexcept {                                               \
      ref_type_release<T>(get());                                         \
      ptr_ = nullptr;                                                     \
    }                                                                     \
    IREE_ATTRIBUTE_ALWAYS_INLINE T* release() noexcept {                  \
      T* p = get();                                                       \
      ptr_ = nullptr;                                                     \
      return p;                                                           \
    }                                                                     \
    IREE_ATTRIBUTE_ALWAYS_INLINE void assign(T* value) noexcept {         \
      reset();                                                            \
      ptr_ = value;                                                       \
    }                                                                     \
    constexpr T* get() const noexcept { return ptr_; }                    \
    constexpr T& operator*() const noexcept { return *get(); }            \
    constexpr T* operator->() const noexcept { return get(); }            \
    constexpr T** operator&() noexcept { return &ptr_; }                  \
    constexpr operator bool() const noexcept { return get() != nullptr; } \
    constexpr bool operator!() const noexcept { return !get(); }          \
    void swap(ref& rhs) { std::swap(ptr_, rhs.ptr_); }                    \
                                                                          \
   private:                                                               \
    mutable T* ptr_ = nullptr;                                            \
  };                                                                      \
  }                                                                       \
  }

#endif  // IREE_VM_REF_CC_H_
