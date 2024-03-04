//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_SWAPPABLE_H
#define _LIBCUDACXX___CONCEPTS_SWAPPABLE_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__concepts/assignable.h"
#include "../__concepts/class_or_enum.h"
#include "../__concepts/common_reference_with.h"
#include "../__concepts/constructible.h"
#include "../__type_traits/extent.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_nothrow_move_assignable.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/remove_cvref.h"
#include "../__type_traits/type_identity.h"
#include "../__type_traits/void_t.h"
#include "../__utility/declval.h"
#include "../__utility/exchange.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

// [concept.swappable]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__swap)

  template<class _Tp>
  void swap(_Tp&, _Tp&) = delete;

#if _LIBCUDACXX_STD_VER > 17
  template<class _Tp, class _Up>
  concept __unqualified_swappable_with =
    (__class_or_enum<remove_cvref_t<_Tp>> || __class_or_enum<remove_cvref_t<_Up>>) &&
    requires(_Tp&& __t, _Up&& __u) {
      swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u));
    };

  template<class _Tp>
  concept __exchangeable =
    !__unqualified_swappable_with<_Tp&, _Tp&> &&
    move_constructible<_Tp> &&
    assignable_from<_Tp&, _Tp>;

#else // ^^^ CXX20 ^^^ / vvv CXX17 vvv

  template<class _Tp, class _Up>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __unqualified_swappable_with_,
    requires(_Tp&& __t, _Up&& __u)(
      swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u))
    ));

  template<class _Tp, class _Up>
  _LIBCUDACXX_CONCEPT __unqualified_swappable_with = _LIBCUDACXX_FRAGMENT(__unqualified_swappable_with_, _Tp, _Up);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __exchangeable_,
    requires()(
      requires(!__unqualified_swappable_with<_Tp&, _Tp&>),
      requires(move_constructible<_Tp>),
      requires(assignable_from<_Tp&, _Tp>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __exchangeable = _LIBCUDACXX_FRAGMENT(__exchangeable_, _Tp);
#endif // _LIBCUDACXX_STD_VER < 20


#if _LIBCUDACXX_STD_VER > 17 && !defined(_LIBCUDACXX_COMPILER_NVHPC) // nvbug4051640
  struct __fn;

#if defined(_LIBCUDACXX_COMPILER_NVCC)
#  pragma nv_diag_suppress 2642
#endif // _LIBCUDACXX_COMPILER_NVCC
  template<class _Tp, class _Up, size_t _Size>
  concept __swappable_arrays =
    !__unqualified_swappable_with<_Tp(&)[_Size], _Up(&)[_Size]> &&
    extent_v<_Tp> == extent_v<_Up> &&
    requires(_Tp(& __t)[_Size], _Up(& __u)[_Size], const __fn& __swap) {
      __swap(__t[0], __u[0]);
    };
#if defined(_LIBCUDACXX_COMPILER_NVCC)
#  pragma nv_diag_default 2642
#endif // _LIBCUDACXX_COMPILER_NVCC
#else
  template<class _Tp, class _Up, size_t _Size, class = void>
  _LIBCUDACXX_INLINE_VAR constexpr bool __swappable_arrays = false;
#endif // _LIBCUDACXX_STD_VER < 20 || defined(_LIBCUDACXX_COMPILER_NVHPC)


  template<class _Tp, class _Up, class = void>
  _LIBCUDACXX_INLINE_VAR constexpr bool __noexcept_swappable_arrays = false;

  struct __fn {
    // 2.1   `S` is `(void)swap(E1, E2)`* if `E1` or `E2` has class or enumeration type and...
    // *The name `swap` is used here unqualified.
    _LIBCUDACXX_TEMPLATE(class _Tp, class _Up)
      (requires __unqualified_swappable_with<_Tp, _Up>)
    _LIBCUDACXX_INLINE_VISIBILITY constexpr void operator()(_Tp&& __t, _Up&& __u) const
      noexcept(noexcept(swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u))))
    {
      swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u));
    }

    // 2.2   Otherwise, if `E1` and `E2` are lvalues of array types with equal extent and...
    _LIBCUDACXX_TEMPLATE(class _Tp, class _Up, size_t _Size)
      (requires __swappable_arrays<_Tp, _Up, _Size>)
    _LIBCUDACXX_INLINE_VISIBILITY constexpr void operator()(_Tp(& __t)[_Size], _Up(& __u)[_Size]) const
      noexcept(__noexcept_swappable_arrays<_Tp, _Up>)
    {
      // TODO(cjdb): replace with `_CUDA_VRANGES::swap_ranges`.
      for (size_t __i = 0; __i < _Size; ++__i) {
        (*this)(__t[__i], __u[__i]);
      }
    }

    // 2.3   Otherwise, if `E1` and `E2` are lvalues of the same type `T` that models...
    _LIBCUDACXX_TEMPLATE(class _Tp)
      (requires __exchangeable<_Tp>)
    _LIBCUDACXX_INLINE_VISIBILITY constexpr void operator()(_Tp& __x, _Tp& __y) const
      noexcept(_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Tp) && _LIBCUDACXX_TRAIT(is_nothrow_move_assignable, _Tp))
    {
      __y = _CUDA_VSTD::exchange(__x, _CUDA_VSTD::move(__y));
    }
  };

#if _LIBCUDACXX_STD_VER < 20 || defined(_LIBCUDACXX_COMPILER_NVHPC)
  template<class _Tp, class _Up, class _Size>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __swappable_arrays_,
    requires(_Tp(& __t)[_Size::value], _Up(& __u)[_Size::value], const __fn& __swap)(
      requires(!__unqualified_swappable_with<_Tp(&)[_Size::value], _Up(&)[_Size::value]>),
      requires(_LIBCUDACXX_TRAIT(extent, _Tp) == _LIBCUDACXX_TRAIT(extent, _Up)),
      (__swap(__t[0], __u[0]))
    ));

  template<class _Tp, class _Up, size_t _Size>
  _LIBCUDACXX_INLINE_VAR constexpr bool __swappable_arrays<_Tp, _Up, _Size, void_t<type_identity_t<_Tp>>> =
    _LIBCUDACXX_FRAGMENT(__swappable_arrays_, _Tp, _Up, _CUDA_VSTD::integral_constant<size_t, _Size>);
#endif // _LIBCUDACXX_STD_VER < 20 || defined(_LIBCUDACXX_COMPILER_NVHPC)

  template<class _Tp, class _Up>
  _LIBCUDACXX_INLINE_VAR constexpr bool __noexcept_swappable_arrays<_Tp, _Up, void_t<type_identity_t<_Tp>>> =
    noexcept(__swap::__fn{}(_CUDA_VSTD::declval<_Tp&>(), _CUDA_VSTD::declval<_Up&>()));

_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto swap = __swap::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17
template<class _Tp>
concept swappable = requires(_Tp& __a, _Tp& __b) { _CUDA_VRANGES::swap(__a, __b); };

template<class _Tp, class _Up>
concept swappable_with =
  common_reference_with<_Tp, _Up> &&
  requires(_Tp&& __t, _Up&& __u) {
    _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Tp>(__t));
    _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Up>(__u), _CUDA_VSTD::forward<_Up>(__u));
    _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u));
    _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Up>(__u), _CUDA_VSTD::forward<_Tp>(__t));
  };
#else
template<class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __swappable_,
  requires(_Tp& __a, _Tp& __b) //
  (_CUDA_VRANGES::swap(__a, __b)));

template<class _Tp>
_LIBCUDACXX_CONCEPT swappable = _LIBCUDACXX_FRAGMENT(__swappable_, _Tp);

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __swappable_with_,
  requires(_Tp&& __t, _Up&& __u) //
  (requires(common_reference_with<_Tp, _Up>),
   _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Tp>(__t)),
   _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Up>(__u), _CUDA_VSTD::forward<_Up>(__u)),
   _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u)),
   _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Up>(__u), _CUDA_VSTD::forward<_Tp>(__t))));

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT swappable_with = _LIBCUDACXX_FRAGMENT(__swappable_with_, _Tp, _Up);
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX_STD_VER > 11

#endif // _LIBCUDACXX___CONCEPTS_SWAPPABLE_H
