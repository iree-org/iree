//===----------------------------------------------------------------------===//
//
// Copyright (c) Facebook, Inc. and its affiliates.
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CONCEPTS
#define _CUDA___CONCEPTS

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#if _LIBCUDACXX_STD_VER > 11

#define _LIBCUDACXX_PP_CAT_(_Xp, ...) _Xp##__VA_ARGS__
#define _LIBCUDACXX_PP_CAT(_Xp, ...) _LIBCUDACXX_PP_CAT_(_Xp, __VA_ARGS__)

#define _LIBCUDACXX_PP_CAT2_(_Xp, ...) _Xp##__VA_ARGS__
#define _LIBCUDACXX_PP_CAT2(_Xp, ...) _LIBCUDACXX_PP_CAT2_(_Xp, __VA_ARGS__)

#define _LIBCUDACXX_PP_CAT3_(_Xp, ...) _Xp##__VA_ARGS__
#define _LIBCUDACXX_PP_CAT3(_Xp, ...) _LIBCUDACXX_PP_CAT3_(_Xp, __VA_ARGS__)

#define _LIBCUDACXX_PP_CAT4_(_Xp, ...) _Xp##__VA_ARGS__
#define _LIBCUDACXX_PP_CAT4(_Xp, ...) _LIBCUDACXX_PP_CAT4_(_Xp, __VA_ARGS__)

#define _LIBCUDACXX_PP_EVAL_(_Xp, _ARGS) _Xp _ARGS
#define _LIBCUDACXX_PP_EVAL(_Xp, ...) _LIBCUDACXX_PP_EVAL_(_Xp, (__VA_ARGS__))

#define _LIBCUDACXX_PP_EVAL2_(_Xp, _ARGS) _Xp _ARGS
#define _LIBCUDACXX_PP_EVAL2(_Xp, ...) _LIBCUDACXX_PP_EVAL2_(_Xp, (__VA_ARGS__))

#define _LIBCUDACXX_PP_EXPAND(...) __VA_ARGS__
#define _LIBCUDACXX_PP_EAT(...)

#define _LIBCUDACXX_PP_CHECK(...)                                              \
  _LIBCUDACXX_PP_EXPAND(_LIBCUDACXX_PP_CHECK_N(__VA_ARGS__, 0, ))
#define _LIBCUDACXX_PP_CHECK_N(_Xp, _Num, ...) _Num
#define _LIBCUDACXX_PP_PROBE(_Xp) _Xp, 1,
#define _LIBCUDACXX_PP_PROBE_N(_Xp, _Num) _Xp, _Num,

#define _LIBCUDACXX_PP_IS_PAREN(_Xp)                                           \
  _LIBCUDACXX_PP_CHECK(_LIBCUDACXX_PP_IS_PAREN_PROBE _Xp)
#define _LIBCUDACXX_PP_IS_PAREN_PROBE(...) _LIBCUDACXX_PP_PROBE(~)

// The final _LIBCUDACXX_PP_EXPAND here is to avoid
// https://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
#define _LIBCUDACXX_PP_COUNT(...)                                              \
  _LIBCUDACXX_PP_EXPAND(_LIBCUDACXX_PP_COUNT_(                                 \
      __VA_ARGS__, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, \
      35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,  \
      17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, ))            \
  /**/
#define _LIBCUDACXX_PP_COUNT_(                                                 \
    _01, _02, _03, _04, _05, _06, _07, _08, _09, _10, _11, _12, _13, _14, _15, \
    _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, \
    _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, \
    _46, _47, _48, _49, _50, _Np, ...)                                         \
  _Np /**/

#define _LIBCUDACXX_PP_IIF(_BIT) _LIBCUDACXX_PP_CAT_(_LIBCUDACXX_PP_IIF_, _BIT)
#define _LIBCUDACXX_PP_IIF_0(_TRUE, ...) __VA_ARGS__
#define _LIBCUDACXX_PP_IIF_1(_TRUE, ...) _TRUE

#define _LIBCUDACXX_PP_LPAREN (

#define _LIBCUDACXX_PP_NOT(_BIT) _LIBCUDACXX_PP_CAT_(_LIBCUDACXX_PP_NOT_, _BIT)
#define _LIBCUDACXX_PP_NOT_0 1
#define _LIBCUDACXX_PP_NOT_1 0

#define _LIBCUDACXX_PP_EMPTY()
#define _LIBCUDACXX_PP_COMMA() ,
#define _LIBCUDACXX_PP_LBRACE() {
#define _LIBCUDACXX_PP_RBRACE() }
#define _LIBCUDACXX_PP_COMMA_IIF(_Xp)                                          \
  _LIBCUDACXX_PP_IIF(_Xp)(_LIBCUDACXX_PP_EMPTY, _LIBCUDACXX_PP_COMMA)() /**/

#define _LIBCUDACXX_PP_FOR_EACH(_Mp, ...)                                      \
  _LIBCUDACXX_PP_FOR_EACH_N(_LIBCUDACXX_PP_COUNT(__VA_ARGS__), _Mp, __VA_ARGS__)
#define _LIBCUDACXX_PP_FOR_EACH_N(_Np, _Mp, ...)                               \
  _LIBCUDACXX_PP_CAT2(_LIBCUDACXX_PP_FOR_EACH_, _Np)(_Mp, __VA_ARGS__)
#define _LIBCUDACXX_PP_FOR_EACH_1(_Mp, _1) _Mp(_1)
#define _LIBCUDACXX_PP_FOR_EACH_2(_Mp, _1, _2) _Mp(_1) _Mp(_2)
#define _LIBCUDACXX_PP_FOR_EACH_3(_Mp, _1, _2, _3) _Mp(_1) _Mp(_2) _Mp(_3)
#define _LIBCUDACXX_PP_FOR_EACH_4(_Mp, _1, _2, _3, _4)                         \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4)
#define _LIBCUDACXX_PP_FOR_EACH_5(_Mp, _1, _2, _3, _4, _5)                     \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5)
#define _LIBCUDACXX_PP_FOR_EACH_6(_Mp, _1, _2, _3, _4, _5, _6)                 \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6)
#define _LIBCUDACXX_PP_FOR_EACH_7(_Mp, _1, _2, _3, _4, _5, _6, _7)             \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7)
#define _LIBCUDACXX_PP_FOR_EACH_8(_Mp, _1, _2, _3, _4, _5, _6, _7, _8)         \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8)

#define _LIBCUDACXX_PP_PROBE_EMPTY_PROBE__LIBCUDACXX_PP_PROBE_EMPTY            \
  _LIBCUDACXX_PP_PROBE(~)

#define _LIBCUDACXX_PP_PROBE_EMPTY()
#define _LIBCUDACXX_PP_IS_NOT_EMPTY(...)                                       \
  _LIBCUDACXX_PP_EVAL(                                                         \
      _LIBCUDACXX_PP_CHECK,                                                    \
      _LIBCUDACXX_PP_CAT(_LIBCUDACXX_PP_PROBE_EMPTY_PROBE_,                    \
                         _LIBCUDACXX_PP_PROBE_EMPTY __VA_ARGS__()))            \
  /**/

#define _LIBCUDACXX_PP_TAIL(_, ...) __VA_ARGS__

#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_M0(_REQ)                             \
  _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_SELECT_(_REQ)(_REQ)
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_M1(_REQ) _LIBCUDACXX_PP_EXPAND _REQ
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_(...)                                \
  { _LIBCUDACXX_PP_FOR_EACH(_LIBCUDACXX_CONCEPT_FRAGMENT_REQS_M, __VA_ARGS__) }
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_SELECT_(_REQ)                        \
  _LIBCUDACXX_PP_CAT3(                                                         \
      _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_SELECT_,                               \
      _LIBCUDACXX_PP_EVAL(                                                     \
          _LIBCUDACXX_PP_CHECK,                                                \
          _LIBCUDACXX_PP_CAT3(_LIBCUDACXX_CONCEPT_FRAGMENT_REQS_SELECT_PROBE_, \
                              _REQ)))                                          \
  /**/
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_SELECT_PROBE_requires                \
  _LIBCUDACXX_PP_PROBE_N(~, 1)
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_SELECT_PROBE_noexcept                \
  _LIBCUDACXX_PP_PROBE_N(~, 2)
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_SELECT_PROBE_typename                \
  _LIBCUDACXX_PP_PROBE_N(~, 3)

#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_SELECT_0 _LIBCUDACXX_PP_EXPAND
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_SELECT_1                             \
  _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_OR_NOEXCEPT
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_SELECT_2                             \
  _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_OR_NOEXCEPT
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_SELECT_3                             \
  _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_OR_NOEXCEPT
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_OR_NOEXCEPT(_REQ)           \
  _LIBCUDACXX_PP_CAT4(_LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_, _REQ)
#define _LIBCUDACXX_PP_EAT_TYPENAME_PROBE_typename _LIBCUDACXX_PP_PROBE(~)
#define _LIBCUDACXX_PP_EAT_TYPENAME_SELECT_(_Xp, ...)                          \
  _LIBCUDACXX_PP_CAT3(                                                         \
      _LIBCUDACXX_PP_EAT_TYPENAME_SELECT_,                                     \
      _LIBCUDACXX_PP_EVAL(                                                     \
          _LIBCUDACXX_PP_CHECK,                                                \
          _LIBCUDACXX_PP_CAT3(_LIBCUDACXX_PP_EAT_TYPENAME_PROBE_, _Xp)))
#define _LIBCUDACXX_PP_EAT_TYPENAME_(...)                                      \
  _LIBCUDACXX_PP_EVAL2(_LIBCUDACXX_PP_EAT_TYPENAME_SELECT_, __VA_ARGS__, )     \
  (__VA_ARGS__)
#define _LIBCUDACXX_PP_EAT_TYPENAME_SELECT_0(...) __VA_ARGS__
#define _LIBCUDACXX_PP_EAT_TYPENAME_SELECT_1(...)                              \
  _LIBCUDACXX_PP_CAT3(_LIBCUDACXX_PP_EAT_TYPENAME_, __VA_ARGS__)
#define _LIBCUDACXX_PP_EAT_TYPENAME_typename

#if (defined(__cpp_concepts) && _LIBCUDACXX_STD_VER >= 20) ||                  \
    defined(_LIBCUDACXX_DOXYGEN_INVOKED)

#define _LIBCUDACXX_CONCEPT concept

#define _LIBCUDACXX_CONCEPT_FRAGMENT(_NAME, ...)                               \
  concept _NAME =                                                              \
      _LIBCUDACXX_PP_CAT(_LIBCUDACXX_CONCEPT_FRAGMENT_REQS_, __VA_ARGS__)
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_requires(...)                        \
  requires(__VA_ARGS__) _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_M(_REQ)                              \
  _LIBCUDACXX_PP_CAT2(_LIBCUDACXX_CONCEPT_FRAGMENT_REQS_M,                     \
                      _LIBCUDACXX_PP_IS_PAREN(_REQ))                           \
  (_REQ);
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_requires(...)               \
  requires __VA_ARGS__
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_typename(...)               \
  typename _LIBCUDACXX_PP_EAT_TYPENAME_(__VA_ARGS__)
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_noexcept(...)               \
  { __VA_ARGS__ }                                                              \
  noexcept

#define _LIBCUDACXX_FRAGMENT(_NAME, ...) _NAME<__VA_ARGS__>

#else

#define _LIBCUDACXX_CONCEPT _LIBCUDACXX_INLINE_VAR constexpr bool

#define _LIBCUDACXX_CONCEPT_FRAGMENT(_NAME, ...)                               \
  _LIBCUDACXX_INLINE_VISIBILITY auto _NAME##_LIBCUDACXX_CONCEPT_FRAGMENT_impl_ \
          _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_##__VA_ARGS__ > {}                 \
  template <typename... _As>                                                   \
  _LIBCUDACXX_INLINE_VISIBILITY char _NAME##_LIBCUDACXX_CONCEPT_FRAGMENT_(     \
      _Concept::_Tag<_As...> *,                                                \
      decltype(&_NAME##_LIBCUDACXX_CONCEPT_FRAGMENT_impl_<_As...>));           \
  _LIBCUDACXX_INLINE_VISIBILITY char(                                          \
      &_NAME##_LIBCUDACXX_CONCEPT_FRAGMENT_(...))[2] /**/
#if defined(_MSC_VER) && !defined(__clang__)
#define _LIBCUDACXX_CONCEPT_FRAGMENT_TRUE(...)                                 \
  _Concept::_Is_true<decltype(_LIBCUDACXX_PP_FOR_EACH(                         \
      _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_M, __VA_ARGS__) void())>()
#else
#define _LIBCUDACXX_CONCEPT_FRAGMENT_TRUE(...)                                 \
  !(decltype(_LIBCUDACXX_PP_FOR_EACH(_LIBCUDACXX_CONCEPT_FRAGMENT_REQS_M,      \
                                     __VA_ARGS__) void(),                      \
             false){})
#endif
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_requires(...)                        \
  (__VA_ARGS__)->_Concept::_Enable_if_t < _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_2_
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_2_(...)                              \
  _LIBCUDACXX_CONCEPT_FRAGMENT_TRUE(__VA_ARGS__)
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_M(_REQ)                              \
  _LIBCUDACXX_PP_CAT2(_LIBCUDACXX_CONCEPT_FRAGMENT_REQS_M,                     \
                      _LIBCUDACXX_PP_IS_PAREN(_REQ))                           \
  (_REQ),
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_requires(...)               \
  _Concept::_Requires<__VA_ARGS__>
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_typename(...)               \
  static_cast<_Concept::_Tag<__VA_ARGS__> *>(nullptr)
#if defined(__GNUC__) && !defined(__clang__)
// GCC can't mangle noexcept expressions, so just check that the
// expression is well-formed.
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70790
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_noexcept(...) __VA_ARGS__
#else
#define _LIBCUDACXX_CONCEPT_FRAGMENT_REQS_REQUIRES_noexcept(...)               \
  _Concept::_Requires<noexcept(__VA_ARGS__)>
#endif

#define _LIBCUDACXX_FRAGMENT(_NAME, ...)                                       \
  (1u == sizeof(_NAME##_LIBCUDACXX_CONCEPT_FRAGMENT_(                          \
             static_cast<_Concept::_Tag<__VA_ARGS__> *>(nullptr), nullptr)))

#endif

////////////////////////////////////////////////////////////////////////////////
// _LIBCUDACXX_TEMPLATE
// Usage:
//   _LIBCUDACXX_TEMPLATE(typename A, typename _Bp)
//     (requires Concept1<A> _LIBCUDACXX_AND Concept2<_Bp>)
//   void foo(A a, _Bp b)
//   {}
#if (defined(__cpp_concepts) && _LIBCUDACXX_STD_VER >= 20)
#define _LIBCUDACXX_TEMPLATE(...)                                              \
  template <__VA_ARGS__> _LIBCUDACXX_PP_EXPAND /**/
#define _LIBCUDACXX_AND &&                     /**/
#else
#define _LIBCUDACXX_TEMPLATE(...)                                              \
  template <__VA_ARGS__ _LIBCUDACXX_TEMPLATE_SFINAE_AUX_ /**/
#define _LIBCUDACXX_AND                                                        \
  &&_LIBCUDACXX_true_, int > = 0, _Concept::_Enable_if_t < /**/
#endif

#define _LIBCUDACXX_TEMPLATE_SFINAE(...)                                       \
  template <__VA_ARGS__ _LIBCUDACXX_TEMPLATE_SFINAE_AUX_ /**/
#define _LIBCUDACXX_TEMPLATE_SFINAE_AUX_(...)                                  \
  , bool _LIBCUDACXX_true_ = true,                                             \
         _Concept::_Enable_if_t <                                              \
                 _LIBCUDACXX_PP_CAT(_LIBCUDACXX_TEMPLATE_SFINAE_AUX_3_,        \
                                    __VA_ARGS__) &&                            \
             _LIBCUDACXX_true_,                                                \
         int > = 0 > /**/
#define _LIBCUDACXX_TEMPLATE_SFINAE_AUX_3_requires

namespace _Concept {
template <bool> struct _Select {};

template <> struct _Select<true> { template <class _Tp> using type = _Tp; };

template <bool _Bp, class _Tp = void>
using _Enable_if_t = typename _Select<_Bp>::template type<_Tp>;

template <typename...> struct _Tag;
template <class>
_LIBCUDACXX_INLINE_VISIBILITY inline constexpr bool _Is_true() {
  return true;
}

#if defined(__clang__) || defined(_MSC_VER)
template <bool _Bp>
_LIBCUDACXX_INLINE_VISIBILITY _Concept::_Enable_if_t<_Bp> _Requires() {}
#else
template <bool _Bp, _Concept::_Enable_if_t<_Bp, int> = 0>
_LIBCUDACXX_INLINE_VAR constexpr int _Requires = 0;
#endif
} // namespace _Concept

#endif // _LIBCUDACXX_STD_VER > 11

#endif //_CUDA___CONCEPTS
