/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/


#ifndef _LIBCUDACXX___MDSPAN_MACROS_HPP
#define _LIBCUDACXX___MDSPAN_MACROS_HPP

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__mdspan/config.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_void.h"
#include "../__type_traits/remove_reference.h"
#include "../__utility/declval.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 11

#ifndef __MDSPAN_HOST_DEVICE
#  if defined(__MDSPAN_HAS_CUDA) || defined(__MDSPAN_HAS_HIP)
#    define __MDSPAN_HOST_DEVICE __host__ __device__
#  else
#    define __MDSPAN_HOST_DEVICE
#  endif
#endif

#ifndef __MDSPAN_FORCE_INLINE_FUNCTION
#  ifdef __MDSPAN_COMPILER_MSVC // Microsoft compilers
#    define __MDSPAN_FORCE_INLINE_FUNCTION __forceinline __MDSPAN_HOST_DEVICE
#  else
#    define __MDSPAN_FORCE_INLINE_FUNCTION __attribute__((always_inline)) __MDSPAN_HOST_DEVICE
#  endif
#endif

#ifndef __MDSPAN_INLINE_FUNCTION
#  define __MDSPAN_INLINE_FUNCTION inline __MDSPAN_HOST_DEVICE
#endif

// In CUDA defaulted functions do not need host device markup
#ifndef __MDSPAN_INLINE_FUNCTION_DEFAULTED
#  define __MDSPAN_INLINE_FUNCTION_DEFAULTED
#endif

//==============================================================================
// <editor-fold desc="Preprocessor helpers"> {{{1

#define __MDSPAN_PP_COUNT(...) \
  __MDSPAN_PP_INTERNAL_EXPAND_ARGS_PRIVATE( \
    __MDSPAN_PP_INTERNAL_ARGS_AUGMENTER(__VA_ARGS__) \
  )

#define __MDSPAN_PP_INTERNAL_ARGS_AUGMENTER(...) unused, __VA_ARGS__
#define __MDSPAN_PP_INTERNAL_EXPAND(x) x
#define __MDSPAN_PP_INTERNAL_EXPAND_ARGS_PRIVATE(...) \
  __MDSPAN_PP_INTERNAL_EXPAND( \
    __MDSPAN_PP_INTERNAL_COUNT_PRIVATE( \
      __VA_ARGS__, 69, 68, 67, 66, 65, 64, 63, 62, 61, \
      60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,  \
      48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37,  \
      36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25,  \
      24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,  \
      12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 \
    ) \
  )
# define __MDSPAN_PP_INTERNAL_COUNT_PRIVATE( \
         _1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, \
    _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
    _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, \
    _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, \
    _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, \
    _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, \
    _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, \
    _70, count, ...) count \
    /**/

#define __MDSPAN_PP_STRINGIFY_IMPL(x) #x
#define __MDSPAN_PP_STRINGIFY(x) __MDSPAN_PP_STRINGIFY_IMPL(x)

#define __MDSPAN_PP_CAT_IMPL(x, y) x ## y
#define __MDSPAN_PP_CAT(x, y) __MDSPAN_PP_CAT_IMPL(x, y)

#define __MDSPAN_PP_EVAL(X, ...) X(__VA_ARGS__)

#define __MDSPAN_PP_REMOVE_PARENS_IMPL(...) __VA_ARGS__
#define __MDSPAN_PP_REMOVE_PARENS(...) __MDSPAN_PP_REMOVE_PARENS_IMPL __VA_ARGS__

// </editor-fold> end Preprocessor helpers }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Concept emulation"> {{{1

// These compatibility macros don't help with partial ordering, but they should do the trick
// for what we need to do with concepts in mdspan
#ifdef __MDSPAN_USE_CONCEPTS
#  define __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) > requires REQ
#  define __MDSPAN_FUNCTION_REQUIRES(PAREN_PREQUALS, FNAME, PAREN_PARAMS, QUALS, REQ) \
     __MDSPAN_PP_REMOVE_PARENS(PAREN_PREQUALS) FNAME PAREN_PARAMS QUALS requires REQ \
     /**/
#else
#  define __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) , typename _CUDA_VSTD::enable_if<(REQ), int>::type = 0>
#  define __MDSPAN_FUNCTION_REQUIRES(PAREN_PREQUALS, FNAME, PAREN_PARAMS, QUALS, REQ) \
     __MDSPAN_TEMPLATE_REQUIRES( \
       class __function_requires_ignored=void, \
       (_CUDA_VSTD::is_void<__function_requires_ignored>::value && REQ) \
     ) __MDSPAN_PP_REMOVE_PARENS(PAREN_PREQUALS) FNAME PAREN_PARAMS QUALS \
     /**/
#endif


#if defined(__MDSPAN_COMPILER_MSVC)
#  define __MDSPAN_TEMPLATE_REQUIRES(...) \
      __MDSPAN_PP_CAT( \
        __MDSPAN_PP_CAT(__MDSPAN_TEMPLATE_REQUIRES_, __MDSPAN_PP_COUNT(__VA_ARGS__))\
        (__VA_ARGS__), \
      ) \
    /**/
#else
#  define __MDSPAN_TEMPLATE_REQUIRES(...) \
    __MDSPAN_PP_EVAL( \
        __MDSPAN_PP_CAT(__MDSPAN_TEMPLATE_REQUIRES_, __MDSPAN_PP_COUNT(__VA_ARGS__)), \
        __VA_ARGS__ \
    ) \
    /**/
#endif

#define __MDSPAN_TEMPLATE_REQUIRES_2(TP1, REQ) \
  template<TP1 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_3(TP1, TP2, REQ) \
  template<TP1, TP2 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_4(TP1, TP2, TP3, REQ) \
  template<TP1, TP2, TP3 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_5(TP1, TP2, TP3, TP4, REQ) \
  template<TP1, TP2, TP3, TP4 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_6(TP1, TP2, TP3, TP4, TP5, REQ) \
  template<TP1, TP2, TP3, TP4, TP5 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_7(TP1, TP2, TP3, TP4, TP5, TP6, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_8(TP1, TP2, TP3, TP4, TP5, TP6, TP7, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_9(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_10(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_11(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_12(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_13(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_14(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_15(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_16(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_17(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_18(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_19(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/
#define __MDSPAN_TEMPLATE_REQUIRES_20(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18, TP19, REQ) \
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18, TP19 \
    __MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/

#define __MDSPAN_INSTANTIATE_ONLY_IF_USED \
  __MDSPAN_TEMPLATE_REQUIRES( \
    class __instantiate_only_if_used_tparam=void, \
    ( _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_void, __instantiate_only_if_used_tparam) ) \
  ) \
  /**/

// </editor-fold> end Concept emulation }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Return type deduction"> {{{1

#if __MDSPAN_USE_RETURN_TYPE_DEDUCTION
#  define __MDSPAN_DEDUCE_RETURN_TYPE_SINGLE_LINE(SIGNATURE, BODY) \
    auto __MDSPAN_PP_REMOVE_PARENS(SIGNATURE) { return __MDSPAN_PP_REMOVE_PARENS(BODY); }
#  define __MDSPAN_DEDUCE_DECLTYPE_AUTO_RETURN_TYPE_SINGLE_LINE(SIGNATURE, BODY) \
    decltype(auto) __MDSPAN_PP_REMOVE_PARENS(SIGNATURE) { return __MDSPAN_PP_REMOVE_PARENS(BODY); }
#else
#  define __MDSPAN_DEDUCE_RETURN_TYPE_SINGLE_LINE(SIGNATURE, BODY) \
    auto __MDSPAN_PP_REMOVE_PARENS(SIGNATURE) \
      -> _CUDA_VSTD::remove_cv_t<_CUDA_VSTD::remove_reference_t<decltype(BODY)>> \
    { return __MDSPAN_PP_REMOVE_PARENS(BODY); }
#  define __MDSPAN_DEDUCE_DECLTYPE_AUTO_RETURN_TYPE_SINGLE_LINE(SIGNATURE, BODY) \
    auto __MDSPAN_PP_REMOVE_PARENS(SIGNATURE) \
      -> decltype(BODY) \
    { return __MDSPAN_PP_REMOVE_PARENS(BODY); }

#endif

// </editor-fold> end Return type deduction }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="fold expressions"> {{{1

struct __mdspan_enable_fold_comma { };

#ifdef __MDSPAN_USE_FOLD_EXPRESSIONS
#  define __MDSPAN_FOLD_AND(...) ((__VA_ARGS__) && ...)
#  define __MDSPAN_FOLD_AND_TEMPLATE(...) ((__VA_ARGS__) && ...)
#  define __MDSPAN_FOLD_OR(...) ((__VA_ARGS__) || ...)
#  define __MDSPAN_FOLD_ASSIGN_LEFT(__INIT, ...) (__INIT = ... = (__VA_ARGS__))
#  define __MDSPAN_FOLD_ASSIGN_RIGHT(__PACK, ...) (__PACK = ... = (__VA_ARGS__))
#  define __MDSPAN_FOLD_TIMES_RIGHT(__PACK, ...) (__PACK * ... * (__VA_ARGS__))
#  define __MDSPAN_FOLD_PLUS_RIGHT(__PACK, ...) (__PACK + ... + (__VA_ARGS__))
#  define __MDSPAN_FOLD_COMMA(...) ((__VA_ARGS__), ...)
#else

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __fold_compatibility_impl {

// We could probably be more clever here, but at the (small) risk of losing some compiler understanding.  For the
// few operations we need, it's not worth generalizing over the operation

#if __MDSPAN_USE_RETURN_TYPE_DEDUCTION

__MDSPAN_FORCE_INLINE_FUNCTION
constexpr decltype(auto) __fold_right_and_impl() {
  return true;
}

template <class _Arg, class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr decltype(auto) __fold_right_and_impl(_Arg&& __arg, _Args&&... __args) {
  return ((_Arg&&)__arg) && __fold_compatibility_impl::__fold_right_and_impl((_Args&&)__args...);
}

__MDSPAN_FORCE_INLINE_FUNCTION
constexpr decltype(auto) __fold_right_or_impl() {
  return false;
}

template <class _Arg, class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_right_or_impl(_Arg&& __arg, _Args&&... __args) {
  return ((_Arg&&)__arg) || __fold_compatibility_impl::__fold_right_or_impl((_Args&&)__args...);
}

template <class _Arg1>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_left_assign_impl(_Arg1&& __arg1) {
  return (_Arg1&&)__arg1;
}

template <class _Arg1, class _Arg2, class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_left_assign_impl(_Arg1&& __arg1, _Arg2&& __arg2, _Args&&... __args) {
  return __fold_compatibility_impl::__fold_left_assign_impl((((_Arg1&&)__arg1) = ((_Arg2&&)__arg2)), (_Args&&)__args...);
}

template <class _Arg1>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_right_assign_impl(_Arg1&& __arg1) {
  return (_Arg1&&)__arg1;
}

template <class _Arg1, class _Arg2, class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_right_assign_impl(_Arg1&& __arg1, _Arg2&& __arg2,  _Args&&... __args) {
  return ((_Arg1&&)__arg1) = __fold_compatibility_impl::__fold_right_assign_impl((_Arg2&&)__arg2, (_Args&&)__args...);
}

template <class _Arg1>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_right_plus_impl(_Arg1&& __arg1) {
  return (_Arg1&&)__arg1;
}

template <class _Arg1, class _Arg2, class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_right_plus_impl(_Arg1&& __arg1, _Arg2&& __arg2, _Args&&... __args) {
  return ((_Arg1&&)__arg1) + __fold_compatibility_impl::__fold_right_plus_impl((_Arg2&&)__arg2, (_Args&&)__args...);
}

template <class _Arg1>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_right_times_impl(_Arg1&& __arg1) {
  return (_Arg1&&)__arg1;
}

template <class _Arg1, class _Arg2, class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_right_times_impl(_Arg1&& __arg1, _Arg2&& __arg2, _Args&&... __args) {
  return ((_Arg1&&)__arg1) * __fold_compatibility_impl::__fold_right_times_impl((_Arg2&&)__arg2, (_Args&&)__args...);
}

#else

//------------------------------------------------------------------------------
// <editor-fold desc="right and"> {{{2

template <class... _Args>
struct __fold_right_and_impl_;
template <>
struct __fold_right_and_impl_<> {
  using __rv = bool;
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl() noexcept {
    return true;
  }
};
template <class _Arg, class... _Args>
struct __fold_right_and_impl_<_Arg, _Args...> {
  using __next_t = __fold_right_and_impl_<_Args...>;
  using __rv = decltype(_CUDA_VSTD::declval<_Arg>() && _CUDA_VSTD::declval<typename __next_t::__rv>());
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(_Arg&& __arg, _Args&&... __args) noexcept {
    return ((_Arg&&)__arg) && __next_t::__impl((_Args&&)__args...);
  }
};

template <class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_right_and_impl_<_Args...>::__rv
__fold_right_and_impl(_Args&&... __args) {
  return __fold_right_and_impl_<_Args...>::__impl((_Args&&)__args...);
}

// </editor-fold> end right and }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="right or"> {{{2

template <class... _Args>
struct __fold_right_or_impl_;
template <>
struct __fold_right_or_impl_<> {
  using __rv = bool;
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl() noexcept {
    return false;
  }
};
template <class _Arg, class... _Args>
struct __fold_right_or_impl_<_Arg, _Args...> {
  using __next_t = __fold_right_or_impl_<_Args...>;
  using __rv = decltype(_CUDA_VSTD::declval<_Arg>() || _CUDA_VSTD::declval<typename __next_t::__rv>());
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(_Arg&& __arg, _Args&&... __args) noexcept {
    return ((_Arg&&)__arg) || __next_t::__impl((_Args&&)__args...);
  }
};

template <class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_right_or_impl_<_Args...>::__rv
__fold_right_or_impl(_Args&&... __args) {
  return __fold_right_or_impl_<_Args...>::__impl((_Args&&)__args...);
}

// </editor-fold> end right or }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="right plus"> {{{2

template <class... _Args>
struct __fold_right_plus_impl_;
template <class _Arg>
struct __fold_right_plus_impl_<_Arg> {
  using __rv = _Arg&&;
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(_Arg&& __arg) noexcept {
    return (_Arg&&)__arg;
  }
};
template <class _Arg1, class _Arg2, class... _Args>
struct __fold_right_plus_impl_<_Arg1, _Arg2, _Args...> {
  using __next_t = __fold_right_plus_impl_<_Arg2, _Args...>;
  using __rv = decltype(_CUDA_VSTD::declval<_Arg1>() + _CUDA_VSTD::declval<typename __next_t::__rv>());
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(_Arg1&& __arg, _Arg2&& __arg2, _Args&&... __args) noexcept {
    return ((_Arg1&&)__arg) + __next_t::__impl((_Arg2&&)__arg2, (_Args&&)__args...);
  }
};

template <class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_right_plus_impl_<_Args...>::__rv
__fold_right_plus_impl(_Args&&... __args) {
  return __fold_right_plus_impl_<_Args...>::__impl((_Args&&)__args...);
}

// </editor-fold> end right plus }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="right times"> {{{2

template <class... _Args>
struct __fold_right_times_impl_;
template <class _Arg>
struct __fold_right_times_impl_<_Arg> {
  using __rv = _Arg&&;
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(_Arg&& __arg) noexcept {
    return (_Arg&&)__arg;
  }
};
template <class _Arg1, class _Arg2, class... _Args>
struct __fold_right_times_impl_<_Arg1, _Arg2, _Args...> {
  using __next_t = __fold_right_times_impl_<_Arg2, _Args...>;
  using __rv = decltype(_CUDA_VSTD::declval<_Arg1>() * _CUDA_VSTD::declval<typename __next_t::__rv>());
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(_Arg1&& __arg, _Arg2&& __arg2, _Args&&... __args) noexcept {
    return ((_Arg1&&)__arg) * __next_t::__impl((_Arg2&&)__arg2, (_Args&&)__args...);
  }
};

template <class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_right_times_impl_<_Args...>::__rv
__fold_right_times_impl(_Args&&... __args) {
  return __fold_right_times_impl_<_Args...>::__impl((_Args&&)__args...);
}

// </editor-fold> end right times }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="right assign"> {{{2

template <class... _Args>
struct __fold_right_assign_impl_;
template <class _Arg>
struct __fold_right_assign_impl_<_Arg> {
  using __rv = _Arg&&;
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(_Arg&& __arg) noexcept {
    return (_Arg&&)__arg;
  }
};
template <class _Arg1, class _Arg2, class... _Args>
struct __fold_right_assign_impl_<_Arg1, _Arg2, _Args...> {
  using __next_t = __fold_right_assign_impl_<_Arg2, _Args...>;
  using __rv = decltype(_CUDA_VSTD::declval<_Arg1>() = _CUDA_VSTD::declval<typename __next_t::__rv>());
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(_Arg1&& __arg, _Arg2&& __arg2, _Args&&... __args) noexcept {
    return ((_Arg1&&)__arg) = __next_t::__impl((_Arg2&&)__arg2, (_Args&&)__args...);
  }
};

template <class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_right_assign_impl_<_Args...>::__rv
__fold_right_assign_impl(_Args&&... __args) {
  return __fold_right_assign_impl_<_Args...>::__impl((_Args&&)__args...);
}

// </editor-fold> end right assign }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="left assign"> {{{2

template <class... _Args>
struct __fold_left_assign_impl_;
template <class _Arg>
struct __fold_left_assign_impl_<_Arg> {
  using __rv = _Arg&&;
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(_Arg&& __arg) noexcept {
    return (_Arg&&)__arg;
  }
};
template <class _Arg1, class _Arg2, class... _Args>
struct __fold_left_assign_impl_<_Arg1, _Arg2, _Args...> {
  using __assign_result_t = decltype(_CUDA_VSTD::declval<_Arg1>() = _CUDA_VSTD::declval<_Arg2>());
  using __next_t = __fold_left_assign_impl_<__assign_result_t, _Args...>;
  using __rv = typename __next_t::__rv;
  __MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(_Arg1&& __arg, _Arg2&& __arg2, _Args&&... __args) noexcept {
    return __next_t::__impl(((_Arg1&&)__arg) = (_Arg2&&)__arg2, (_Args&&)__args...);
  }
};

template <class... _Args>
__MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_left_assign_impl_<_Args...>::__rv
__fold_left_assign_impl(_Args&&... __args) {
  return __fold_left_assign_impl_<_Args...>::__impl((_Args&&)__args...);
}

// </editor-fold> end left assign }}}2
//------------------------------------------------------------------------------

#endif


template <class... _Args>
__MDSPAN_HOST_DEVICE
constexpr __mdspan_enable_fold_comma __fold_comma_impl(_Args&&...) noexcept { return { }; }

template <bool... _Bs>
struct __bools;

} // __fold_compatibility_impl

_LIBCUDACXX_END_NAMESPACE_STD

#  define __MDSPAN_FOLD_AND(...) _CUDA_VSTD::__fold_compatibility_impl::__fold_right_and_impl((__VA_ARGS__)...)
#  define __MDSPAN_FOLD_OR(...) _CUDA_VSTD::__fold_compatibility_impl::__fold_right_or_impl((__VA_ARGS__)...)
#  define __MDSPAN_FOLD_ASSIGN_LEFT(__INIT, ...) _CUDA_VSTD::__fold_compatibility_impl::__fold_left_assign_impl(__INIT, (__VA_ARGS__)...)
#  define __MDSPAN_FOLD_ASSIGN_RIGHT(__PACK, ...) _CUDA_VSTD::__fold_compatibility_impl::__fold_right_assign_impl((__PACK)..., __VA_ARGS__)
#  define __MDSPAN_FOLD_TIMES_RIGHT(__PACK, ...) _CUDA_VSTD::__fold_compatibility_impl::__fold_right_times_impl((__PACK)..., __VA_ARGS__)
#  define __MDSPAN_FOLD_PLUS_RIGHT(__PACK, ...) _CUDA_VSTD::__fold_compatibility_impl::__fold_right_plus_impl((__PACK)..., __VA_ARGS__)
#  define __MDSPAN_FOLD_COMMA(...) _CUDA_VSTD::__fold_compatibility_impl::__fold_comma_impl((__VA_ARGS__)...)

#  define __MDSPAN_FOLD_AND_TEMPLATE(...) \
  _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_same, __fold_compatibility_impl::__bools<(__VA_ARGS__)..., true>, __fold_compatibility_impl::__bools<true, (__VA_ARGS__)...>)

#endif

// </editor-fold> end Variable template compatibility }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Pre-C++14 constexpr"> {{{1

#if __MDSPAN_USE_CONSTEXPR_14
// Workaround for a bug (I think?) in EDG frontends
#  ifdef __EDG__
#    define __MDSPAN_CONSTEXPR_14_DEFAULTED
#  else
#    define __MDSPAN_CONSTEXPR_14_DEFAULTED constexpr
#  endif
#else
#  define __MDSPAN_CONSTEXPR_14_DEFAULTED
#endif

// </editor-fold> end Pre-C++14 constexpr }}}1
//==============================================================================

#endif // _LIBCUDACXX_STD_VER > 11

#endif // _LIBCUDACXX___MDSPAN_MACROS_HPP
