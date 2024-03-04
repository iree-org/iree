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

#ifndef _LIBCUDACXX___MDSPAN_CONFIG_HPP
#define _LIBCUDACXX___MDSPAN_CONFIG_HPP

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if _LIBCUDACXX_STD_VER > 11

#ifndef __has_include
#  define __has_include(x) 0
#endif

#ifndef __cuda_std__
#if __has_include(<version>)
#  include <version>
#else
#  include <type_traits>
#  include <utility>
#endif
#endif

#ifdef _MSVC_LANG
#define __MDSPAN_CPLUSPLUS _MSVC_LANG
#else
#define __MDSPAN_CPLUSPLUS __cplusplus
#endif

#define __MDSPAN_CXX_STD_14 201402L
#define __MDSPAN_CXX_STD_17 201703L
#define __MDSPAN_CXX_STD_20 202002L

#define __MDSPAN_HAS_CXX_14 (__MDSPAN_CPLUSPLUS >= __MDSPAN_CXX_STD_14)
#define __MDSPAN_HAS_CXX_17 (__MDSPAN_CPLUSPLUS >= __MDSPAN_CXX_STD_17)
#define __MDSPAN_HAS_CXX_20 (__MDSPAN_CPLUSPLUS >= __MDSPAN_CXX_STD_20)

static_assert(__MDSPAN_CPLUSPLUS >= __MDSPAN_CXX_STD_14, "mdspan requires C++14 or later.");

#ifndef __MDSPAN_COMPILER_CLANG
#  if defined(__clang__)
#    define __MDSPAN_COMPILER_CLANG __clang__
#  endif
#endif

#if !defined(__MDSPAN_COMPILER_MSVC) && !defined(__MDSPAN_COMPILER_MSVC_CLANG)
#  if defined(_MSC_VER)
#    if !defined(__MDSPAN_COMPILER_CLANG)
#      define __MDSPAN_COMPILER_MSVC _MSC_VER
#    else
#      define __MDSPAN_COMPILER_MSVC_CLANG _MSC_VER
#    endif
#  endif
#endif

#ifndef __MDSPAN_COMPILER_INTEL
#  ifdef __INTEL_COMPILER
#    define __MDSPAN_COMPILER_INTEL __INTEL_COMPILER
#  endif
#endif

#ifndef __MDSPAN_COMPILER_APPLECLANG
#  ifdef __apple_build_version__
#    define __MDSPAN_COMPILER_APPLECLANG __apple_build_version__
#  endif
#endif

#ifndef __MDSPAN_HAS_CUDA
#  if defined(__CUDACC__)
#    define __MDSPAN_HAS_CUDA __CUDACC__
#  endif
#endif

#ifndef __MDSPAN_HAS_HIP
#  if defined(__HIPCC__)
#    define __MDSPAN_HAS_HIP __HIPCC__
#  endif
#endif

#ifndef __has_cpp_attribute
#  define __has_cpp_attribute(x) 0
#endif

#ifndef __MDSPAN_PRESERVE_STANDARD_LAYOUT
// Preserve standard layout by default, but we're not removing the old version
// that turns this off until we're sure this doesn't have an unreasonable cost
// to the compiler or optimizer.
#  define __MDSPAN_PRESERVE_STANDARD_LAYOUT 1
#endif

#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
#  if ((__has_cpp_attribute(no_unique_address) >= 201803L) && \
       (!defined(_LIBCUDACXX_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS) || __MDSPAN_HAS_CXX_20))
#    define __MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS 1
#    define __MDSPAN_NO_UNIQUE_ADDRESS [[no_unique_address]]
#  else
#    define __MDSPAN_NO_UNIQUE_ADDRESS
#  endif
#endif

#ifndef __MDSPAN_USE_CONCEPTS
// Looks like concepts doesn't work in CUDA 12
#  if defined(__cpp_concepts) && __cpp_concepts >= 201507L && !defined __cuda_std__
#    define __MDSPAN_USE_CONCEPTS 1
#  endif
#endif

#ifndef __MDSPAN_USE_FOLD_EXPRESSIONS
#  if (defined(__cpp_fold_expressions) && __cpp_fold_expressions >= 201603L) \
          || (!defined(__cpp_fold_expressions) && __MDSPAN_HAS_CXX_17)
#    define __MDSPAN_USE_FOLD_EXPRESSIONS 1
#  endif
#endif

#ifndef __MDSPAN_NEEDS_TRAIT_VARIABLE_TEMPLATE_BACKPORTS
#  if (!(defined(__cpp_lib_type_trait_variable_templates) && __cpp_lib_type_trait_variable_templates >= 201510L) \
          || !__MDSPAN_HAS_CXX_17)
#    if !(defined(__MDSPAN_COMPILER_APPLECLANG) && __MDSPAN_HAS_CXX_17)
#      define __MDSPAN_NEEDS_TRAIT_VARIABLE_TEMPLATE_BACKPORTS 1
#    endif
#  endif
#endif

#ifndef __MDSPAN_USE_VARIABLE_TEMPLATES
#  if (defined(__cpp_variable_templates) && __cpp_variable_templates >= 201304 && __MDSPAN_HAS_CXX_17) \
        || (!defined(__cpp_variable_templates) && __MDSPAN_HAS_CXX_17)
#    define __MDSPAN_USE_VARIABLE_TEMPLATES 1
#  endif
#endif // __MDSPAN_USE_VARIABLE_TEMPLATES

#ifndef __MDSPAN_USE_CONSTEXPR_14
#  if (defined(__cpp_constexpr) && __cpp_constexpr >= 201304) \
        || (!defined(__cpp_constexpr) && __MDSPAN_HAS_CXX_14) \
        && (!(defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1700))
#    define __MDSPAN_USE_CONSTEXPR_14 1
#  endif
#endif

#ifndef __MDSPAN_USE_INTEGER_SEQUENCE
#  if defined(__MDSPAN_COMPILER_MSVC)
#    if (defined(__cpp_lib_integer_sequence) && __cpp_lib_integer_sequence >= 201304)
#      define __MDSPAN_USE_INTEGER_SEQUENCE 1
#    endif
#  endif
#endif
#ifndef __MDSPAN_USE_INTEGER_SEQUENCE
#  if (defined(__cpp_lib_integer_sequence) && __cpp_lib_integer_sequence >= 201304) \
        || (!defined(__cpp_lib_integer_sequence) && __MDSPAN_HAS_CXX_14) \
        /* as far as I can tell, libc++ seems to think this is a C++11 feature... */ \
        || (defined(__GLIBCXX__) && __GLIBCXX__ > 20150422 && __GNUC__ < 5 && !defined(__INTEL_CXX11_MODE__))
     // several compilers lie about integer_sequence working properly unless the C++14 standard is used
#    define __MDSPAN_USE_INTEGER_SEQUENCE 1
#  elif defined(__MDSPAN_COMPILER_APPLECLANG) && __MDSPAN_HAS_CXX_14
     // appleclang seems to be missing the __cpp_lib_... macros, but doesn't seem to lie about C++14 making
     // integer_sequence work
#    define __MDSPAN_USE_INTEGER_SEQUENCE 1
#  endif
#endif

#ifndef __MDSPAN_USE_RETURN_TYPE_DEDUCTION
#  if (defined(__cpp_return_type_deduction) && __cpp_return_type_deduction >= 201304) \
          || (!defined(__cpp_return_type_deduction) && __MDSPAN_HAS_CXX_14)
#    define __MDSPAN_USE_RETURN_TYPE_DEDUCTION 1
#  endif
#endif

#ifndef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
// GCC 10 is known not to work with CTAD for this case.
#  if (defined(__MDSPAN_COMPILER_CLANG) || !defined(__GNUC__) || __GNUC__ >= 11) \
      && ((defined(__cpp_deduction_guides) && __cpp_deduction_guides >= 201703) \
         || (!defined(__cpp_deduction_guides) && __MDSPAN_HAS_CXX_17))
#    define __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION 1
#  endif
#endif

#ifndef __MDSPAN_USE_ALIAS_TEMPLATE_ARGUMENT_DEDUCTION
// GCC 10 is known not to work with CTAD for this case.
#  if (defined(__MDSPAN_COMPILER_CLANG) || !defined(__GNUC__) || __GNUC__ >= 11) \
      && ((defined(__cpp_deduction_guides) && __cpp_deduction_guides >= 201907) \
          || (!defined(__cpp_deduction_guides) && __MDSPAN_HAS_CXX_20))
#    define __MDSPAN_USE_ALIAS_TEMPLATE_ARGUMENT_DEDUCTION 1
#  endif
#endif

#ifndef __MDSPAN_USE_STANDARD_TRAIT_ALIASES
#  if (defined(__cpp_lib_transformation_trait_aliases) && __cpp_lib_transformation_trait_aliases >= 201304) \
          || (!defined(__cpp_lib_transformation_trait_aliases) && __MDSPAN_HAS_CXX_14)
#    define __MDSPAN_USE_STANDARD_TRAIT_ALIASES 1
#  elif defined(__MDSPAN_COMPILER_APPLECLANG) && __MDSPAN_HAS_CXX_14
     // appleclang seems to be missing the __cpp_lib_... macros, but doesn't seem to lie about C++14
#    define __MDSPAN_USE_STANDARD_TRAIT_ALIASES 1
#  endif
#endif

#ifndef __MDSPAN_DEFAULTED_CONSTRUCTORS_INHERITANCE_WORKAROUND
#  ifdef __GNUC__
#    if __GNUC__ < 9
#      define __MDSPAN_DEFAULTED_CONSTRUCTORS_INHERITANCE_WORKAROUND 1
#    endif
#  endif
#endif

#ifndef __MDSPAN_CONDITIONAL_EXPLICIT
#  if __MDSPAN_HAS_CXX_20 && !defined(__MDSPAN_COMPILER_MSVC)
#    define __MDSPAN_CONDITIONAL_EXPLICIT(COND) explicit(COND)
#  else
#    define __MDSPAN_CONDITIONAL_EXPLICIT(COND)
#  endif
#endif

#ifndef __MDSPAN_USE_BRACKET_OPERATOR
#  if defined(__cpp_multidimensional_subscript)
#    define __MDSPAN_USE_BRACKET_OPERATOR 1
#  else
#    define __MDSPAN_USE_BRACKET_OPERATOR 0
#  endif
#endif

#ifndef __MDSPAN_USE_PAREN_OPERATOR
#  if !__MDSPAN_USE_BRACKET_OPERATOR
#    define __MDSPAN_USE_PAREN_OPERATOR 1
#  else
#    define __MDSPAN_USE_PAREN_OPERATOR 0
#  endif
#endif

#if __MDSPAN_USE_BRACKET_OPERATOR
#  define __MDSPAN_OP(mds,...) mds[__VA_ARGS__]
// Corentins demo compiler for subscript chokes on empty [] call,
// though I believe the proposal supports it?
#ifdef __MDSPAN_NO_EMPTY_BRACKET_OPERATOR
#  define __MDSPAN_OP0(mds) mds.accessor().access(mds.data_handle(),0)
#else
#  define __MDSPAN_OP0(mds) mds[]
#endif
#  define __MDSPAN_OP1(mds, a) mds[a]
#  define __MDSPAN_OP2(mds, a, b) mds[a,b]
#  define __MDSPAN_OP3(mds, a, b, c) mds[a,b,c]
#  define __MDSPAN_OP4(mds, a, b, c, d) mds[a,b,c,d]
#  define __MDSPAN_OP5(mds, a, b, c, d, e) mds[a,b,c,d,e]
#  define __MDSPAN_OP6(mds, a, b, c, d, e, f) mds[a,b,c,d,e,f]
#else
#  define __MDSPAN_OP(mds,...) mds(__VA_ARGS__)
#  define __MDSPAN_OP0(mds) mds()
#  define __MDSPAN_OP1(mds, a) mds(a)
#  define __MDSPAN_OP2(mds, a, b) mds(a,b)
#  define __MDSPAN_OP3(mds, a, b, c) mds(a,b,c)
#  define __MDSPAN_OP4(mds, a, b, c, d) mds(a,b,c,d)
#  define __MDSPAN_OP5(mds, a, b, c, d, e) mds(a,b,c,d,e)
#  define __MDSPAN_OP6(mds, a, b, c, d, e, f) mds(a,b,c,d,e,f)
#endif

#endif // _LIBCUDACXX_STD_VER > 11

#endif // _LIBCUDACXX___MDSPAN_CONFIG_HPP
