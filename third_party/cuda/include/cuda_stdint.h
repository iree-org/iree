/*
 * Copyright 2009-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __cuda_stdint_h__
#define __cuda_stdint_h__

// Compiler-specific treatment for C99's stdint.h
//
// By default, this header will use the standard headers (so it
// is your responsibility to make sure they are available), except
// on MSVC before Visual Studio 2010, when they were not provided.
// To support old MSVC, a few of the commonly-used definitions are
// provided here.  If more definitions are needed, add them here,
// or replace these definitions with a complete implementation,
// such as the ones available from Google, Boost, or MSVC10.  You
// can prevent the definition of any of these types (in order to
// use your own) by #defining CU_STDINT_TYPES_ALREADY_DEFINED.

#if !defined(CU_STDINT_TYPES_ALREADY_DEFINED)

// In VS including stdint.h forces the C++ runtime dep - provide an opt-out
// (CU_STDINT_VS_FORCE_NO_STDINT_H) for users that care (notably static
// cudart).
#if defined(_MSC_VER) && ((_MSC_VER < 1600) || defined(CU_STDINT_VS_FORCE_NO_STDINT_H))

// These definitions can be used with MSVC 8 and 9,
// which don't ship with stdint.h:

typedef unsigned   char   uint8_t;

typedef            short  int16_t;
typedef unsigned   short uint16_t;

// To keep it consistent with all MSVC build. define those types
// in the exact same way they are defined with the MSVC headers
#if defined(_MSC_VER)
typedef signed     char    int8_t;

typedef            int     int32_t;
typedef unsigned   int     uint32_t;

typedef long long          int64_t;
typedef unsigned long long uint64_t;
#else
typedef            char    int8_t;

typedef            long   int32_t;
typedef unsigned   long  uint32_t;

typedef          __int64  int64_t;
typedef unsigned __int64 uint64_t;
#endif

#elif defined(__DJGPP__)

// These definitions can be used when compiling
// C code with DJGPP, which only provides stdint.h
// when compiling C++ code with TR1 enabled.

typedef               char    int8_t;
typedef unsigned      char   uint8_t;

typedef               short  int16_t;
typedef unsigned      short uint16_t;

typedef               long   int32_t;
typedef unsigned      long  uint32_t;

typedef          long long   int64_t;
typedef unsigned long long  uint64_t;

#else

// Use standard headers, as specified by C99 and C++ TR1.
// Known to be provided by:
// - gcc/glibc, supported by all versions of glibc
// - djgpp, supported since 2001
// - MSVC, supported by Visual Studio 2010 and later

#include <stdint.h>

#endif

#endif // !defined(CU_STDINT_TYPES_ALREADY_DEFINED)


#endif // file guard
