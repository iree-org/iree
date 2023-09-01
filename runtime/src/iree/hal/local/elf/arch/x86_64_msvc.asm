; Copyright 2021 The IREE Authors
;
; Licensed under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; Microsoft x64 calling convention:
; https://docs.microsoft.com/en-us/cpp/build/x64-calling-convention
; Arguments:
;   RCX, RDX, R8, R9, [stack]...
; Results:
;   RAX
; Non-volatile:
;   RBX, RBP, RDI, RSI, RSP, R12, R13, R14, R15, and XMM6-XMM15
;
; System V AMD64 ABI (used in IREE):
; https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
; Arguments:
;   RDI, RSI, RDX, RCX, R8, R9, [stack]...
; Results:
;   RAX, RDX

; Total size of non-volatile XMM registers.
_SYSV_INTEROP_STACK_SIZE = 10 * 10h

; Function prolog that saves registers that we may clobber while in code
; following the SYS-V x64 ABI.
;
; This also encodes unwind table information (.xdata/.pdata) that is used by
; debuggers/backtrace/etc to be able to look through the function on the stack.
; Though they debugger will be totally confused by the function we call into
; (it'll be expecting the Microsoft conventions and won't find them) it'll at
; least let us see the leaf guest function instead of just a bunch of our
; iree_elf_call_* thunks.
; Docs suck but we are in black magic territory so it's expected:
; https://docs.microsoft.com/en-us/cpp/build/exception-handling-x64?view=msvc-160#unwind-helpers-for-masm
_sysv_interop_prolog MACRO
  ; Save volatile general purpose registers to the stack.
  push rbp
  .pushreg rbp
  mov rbp, rsp
  .setframe rbp, 0
  push rbx
  .pushreg rbx
  push rdi
  .pushreg rdi
  push rsi
  .pushreg rsi
  push r12
  .pushreg r12
  push r13
  .pushreg r13
  push r14
  .pushreg r14
  push r15
  .pushreg r15

  ; Setup stack space for storing the SIMD registers.
  ; NOTE: we adjust this by 8 bytes to get on a 16-byte alignment so we can
  ; use the aligned movaps instruction.
  sub rsp, _SYSV_INTEROP_STACK_SIZE + 8
  .allocstack _SYSV_INTEROP_STACK_SIZE + 8

  ; Save volatile SIMD registers to the stack.
  movaps [rsp + 00h], xmm6
  .savexmm128 xmm6, 00h
  movaps [rsp + 10h], xmm7
  .savexmm128 xmm7, 10h
  movaps [rsp + 20h], xmm8
  .savexmm128 xmm8, 20h
  movaps [rsp + 30h], xmm9
  .savexmm128 xmm9, 30h
  movaps [rsp + 40h], xmm10
  .savexmm128 xmm10, 40h
  movaps [rsp + 50h], xmm11
  .savexmm128 xmm11, 50h
  movaps [rsp + 60h], xmm12
  .savexmm128 xmm12, 60h
  movaps [rsp + 70h], xmm13
  .savexmm128 xmm13, 70h
  movaps [rsp + 80h], xmm14
  .savexmm128 xmm14, 80h
  movaps [rsp + 90h], xmm15
  .savexmm128 xmm15, 90h

  .endprolog
ENDM

; Function epilog that restores registers that we may have clobbered while in
; code following the SYS-V x64 ABI.
_sysv_interop_epilog MACRO
  ; Restore volatile SIMD registers from the stack.
  movaps xmm6, [rsp + 00h]
  movaps xmm7, [rsp + 10h]
  movaps xmm8, [rsp + 20h]
  movaps xmm9, [rsp + 30h]
  movaps xmm10, [rsp + 40h]
  movaps xmm11, [rsp + 50h]
  movaps xmm12, [rsp + 60h]
  movaps xmm13, [rsp + 70h]
  movaps xmm14, [rsp + 80h]
  movaps xmm15, [rsp + 90h]
  add rsp, _SYSV_INTEROP_STACK_SIZE + 8

  ; Restore volatile general purpose registers from the stack.
  pop r15
  pop r14
  pop r13
  pop r12
  pop rsi
  pop rdi
  pop rbx
  leave  ; mov rsp, rbp + pop ebp
ENDM

_TEXT SEGMENT
ALIGN 16

; void iree_elf_call_v_v(const void* symbol_ptr)
iree_elf_call_v_v PROC FRAME
  _sysv_interop_prolog

  ; RCX = symbol_ptr
  call rcx

  _sysv_interop_epilog
  ret
iree_elf_call_v_v ENDP

; void* iree_elf_call_p_i(const void* symbol_ptr, int a0)
iree_elf_call_p_i PROC FRAME
  _sysv_interop_prolog

  ; RCX = symbol_ptr
  ; RDX = a0
  mov rdi, rdx
  call rcx

  _sysv_interop_epilog
  ret
iree_elf_call_p_i ENDP

; void* iree_elf_call_p_ip(const void* symbol_ptr, int a0, void* a1)
iree_elf_call_p_ip PROC FRAME
  _sysv_interop_prolog

  ; RCX = symbol_ptr
  ; RDX = a0
  ; R8 = a1
  mov rdi, rdx
  mov rsi, r8
  call rcx

  _sysv_interop_epilog
  ret
iree_elf_call_p_ip ENDP

; int iree_elf_call_i_p(const void* symbol_ptr, void* a0)
iree_elf_call_i_p PROC FRAME
  _sysv_interop_prolog

  ; RCX = symbol_ptr
  ; RDX = a0
  mov rdi, rdx
  call rcx

  _sysv_interop_epilog
  ret
iree_elf_call_i_p ENDP

; int iree_elf_call_i_ppp(const void* symbol_ptr, void* a0, void* a1, void* a2)
iree_elf_call_i_ppp PROC FRAME
  _sysv_interop_prolog

  ; RCX = symbol_ptr
  ; RDX = a0         <- RDI
  ; R8  = a1         <- RSI
  ; R9  = a2         <- RDX
  mov rdi, rdx
  mov rsi, r8
  mov rdx, r9
  call rcx

  _sysv_interop_epilog
  ret
iree_elf_call_i_ppp ENDP

; void* iree_elf_call_p_ppp(const void* symbol_ptr, void* a0, void* a1, void* a2)
iree_elf_call_p_ppp PROC FRAME
  _sysv_interop_prolog

  ; RCX = symbol_ptr
  ; RDX = a0         <- RDI
  ; R8  = a1         <- RSI
  ; R9  = a2         <- RDX
  mov rdi, rdx
  mov rsi, r8
  mov rdx, r9
  call rcx

  _sysv_interop_epilog
  ret
iree_elf_call_p_ppp ENDP

; int iree_elf_thunk_i_ppp(const void* symbol_ptr, void* a0, void* a1, void* a2)
iree_elf_thunk_i_ppp PROC FRAME
  _sysv_interop_prolog

  ; RDI = symbol_ptr
  ; RSI = a0         -> RCX
  ; RDX = a1         -> RDX
  ; RCX = a2         -> R8
  mov r8, rcx
  mov rcx, rsi
  call rdi

  _sysv_interop_epilog
  ret
iree_elf_thunk_i_ppp ENDP

_TEXT ENDS
END
