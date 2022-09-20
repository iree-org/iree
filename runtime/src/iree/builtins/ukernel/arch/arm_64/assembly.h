// Borrowed from XNNPACK's assembly.h (thanks!)
// clang-format off
#ifdef __wasm__
  .macro BEGIN_FUNCTION name
    .text
    .section    .text.\name,"",@
    .hidden     \name
    .globl      \name
    .type       \name,@function
    \name:
  .endm

  .macro END_FUNCTION name
    end_function
  .endm
#elif defined(__ELF__)
  .macro BEGIN_FUNCTION name
    .text
    .p2align 4
    .global \name
    .hidden \name
    .type \name, %function
    \name:
  .endm

  .macro END_FUNCTION name
    .size \name, .-\name
  .endm
#elif defined(__MACH__)
  .macro BEGIN_FUNCTION name
    .text
    .p2align 4
    .global _\name
    .private_extern _\name
    _\name:
  .endm

  .macro END_FUNCTION name
  .endm
#endif

#ifdef __ELF__
  .macro ALLOW_NON_EXECUTABLE_STACK
    .section ".note.GNU-stack","",%progbits
  .endm
#else
  .macro ALLOW_NON_EXECUTABLE_STACK
  .endm
#endif
