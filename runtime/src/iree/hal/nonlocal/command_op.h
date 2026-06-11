#include <stdint.h>

#define OP_DEVICE_ALLOC 1
#define OP_DEVICE_FREE 2
#define OP_COPY_IN 3
#define OP_COPY_OUT 4
#define OP_COPY 5
#define OP_ELF_LOAD 6
#define OP_ELF_INIT 7
#define OP_ELF_GET_ATTRS 8
#define OP_ELF_CALL 9
#define OP_ELF_FREE 10

typedef uint8_t command_op_t;
typedef void * command_ptr_t;
typedef uint64_t command_value_t;

