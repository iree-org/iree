#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/ops.h"
#include "iree/vm/shims.h"

//=============================================================================
// module "list_ops_ref"
//=============================================================================

struct list_ops_ref_s {
  iree_allocator_t allocator;
};
struct list_ops_ref_state_s {
  iree_allocator_t allocator;
  uint8_t rwdata[0];
  iree_vm_ref_t refs[0];
};
typedef struct list_ops_ref_s list_ops_ref_t;
typedef struct list_ops_ref_state_s list_ops_ref_state_t;

iree_status_t list_ops_ref_test_i8_impl(list_ops_ref_state_t* state) {
  // %c42 = vm.const.i32 42 : i32
  int32_t c42 = 42;

  // %list = vm.list.alloc %c42 : (i32) -> !vm.list<i8>
  iree_vm_type_def_t element_type =
      iree_vm_type_def_make_value_type(IREE_VM_VALUE_TYPE_I8);
  iree_vm_type_def_t* element_type_ptr = &element_type;
  iree_vm_list_t* list = nullptr;
  iree_vm_list_t** list_ptr = &list;
  iree_vm_list_create(element_type_ptr, c42, state->allocator, list_ptr);

  // %sz = vm.list.size %list : (!vm.list<i8>) -> i32
  int32_t sz = iree_vm_list_size(list);

  // %sz_dno = iree.do_not_optimize(%sz) : i32

  // vm.return
  return iree_ok_status();
}

//=============================================================================
// The code below setups functions and lookup tables to implement the vm
// interface
//=============================================================================
//=============================================================================
// module "list_ops_ref"
//=============================================================================

static iree_status_t list_ops_ref_test_i8(iree_vm_stack_t* stack,
                                          list_ops_ref_t* module,
                                          list_ops_ref_state_t* state) {
  return list_ops_ref_test_i8_impl(state);
}
static const iree_vm_native_export_descriptor_t list_ops_ref_exports_[] = {
    {iree_make_cstring_view("test_i8"), iree_make_cstring_view("0v_v"), 0,
     NULL},
};

static const iree_vm_native_import_descriptor_t list_ops_ref_imports_[] = {};

static const iree_vm_native_function_ptr_t list_ops_ref_funcs_[] = {
    {(iree_vm_native_function_shim_t)call_0v_v_shim,
     (iree_vm_native_function_target_t)list_ops_ref_test_i8},
};

static const iree_vm_native_module_descriptor_t list_ops_ref_descriptor_ = {
    iree_make_cstring_view("list_ops_ref"),
    IREE_ARRAYSIZE(list_ops_ref_imports_),
    list_ops_ref_imports_,
    IREE_ARRAYSIZE(list_ops_ref_exports_),
    list_ops_ref_exports_,
    IREE_ARRAYSIZE(list_ops_ref_funcs_),
    list_ops_ref_funcs_,
    0,
    NULL,
};
static iree_status_t list_ops_ref_alloc_state(
    void* self, iree_allocator_t allocator,
    iree_vm_module_state_t** out_module_state) {
  list_ops_ref_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->allocator = allocator;
  state->allocator = iree_allocator_system();
  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}
static void list_ops_ref_free_state(void* self,
                                    iree_vm_module_state_t* module_state) {
  list_ops_ref_state_t* state = (list_ops_ref_state_t*)module_state;
  iree_allocator_free(state->allocator, state);
}

static void list_ops_ref_destroy(void* self) {
  list_ops_ref_t* module = (list_ops_ref_t*)self;
  iree_allocator_free(module->allocator, module);
}

static iree_status_t list_ops_ref_create(iree_allocator_t allocator,
                                         iree_vm_module_t** out_module) {
  // Allocate shared module state.
  list_ops_ref_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  module->allocator = allocator;

  iree_vm_module_t interface;
  iree_status_t status = iree_vm_module_initialize(&interface, module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, module);
    return status;
  }
  interface.destroy = list_ops_ref_destroy;
  interface.alloc_state = list_ops_ref_alloc_state;
  interface.free_state = list_ops_ref_free_state;
  return iree_vm_native_module_create(&interface, &list_ops_ref_descriptor_,
                                      allocator, out_module);
}
