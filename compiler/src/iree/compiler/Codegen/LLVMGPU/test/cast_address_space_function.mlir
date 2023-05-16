// RUN: iree-opt --iree-codegen-lower-ukernel-ops-to-calls --iree-llvmgpu-cast-address-space-function %s --split-input-file | FileCheck %s

module {
  func.func private @foo(memref<f32>, memref<f32, #gpu.address_space<workgroup>>, memref<f32, #gpu.address_space<workgroup>>) 
  func.func @bar() {
    %alloc_1 = memref.alloc() : memref<110xf32, #gpu.address_space<workgroup>>    
    %alloc_2 = memref.alloc() : memref<128xf32>
    %alloc_3 = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
    %a1:4 = memref.extract_strided_metadata %alloc_1 : memref<110xf32, #gpu.address_space<workgroup>> -> memref<f32, #gpu.address_space<workgroup>>, index, index, index
    %a2:4 = memref.extract_strided_metadata %alloc_2 : memref<128xf32> -> memref<f32>, index, index, index
    %a3:4 = memref.extract_strided_metadata %alloc_3 : memref<128xf32, #gpu.address_space<workgroup>> -> memref<f32, #gpu.address_space<workgroup>>, index, index, index
    call @foo(%a2, %a1, %a3) : (memref<f32>, memref<f32, #gpu.address_space<workgroup>>, memref<f32, #gpu.address_space<workgroup>>) -> ()
    return
  }
}

// CHECK:    func.func private @foo(memref<f32>, memref<f32>, memref<f32>) 

// CHECK-LABEL: func.func @bar
// CHECK:     %[[a1:.+]] = memref.alloc() : memref<110xf32, #gpu.address_space<workgroup>>    
// CHECK:     %[[a2:.+]] = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>    
// CHECK:     %[[b1:.+]], %{{.*}}, %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[a1]] : memref<110xf32, #gpu.address_space<workgroup>> -> memref<f32, #gpu.address_space<workgroup>>, index, index, index 
// CHECK:     %[[b2:.+]], %{{.*}}, %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[a2]] : memref<128xf32, #gpu.address_space<workgroup>> -> memref<f32, #gpu.address_space<workgroup>>, index, index, index
// CHECK:     %[[C1:.+]] = memref.memory_space_cast %[[b1]] : memref<f32, #gpu.address_space<workgroup>> to memref<f32>
// CHECK:     %[[C2:.+]] = memref.memory_space_cast %[[b2]] : memref<f32, #gpu.address_space<workgroup>> to memref<f32>
// CHECK:     call @foo(%{{.*}}, %[[C1]], %[[C2]]) : (memref<f32>, memref<f32>, memref<f32>) -> ()

// -----

module {
  func.func @bar() {
    %alloc_1 = memref.alloc() : memref<110xf32, #gpu.address_space<workgroup>>    
    %alloc_2 = memref.alloc() : memref<128xf32>
    %alloc_3 = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
    iree_codegen.ukernel.generic "fastfunction" ins(%alloc_1, %alloc_2 : memref<110xf32, #gpu.address_space<workgroup>>, memref<128xf32>) 
    outs(%alloc_3 : memref<128xf32, #gpu.address_space<workgroup>>) 
    return
  }
}

// CHECK-LABEL: func.func @bar
// CHECK:     call @fastfunction(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (memref<f32>, index, index, memref<f32>, index, index, memref<f32>, index, index) -> ()
