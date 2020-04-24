// ----------------
// --- FINISHED ---
// ----------------

// ----- BASE OPS -----
// unary ops
vmla.unary_op %src, out %dst : f32

// binary ops
vmla.binary_op %lhs, %rhs, out %dst : f32

// ternary ops
vmla.ternary_op %a, %b, %c, out %dst : f32

// ----- MISC -----
// vmla convert
vmla.convert %src, out %dst : f32 -> i8

// vmla transpose
vmla.transpose %src(%src_shape : !shapex.ranked_shape<[64,32]>),
               out %dst(%dst_shape : !shapex.ranked_shape<[32,64]>)
               {permutation = dense<[1, 0]> : tensor<2xi32>} : f32

// vmla batch matmul pseudo
%dst = vmla.batch.matmul.pseudo %lhs, %rhs :
  (tensor<32x256x128xf32>, tensor<32x1x128xf32>) -> tensor<32x1x256xf32>

// vmla batch matmul
vmla.batch.matmul %lhs(%lhs_shape : !shapex.ranked_shape<[8,4,4]>) : f32,
                  %rhs(%rhs_shape : !shapex.ranked_shape<[8,1,4]>) : f32,
                  out %dst(%dst_shape : !shapex.ranked_shape<[8,1,4]>) : f32

// ----- BUFFERS -----
// vmla buffer const
%result = vmla.buffer.const %value : !iree.byte_buffer -> !vmla.buffer

// vmla buffer alloc
%result = vmla.buffer.alloc byte_length = %byte_length : !vmla.buffer

// vmla buffer clone
%result = vmla.buffer.clone %src : !vmla.buffer

// vmla buffer byte length
%result = vmla.buffer.byte_length %value : index

// vmla buffer view
%result = vmla.buffer.view %src[%byte_offset],
                           byte_length = %byte_length : !vmla.buffer

// vmla buffer copy
vmla.buffer.copy %src[%src_byte_offset],
                 out %dst[%dst_byte_offset], byte_length = %byte_length

// vmla buffer fill
vmla.buffer.fill %src, out %dst

// vmla buffer load i32
%result = vmla.buffer.load.i32 %src[%byte_offset] : i32

// ------------------
// --- INCOMPLETE ---
// ------------------

// vmla constant
%result = "vmla.constant"() {value = dense<> : tensor<f32>} : () -> !vmla.buffer
result = vmla.constant value = dense<> : tensor<f32> : !vmla.buffer

// vmla cmp
"vmla.cmp"(%lhs, %rhs, %dst) {predicate = 1 : i32, element_type = f32} : (!vmla.buffer, !vmla.buffer, !vmla.buffer) -> ()
vmla.cmp %lhs, %rhs out %dst {predicate = 1 : i32} : f32

// vmla select
"vmla.select"(%cond, %lhs, %rhs, %dst) { element_type = f32 } : (!vmla.buffer, !vmla.buffer, !vmla.buffer, !vmla.buffer) -> ()
vmla.select %cond, %lhs, %rhs, out %dst : f32


// ----- SHAPE -----
// vmla copy
// can %src_shape != %dst_shape?
"vmla.copy"(%src, %src_shape, %src_indicies, %src_indicies, // ??? Varadic, VMLA_Index
            %dst, %dst_shape, %dst_indicies, %dst_indicies,
            %lengths, %lengths) {element_type = i32}
vmla.copy %src(%src_shape : !shapex.ranked_shape<[64]>)[%src_indicies, %src_indicies]
          out %dst(%dst_shape : !shape.ranked_shape<[32]>)[%dst_indicies, %dst_indicies]
          [%lengths, %lengths] : i32

// vmla reverse
// can %src_shape != %dst_shape?
"vmla.reverse"(%src, %src_shape, %dst, %dst_shape) {dimensions = dense<[1]> : tensor<i32>, element_type = f32} :
              (!vmla.buffer, !shapex.ranked_shape<[4, 8]>, !vmla.buffer, !shapex.ranked_shape<[4, 8]>) -> ()
vmla.reverse %src(%src_shape : !shapex.ranked_shape<[4, 8]>),
             out %dst(%dst_shape : !shapex.ranked_shape<[4, 8]>)
             {dimensions = dense<[1]> : tensor<i32>} : f32

// vmla pad
// Don't have the background knowledge to know if paddings can be moved in a
// semantically useful way
"vmla.pad"(%src, %src_shape, %value, %value_shape, %dst, %dst_shape)
          {edge_padding_low = dense<> : tensor<i32>,
           edge_padding_high = dense<> : tensor<i32>,
           interior_padding = dense<> : tensor<i32>,
           element_type = f32} :
           (!vmla.buffer, !shapex.ranked_shape<[]>, !vmla.buffer, !shapex.ranked_shape<[]>, !vmla.buffer, !shapex.ranked_shape<[]>) -> ()
vmla.pad %src(%src_shape : !shapex.ranked_shape<[]>),
         %value(%value_shape : !shapex.ranked_shape<[]>),
         out %dst(%dst_shape) : !shapex.ranked_shape<[]>)
        {edge_padding_low = dense<> : tensor<i32>,
         edge_padding_high = dense<> : tensor<i32>,
         interior_padding = dense<> : tensor<i32>} : f32

// vmla broadcast
"vmla.broadcast"(%src, %src_shape, %dst, %dst_shape) {element_type = f32} : (stuff) -> ()
vmla.broadcast %src(%src_shape : !shapex.ranked_shape<[]>),
               out %dst(%dst_shape : !shapex.ranked_shape<[]>) : f32

// vmla tile
"vmla.tile"(%src, %src_shape, %dst, %dst_shape) {element_type = f32} : (stuff) -> ()
vmla.tile %src(%src_shape : !shapex.ranked_shape<[]>),
          out %dst(%dst_shape : !shapex.ranked_shape<[]>) : f32

// vmla gather
"vmla.gather"(%src, %src_shape, %indices, %indices_shape, %dst, %dst_shape)
             {dim = 1 : i64, batch_dims = 2 : i64, element_type = f32} : (stuff) -> ()
vmla.gather %src(%src_shape : !shapex.ranked_shape<[]>),
            %indices(%indices_shape : !shapex.ranked_shape<[]>),
            out %dst(%dst_shape : !shapex.ranked_shape<[]>)
            {dim = 1 : i64, batch_dims = 2 : i64} : f32

// ----- CONVOLUTION -----
// vmla conv
"vmla.conv"(%input, %input_shape, %filter, %filter_shape, %dst, %dst_shape)
           {batch_group_count = 1 : i32,
            dst_type = f32,
            feature_group_count = 1 : i32,
            filter_type = f32,
            input_type = f32,
            lhs_dilation = dense<1> : vector<2xi32>,
            padding = dense<[1, 2, 2, 2]> : vector<4xi32>,
            rhs_dilation = dense<1> : vector<2xi32>,
            window_strides = dense<1> : vector<2xi32>} : (stuff) -> ()

vmla.conv %input(%input_shape : !shapex.ranked_shape<[]>) : f32,
          %filter(%filter_shape : !shape.ranked_shape<[]>) : f32,
          out %dst(%dst_shape : !shape.ranked_shape<[]>) : f32,
         {window_strides = dense<1> : vector<2xi32>,
          padding = dense<[1, 2, 2, 2]> : vector<4xi32>,
          lhs_dilation = dense<1> : vector<2xi32>,
          rhs_dilation = dense<1> : vector<2xi32>,
          feature_group_count = 1 : i32,
          batch_group_count = 1 : i32}

// vmla reduce op
"vmla.reduce.op"(%src, %src_shape, %init, %init_shape, %dst, %dst_shape)
                {dimension = 1 : i32, element_type = f32} :
                (!vmla.buffer, !shapex.ranked_shape<[4,8]>, !vmla.buffer, !shapex.ranked_shape<[]>, !vmla.buffer, !shapex.ranked_shape<[4]>) -> ()
vmla.reduce.op %src(%src_shape : !shapex.ranked_shape<[4,8]>),
               %init(%init_shape : !shapex.ranked_shape<[]>),
               out %dst(%dst_shape : !shapex.ranked_shape<[4]>)
               {dimension = 1} : f32

// vmla pooling op
"vmla.pooling.op"(%src, %src_shape, %init, %init_shape, %dst, %dst_shape)
                 {window_dimensions = dense<[2,2]> : tensor<2xi32>,
                  window_strides = dense<[2,2]> : tensor<2xi32>,
                  element_type = f16} : (stuff) -> ()
vmla.pooling.op %src(%src_shape : !shapex.ranked_shape<[32,32]>),
                %init(%init_shape : !shapex.ranked_shape<[]>),
                out %dst(%dst_shape : !shapex.ranked_shape<[16,16]>)
               {window_dimensions = dense<[2,2]> : tensor<2xi32>,
                window_strides = dense<[2,2]> : tensor<2xi32>} : f16


// ----- INTERFACE -----
// vmla interface const
%result = "vmla.interface.const"(%interface) {offset = 3 : IREE_IndexAttr} : (!vmla.buffer) -> !vmla.buffer
%result = vmla.interface.const %interface {offset = 3 : IREE_IndexAttr} : !vmla.buffer

// vmla interface binding
%result = "vmla.interface.binding"(%interface) {binding = 0 : i32, set = 0 : i32} : (!vmla.interface) -> !vmla.buffer
%result = vmla.interface.binding %interface {binding = 0 : i32, set = 0 : i32} : !vmla.buffer
