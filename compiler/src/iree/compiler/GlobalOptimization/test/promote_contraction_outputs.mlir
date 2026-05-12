// RUN: iree-opt --split-input-file -iree-global-opt-promote-contraction-outputs="type=bf16 operation=matmul" %s | FileCheck %s --check-prefix=BF16-MATMUL
// RUN: iree-opt --split-input-file -iree-global-opt-promote-contraction-outputs="type=bf16 operation=conv" %s | FileCheck %s --check-prefix=BF16-CONV
// RUN: iree-opt --split-input-file -iree-global-opt-promote-contraction-outputs="type=f16 operation=matmul" %s | FileCheck %s --check-prefix=F16-MATMUL
// RUN: iree-opt --split-input-file -iree-global-opt-promote-contraction-outputs="type=f16 operation=conv" %s | FileCheck %s --check-prefix=F16-CONV

util.func public @matmul_bf16bf16bf16(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16>
  util.return %0 : tensor<100x500xbf16>
}
// BF16-MATMUL-LABEL: util.func public @matmul_bf16bf16bf16
// BF16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<100x500xf32>
// BF16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%arg2 : tensor<100x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<100x500xf32>)
// BF16-MATMUL:           arith.extf %{{.+}} : bf16 to f32
// BF16-MATMUL:         %[[MATMUL:.+]] = linalg.matmul
// BF16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<100x500xf32>)
// BF16-MATMUL:         %[[EMPTY_BF16:.+]] = tensor.empty() : tensor<100x500xbf16>
// BF16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<100x500xf32>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_BF16]] : tensor<100x500xbf16>)
// BF16-MATMUL:           arith.truncf %{{.+}} : f32 to bf16
// BF16-MATMUL:         util.return %[[TRUNCF]]

// BF16-CONV-LABEL: util.func public @matmul_bf16bf16bf16
// BF16-CONV:         %[[MATMUL:.+]] = linalg.matmul
// BF16-CONV-SAME:      ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
// BF16-CONV-SAME:      outs(%arg2 : tensor<100x500xbf16>)
// BF16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @matmul_f16f16f16(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf16>) -> tensor<100x500xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf16>) -> tensor<100x500xf16>
  util.return %0 : tensor<100x500xf16>
}
// F16-MATMUL-LABEL: util.func public @matmul_f16f16f16
// F16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<100x500xf32>
// F16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%arg2 : tensor<100x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<100x500xf32>)
// F16-MATMUL:           arith.extf %{{.+}} : f16 to f32
// F16-MATMUL:         %[[MATMUL:.+]] = linalg.matmul
// F16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<100x500xf32>)
// F16-MATMUL:         %[[EMPTY_F16:.+]] = tensor.empty() : tensor<100x500xf16>
// F16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<100x500xf32>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F16]] : tensor<100x500xf16>)
// F16-MATMUL:           arith.truncf %{{.+}} : f32 to f16
// F16-MATMUL:         util.return %[[TRUNCF]]

// F16-CONV-LABEL: util.func public @matmul_f16f16f16
// F16-CONV:         %[[MATMUL:.+]] = linalg.matmul
// F16-CONV-SAME:      ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
// F16-CONV-SAME:      outs(%arg2 : tensor<100x500xf16>)
// F16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @dynamic_matmul_bf16bf16bf16(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>,
    %arg2 : tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%arg2 : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  util.return %0 : tensor<?x?xbf16>
}
// BF16-MATMUL-LABEL: util.func public @dynamic_matmul_bf16bf16bf16
// BF16-MATMUL-DAG:     %[[C0:.+]] = arith.constant 0 : index
// BF16-MATMUL-DAG:     %[[C1:.+]] = arith.constant 1 : index
// BF16-MATMUL:         %[[DIM0:.+]] = tensor.dim %arg2, %[[C0]]
// BF16-MATMUL:         %[[DIM1:.+]] = tensor.dim %arg2, %[[C1]]
// BF16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// BF16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%arg2 : tensor<?x?xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<?x?xf32>)
// BF16-MATMUL:           arith.extf %{{.+}} : bf16 to f32
// BF16-MATMUL:         %[[MATMUL:.+]] = linalg.matmul
// BF16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<?x?xf32>)
// BF16-MATMUL:         %[[EMPTY_BF16:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xbf16>
// BF16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<?x?xf32>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_BF16]] : tensor<?x?xbf16>)
// BF16-MATMUL:           arith.truncf %{{.+}} : f32 to bf16
// BF16-MATMUL:         util.return %[[TRUNCF]]

// BF16-CONV-LABEL: util.func public @dynamic_matmul_bf16bf16bf16
// BF16-CONV:         %[[MATMUL:.+]] = linalg.matmul
// BF16-CONV-SAME:      ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
// BF16-CONV-SAME:      outs(%arg2 : tensor<?x?xbf16>)
// BF16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @dynamic_matmul_f16f16f16(%arg0 : tensor<?x?xf16>, %arg1 : tensor<?x?xf16>,
    %arg2 : tensor<?x?xf16>) -> tensor<?x?xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%arg2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  util.return %0 : tensor<?x?xf16>
}
// F16-MATMUL-LABEL: util.func public @dynamic_matmul_f16f16f16
// F16-MATMUL-DAG:     %[[C0:.+]] = arith.constant 0 : index
// F16-MATMUL-DAG:     %[[C1:.+]] = arith.constant 1 : index
// F16-MATMUL:         %[[DIM0:.+]] = tensor.dim %arg2, %[[C0]]
// F16-MATMUL:         %[[DIM1:.+]] = tensor.dim %arg2, %[[C1]]
// F16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// F16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%arg2 : tensor<?x?xf16>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<?x?xf32>)
// F16-MATMUL:           arith.extf %{{.+}} : f16 to f32
// F16-MATMUL:         %[[MATMUL:.+]] = linalg.matmul
// F16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>)
// F16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<?x?xf32>)
// F16-MATMUL:         %[[EMPTY_F16:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf16>
// F16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<?x?xf32>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F16]] : tensor<?x?xf16>)
// F16-MATMUL:           arith.truncf %{{.+}} : f32 to f16
// F16-MATMUL:         util.return %[[TRUNCF]]

// F16-CONV-LABEL: util.func public @dynamic_matmul_f16f16f16
// F16-CONV:         %[[MATMUL:.+]] = linalg.matmul
// F16-CONV-SAME:      ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>)
// F16-CONV-SAME:      outs(%arg2 : tensor<?x?xf16>)
// F16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @batch_matmul_bf16bf16bf16(%arg0 : tensor<4x100x250xbf16>, %arg1 : tensor<4x250x500xbf16>,
    %arg2 : tensor<4x100x500xbf16>) -> tensor<4x100x500xbf16> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<4x100x250xbf16>, tensor<4x250x500xbf16>)
      outs(%arg2 : tensor<4x100x500xbf16>) -> tensor<4x100x500xbf16>
  util.return %0 : tensor<4x100x500xbf16>
}
// BF16-MATMUL-LABEL: util.func public @batch_matmul_bf16bf16bf16
// BF16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<4x100x500xf32>
// BF16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%arg2 : tensor<4x100x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<4x100x500xf32>)
// BF16-MATMUL:           arith.extf %{{.+}} : bf16 to f32
// BF16-MATMUL:         %[[MATMUL:.+]] = linalg.batch_matmul
// BF16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<4x100x250xbf16>, tensor<4x250x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<4x100x500xf32>)
// BF16-MATMUL:         %[[EMPTY_BF16:.+]] = tensor.empty() : tensor<4x100x500xbf16>
// BF16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<4x100x500xf32>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_BF16]] : tensor<4x100x500xbf16>)
// BF16-MATMUL:           arith.truncf %{{.+}} : f32 to bf16
// BF16-MATMUL:         util.return %[[TRUNCF]]

// BF16-CONV-LABEL: util.func public @batch_matmul_bf16bf16bf16
// BF16-CONV:         %[[MATMUL:.+]] = linalg.batch_matmul
// BF16-CONV-SAME:      ins(%arg0, %arg1 : tensor<4x100x250xbf16>, tensor<4x250x500xbf16>)
// BF16-CONV-SAME:      outs(%arg2 : tensor<4x100x500xbf16>)
// BF16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @batch_matmul_f16f16f16(%arg0 : tensor<4x100x250xf16>, %arg1 : tensor<4x250x500xf16>,
    %arg2 : tensor<4x100x500xf16>) -> tensor<4x100x500xf16> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<4x100x250xf16>, tensor<4x250x500xf16>)
      outs(%arg2 : tensor<4x100x500xf16>) -> tensor<4x100x500xf16>
  util.return %0 : tensor<4x100x500xf16>
}
// F16-MATMUL-LABEL: util.func public @batch_matmul_f16f16f16
// F16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<4x100x500xf32>
// F16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%arg2 : tensor<4x100x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<4x100x500xf32>)
// F16-MATMUL:           arith.extf %{{.+}} : f16 to f32
// F16-MATMUL:         %[[MATMUL:.+]] = linalg.batch_matmul
// F16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<4x100x250xf16>, tensor<4x250x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<4x100x500xf32>)
// F16-MATMUL:         %[[EMPTY_F16:.+]] = tensor.empty() : tensor<4x100x500xf16>
// F16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<4x100x500xf32>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F16]] : tensor<4x100x500xf16>)
// F16-MATMUL:           arith.truncf %{{.+}} : f32 to f16
// F16-MATMUL:         util.return %[[TRUNCF]]

// F16-CONV-LABEL: util.func public @batch_matmul_f16f16f16
// F16-CONV:         %[[MATMUL:.+]] = linalg.batch_matmul
// F16-CONV-SAME:      ins(%arg0, %arg1 : tensor<4x100x250xf16>, tensor<4x250x500xf16>)
// F16-CONV-SAME:      outs(%arg2 : tensor<4x100x500xf16>)
// F16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @matvec_bf16bf16bf16(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250xbf16>,
    %arg2 : tensor<100xbf16>) -> tensor<100xbf16> {
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250xbf16>)
      outs(%arg2 : tensor<100xbf16>) -> tensor<100xbf16>
  util.return %0 : tensor<100xbf16>
}
// BF16-MATMUL-LABEL: util.func public @matvec_bf16bf16bf16
// BF16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<100xf32>
// BF16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%arg2 : tensor<100xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<100xf32>)
// BF16-MATMUL:           arith.extf %{{.+}} : bf16 to f32
// BF16-MATMUL:         %[[MATVEC:.+]] = linalg.matvec
// BF16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<100xf32>)
// BF16-MATMUL:         %[[EMPTY_BF16:.+]] = tensor.empty() : tensor<100xbf16>
// BF16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%[[MATVEC]] : tensor<100xf32>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_BF16]] : tensor<100xbf16>)
// BF16-MATMUL:           arith.truncf %{{.+}} : f32 to bf16
// BF16-MATMUL:         util.return %[[TRUNCF]]

// BF16-CONV-LABEL: util.func public @matvec_bf16bf16bf16
// BF16-CONV:         %[[MATVEC:.+]] = linalg.matvec
// BF16-CONV-SAME:      ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250xbf16>)
// BF16-CONV-SAME:      outs(%arg2 : tensor<100xbf16>)
// BF16-CONV:         util.return %[[MATVEC]]

// -----

util.func public @matvec_f16f16f16(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250xf16>,
    %arg2 : tensor<100xf16>) -> tensor<100xf16> {
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250xf16>)
      outs(%arg2 : tensor<100xf16>) -> tensor<100xf16>
  util.return %0 : tensor<100xf16>
}
// F16-MATMUL-LABEL: util.func public @matvec_f16f16f16
// F16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<100xf32>
// F16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%arg2 : tensor<100xf16>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<100xf32>)
// F16-MATMUL:           arith.extf %{{.+}} : f16 to f32
// F16-MATMUL:         %[[MATVEC:.+]] = linalg.matvec
// F16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250xf16>)
// F16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<100xf32>)
// F16-MATMUL:         %[[EMPTY_F16:.+]] = tensor.empty() : tensor<100xf16>
// F16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%[[MATVEC]] : tensor<100xf32>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F16]] : tensor<100xf16>)
// F16-MATMUL:           arith.truncf %{{.+}} : f32 to f16
// F16-MATMUL:         util.return %[[TRUNCF]]

// F16-CONV-LABEL: util.func public @matvec_f16f16f16
// F16-CONV:         %[[MATVEC:.+]] = linalg.matvec
// F16-CONV-SAME:      ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250xf16>)
// F16-CONV-SAME:      outs(%arg2 : tensor<100xf16>)
// F16-CONV:         util.return %[[MATVEC]]

// -----

util.func public @batch_vecmat_bf16bf16bf16(%arg0 : tensor<4x250xbf16>, %arg1 : tensor<4x250x500xbf16>,
    %arg2 : tensor<4x500xbf16>) -> tensor<4x500xbf16> {
  %0 = linalg.batch_vecmat ins(%arg0, %arg1 : tensor<4x250xbf16>, tensor<4x250x500xbf16>)
      outs(%arg2 : tensor<4x500xbf16>) -> tensor<4x500xbf16>
  util.return %0 : tensor<4x500xbf16>
}
// BF16-MATMUL-LABEL: util.func public @batch_vecmat_bf16bf16bf16
// BF16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<4x500xf32>
// BF16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%arg2 : tensor<4x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<4x500xf32>)
// BF16-MATMUL:           arith.extf %{{.+}} : bf16 to f32
// BF16-MATMUL:         %[[VECMAT:.+]] = linalg.batch_vecmat
// BF16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<4x250xbf16>, tensor<4x250x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<4x500xf32>)
// BF16-MATMUL:         %[[EMPTY_BF16:.+]] = tensor.empty() : tensor<4x500xbf16>
// BF16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%[[VECMAT]] : tensor<4x500xf32>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_BF16]] : tensor<4x500xbf16>)
// BF16-MATMUL:           arith.truncf %{{.+}} : f32 to bf16
// BF16-MATMUL:         util.return %[[TRUNCF]]

// BF16-CONV-LABEL: util.func public @batch_vecmat_bf16bf16bf16
// BF16-CONV:         %[[VECMAT:.+]] = linalg.batch_vecmat
// BF16-CONV-SAME:      ins(%arg0, %arg1 : tensor<4x250xbf16>, tensor<4x250x500xbf16>)
// BF16-CONV-SAME:      outs(%arg2 : tensor<4x500xbf16>)
// BF16-CONV:         util.return %[[VECMAT]]

// -----

util.func public @batch_vecmat_f16f16f16(%arg0 : tensor<4x250xf16>, %arg1 : tensor<4x250x500xf16>,
    %arg2 : tensor<4x500xf16>) -> tensor<4x500xf16> {
  %0 = linalg.batch_vecmat ins(%arg0, %arg1 : tensor<4x250xf16>, tensor<4x250x500xf16>)
      outs(%arg2 : tensor<4x500xf16>) -> tensor<4x500xf16>
  util.return %0 : tensor<4x500xf16>
}
// F16-MATMUL-LABEL: util.func public @batch_vecmat_f16f16f16
// F16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<4x500xf32>
// F16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%arg2 : tensor<4x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<4x500xf32>)
// F16-MATMUL:           arith.extf %{{.+}} : f16 to f32
// F16-MATMUL:         %[[VECMAT:.+]] = linalg.batch_vecmat
// F16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<4x250xf16>, tensor<4x250x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<4x500xf32>)
// F16-MATMUL:         %[[EMPTY_F16:.+]] = tensor.empty() : tensor<4x500xf16>
// F16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%[[VECMAT]] : tensor<4x500xf32>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F16]] : tensor<4x500xf16>)
// F16-MATMUL:           arith.truncf %{{.+}} : f32 to f16
// F16-MATMUL:         util.return %[[TRUNCF]]

// F16-CONV-LABEL: util.func public @batch_vecmat_f16f16f16
// F16-CONV:         %[[VECMAT:.+]] = linalg.batch_vecmat
// F16-CONV-SAME:      ins(%arg0, %arg1 : tensor<4x250xf16>, tensor<4x250x500xf16>)
// F16-CONV-SAME:      outs(%arg2 : tensor<4x500xf16>)
// F16-CONV:         util.return %[[VECMAT]]

// -----

util.func public @nonmatch_matmul_bf16bf16f64(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xf64>) -> tensor<100x500xf64> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xf64>) -> tensor<100x500xf64>
  util.return %0 : tensor<100x500xf64>
}
// BF16-MATMUL-LABEL: util.func public @nonmatch_matmul_bf16bf16f64
// BF16-MATMUL:         %[[MATMUL:.+]] = linalg.matmul
// BF16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
// BF16-MATMUL-SAME:      outs(%arg2 : tensor<100x500xf64>)
// BF16-MATMUL:         util.return %[[MATMUL]]

// BF16-CONV-LABEL: util.func public @nonmatch_matmul_bf16bf16f64
// BF16-CONV:         %[[MATMUL:.+]] = linalg.matmul
// BF16-CONV-SAME:      ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
// BF16-CONV-SAME:      outs(%arg2 : tensor<100x500xf64>)
// BF16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @nonmatch_matmul_f16f16f64(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf64>) -> tensor<100x500xf64> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf64>) -> tensor<100x500xf64>
  util.return %0 : tensor<100x500xf64>
}
// F16-MATMUL-LABEL: util.func public @nonmatch_matmul_f16f16f64
// F16-MATMUL:         %[[MATMUL:.+]] = linalg.matmul
// F16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
// F16-MATMUL-SAME:      outs(%arg2 : tensor<100x500xf64>)
// F16-MATMUL:         util.return %[[MATMUL]]

// F16-CONV-LABEL: util.func public @nonmatch_matmul_f16f16f64
// F16-CONV:         %[[MATMUL:.+]] = linalg.matmul
// F16-CONV-SAME:      ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
// F16-CONV-SAME:      outs(%arg2 : tensor<100x500xf64>)
// F16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @batch_matmul_transpose_a_bf16bf16bf16(%arg0 : tensor<4x250x100xbf16>, %arg1 : tensor<4x250x500xbf16>,
    %arg2 : tensor<4x100x500xbf16>) -> tensor<4x100x500xbf16> {
  %0 = linalg.batch_matmul
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ]
      ins(%arg0, %arg1 : tensor<4x250x100xbf16>, tensor<4x250x500xbf16>)
      outs(%arg2 : tensor<4x100x500xbf16>) -> tensor<4x100x500xbf16>
  util.return %0 : tensor<4x100x500xbf16>
}
// BF16-MATMUL-LABEL: util.func public @batch_matmul_transpose_a_bf16bf16bf16
// BF16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<4x100x500xf32>
// BF16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%arg2 : tensor<4x100x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<4x100x500xf32>)
// BF16-MATMUL:           arith.extf %{{.+}} : bf16 to f32
// BF16-MATMUL:         %[[MATMUL:.+]] = linalg.batch_matmul
// BF16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<4x250x100xbf16>, tensor<4x250x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<4x100x500xf32>)
// BF16-MATMUL:         %[[EMPTY_BF16:.+]] = tensor.empty() : tensor<4x100x500xbf16>
// BF16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<4x100x500xf32>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_BF16]] : tensor<4x100x500xbf16>)
// BF16-MATMUL:           arith.truncf %{{.+}} : f32 to bf16
// BF16-MATMUL:         util.return %[[TRUNCF]]

// BF16-CONV-LABEL: util.func public @batch_matmul_transpose_a_bf16bf16bf16
// BF16-CONV:         %[[MATMUL:.+]] = linalg.batch_matmul
// BF16-CONV-SAME:      ins(%arg0, %arg1 : tensor<4x250x100xbf16>, tensor<4x250x500xbf16>)
// BF16-CONV-SAME:      outs(%arg2 : tensor<4x100x500xbf16>)
// BF16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @batch_matmul_transpose_a_f16f16f16(%arg0 : tensor<4x250x100xf16>, %arg1 : tensor<4x250x500xf16>,
    %arg2 : tensor<4x100x500xf16>) -> tensor<4x100x500xf16> {
  %0 = linalg.batch_matmul
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ]
      ins(%arg0, %arg1 : tensor<4x250x100xf16>, tensor<4x250x500xf16>)
      outs(%arg2 : tensor<4x100x500xf16>) -> tensor<4x100x500xf16>
  util.return %0 : tensor<4x100x500xf16>
}
// F16-MATMUL-LABEL: util.func public @batch_matmul_transpose_a_f16f16f16
// F16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<4x100x500xf32>
// F16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%arg2 : tensor<4x100x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<4x100x500xf32>)
// F16-MATMUL:           arith.extf %{{.+}} : f16 to f32
// F16-MATMUL:         %[[MATMUL:.+]] = linalg.batch_matmul
// F16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<4x250x100xf16>, tensor<4x250x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<4x100x500xf32>)
// F16-MATMUL:         %[[EMPTY_F16:.+]] = tensor.empty() : tensor<4x100x500xf16>
// F16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<4x100x500xf32>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F16]] : tensor<4x100x500xf16>)
// F16-MATMUL:           arith.truncf %{{.+}} : f32 to f16
// F16-MATMUL:         util.return %[[TRUNCF]]

// F16-CONV-LABEL: util.func public @batch_matmul_transpose_a_f16f16f16
// F16-CONV:         %[[MATMUL:.+]] = linalg.batch_matmul
// F16-CONV-SAME:      ins(%arg0, %arg1 : tensor<4x250x100xf16>, tensor<4x250x500xf16>)
// F16-CONV-SAME:      outs(%arg2 : tensor<4x100x500xf16>)
// F16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @batch_matmul_transpose_b_bf16bf16bf16(%arg0 : tensor<4x100x250xbf16>, %arg1 : tensor<4x500x250xbf16>,
    %arg2 : tensor<4x100x500xbf16>) -> tensor<4x100x500xbf16> {
  %0 = linalg.batch_matmul
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ]
      ins(%arg0, %arg1 : tensor<4x100x250xbf16>, tensor<4x500x250xbf16>)
      outs(%arg2 : tensor<4x100x500xbf16>) -> tensor<4x100x500xbf16>
  util.return %0 : tensor<4x100x500xbf16>
}
// BF16-MATMUL-LABEL: util.func public @batch_matmul_transpose_b_bf16bf16bf16
// BF16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<4x100x500xf32>
// BF16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%arg2 : tensor<4x100x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<4x100x500xf32>)
// BF16-MATMUL:           arith.extf %{{.+}} : bf16 to f32
// BF16-MATMUL:         %[[MATMUL:.+]] = linalg.batch_matmul
// BF16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<4x100x250xbf16>, tensor<4x500x250xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<4x100x500xf32>)
// BF16-MATMUL:         %[[EMPTY_BF16:.+]] = tensor.empty() : tensor<4x100x500xbf16>
// BF16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<4x100x500xf32>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_BF16]] : tensor<4x100x500xbf16>)
// BF16-MATMUL:           arith.truncf %{{.+}} : f32 to bf16
// BF16-MATMUL:         util.return %[[TRUNCF]]

// BF16-CONV-LABEL: util.func public @batch_matmul_transpose_b_bf16bf16bf16
// BF16-CONV:         %[[MATMUL:.+]] = linalg.batch_matmul
// BF16-CONV-SAME:      ins(%arg0, %arg1 : tensor<4x100x250xbf16>, tensor<4x500x250xbf16>)
// BF16-CONV-SAME:      outs(%arg2 : tensor<4x100x500xbf16>)
// BF16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @batch_matmul_transpose_b_f16f16f16(%arg0 : tensor<4x100x250xf16>, %arg1 : tensor<4x500x250xf16>,
    %arg2 : tensor<4x100x500xf16>) -> tensor<4x100x500xf16> {
  %0 = linalg.batch_matmul
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ]
      ins(%arg0, %arg1 : tensor<4x100x250xf16>, tensor<4x500x250xf16>)
      outs(%arg2 : tensor<4x100x500xf16>) -> tensor<4x100x500xf16>
  util.return %0 : tensor<4x100x500xf16>
}
// F16-MATMUL-LABEL: util.func public @batch_matmul_transpose_b_f16f16f16
// F16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<4x100x500xf32>
// F16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%arg2 : tensor<4x100x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<4x100x500xf32>)
// F16-MATMUL:           arith.extf %{{.+}} : f16 to f32
// F16-MATMUL:         %[[MATMUL:.+]] = linalg.batch_matmul
// F16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<4x100x250xf16>, tensor<4x500x250xf16>)
// F16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<4x100x500xf32>)
// F16-MATMUL:         %[[EMPTY_F16:.+]] = tensor.empty() : tensor<4x100x500xf16>
// F16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<4x100x500xf32>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F16]] : tensor<4x100x500xf16>)
// F16-MATMUL:           arith.truncf %{{.+}} : f32 to f16
// F16-MATMUL:         util.return %[[TRUNCF]]

// F16-CONV-LABEL: util.func public @batch_matmul_transpose_b_f16f16f16
// F16-CONV:         %[[MATMUL:.+]] = linalg.batch_matmul
// F16-CONV-SAME:      ins(%arg0, %arg1 : tensor<4x100x250xf16>, tensor<4x500x250xf16>)
// F16-CONV-SAME:      outs(%arg2 : tensor<4x100x500xf16>)
// F16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @matmul_transpose_a_bf16bf16bf16(%arg0 : tensor<250x100xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16> {
  %0 = linalg.matmul
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2, d0)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ]
      ins(%arg0, %arg1 : tensor<250x100xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16>
  util.return %0 : tensor<100x500xbf16>
}
// BF16-MATMUL-LABEL: util.func public @matmul_transpose_a_bf16bf16bf16
// BF16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<100x500xf32>
// BF16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%arg2 : tensor<100x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<100x500xf32>)
// BF16-MATMUL:           arith.extf %{{.+}} : bf16 to f32
// BF16-MATMUL:         %[[MATMUL:.+]] = linalg.matmul
// BF16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<250x100xbf16>, tensor<250x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<100x500xf32>)
// BF16-MATMUL:         %[[EMPTY_BF16:.+]] = tensor.empty() : tensor<100x500xbf16>
// BF16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<100x500xf32>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_BF16]] : tensor<100x500xbf16>)
// BF16-MATMUL:           arith.truncf %{{.+}} : f32 to bf16
// BF16-MATMUL:         util.return %[[TRUNCF]]

// BF16-CONV-LABEL: util.func public @matmul_transpose_a_bf16bf16bf16
// BF16-CONV:         %[[MATMUL:.+]] = linalg.matmul
// BF16-CONV-SAME:      ins(%arg0, %arg1 : tensor<250x100xbf16>, tensor<250x500xbf16>)
// BF16-CONV-SAME:      outs(%arg2 : tensor<100x500xbf16>)
// BF16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @matmul_transpose_a_f16f16f16(%arg0 : tensor<250x100xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf16>) -> tensor<100x500xf16> {
  %0 = linalg.matmul
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2, d0)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ]
      ins(%arg0, %arg1 : tensor<250x100xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf16>) -> tensor<100x500xf16>
  util.return %0 : tensor<100x500xf16>
}
// F16-MATMUL-LABEL: util.func public @matmul_transpose_a_f16f16f16
// F16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<100x500xf32>
// F16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%arg2 : tensor<100x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<100x500xf32>)
// F16-MATMUL:           arith.extf %{{.+}} : f16 to f32
// F16-MATMUL:         %[[MATMUL:.+]] = linalg.matmul
// F16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<250x100xf16>, tensor<250x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<100x500xf32>)
// F16-MATMUL:         %[[EMPTY_F16:.+]] = tensor.empty() : tensor<100x500xf16>
// F16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<100x500xf32>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F16]] : tensor<100x500xf16>)
// F16-MATMUL:           arith.truncf %{{.+}} : f32 to f16
// F16-MATMUL:         util.return %[[TRUNCF]]

// F16-CONV-LABEL: util.func public @matmul_transpose_a_f16f16f16
// F16-CONV:         %[[MATMUL:.+]] = linalg.matmul
// F16-CONV-SAME:      ins(%arg0, %arg1 : tensor<250x100xf16>, tensor<250x500xf16>)
// F16-CONV-SAME:      outs(%arg2 : tensor<100x500xf16>)
// F16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @matmul_transpose_b_bf16bf16bf16(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<500x250xbf16>,
    %arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16> {
  %0 = linalg.matmul
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ]
      ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<500x250xbf16>)
      outs(%arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16>
  util.return %0 : tensor<100x500xbf16>
}
// BF16-MATMUL-LABEL: util.func public @matmul_transpose_b_bf16bf16bf16
// BF16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<100x500xf32>
// BF16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%arg2 : tensor<100x500xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<100x500xf32>)
// BF16-MATMUL:           arith.extf %{{.+}} : bf16 to f32
// BF16-MATMUL:         %[[MATMUL:.+]] = linalg.matmul
// BF16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<500x250xbf16>)
// BF16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<100x500xf32>)
// BF16-MATMUL:         %[[EMPTY_BF16:.+]] = tensor.empty() : tensor<100x500xbf16>
// BF16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// BF16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<100x500xf32>)
// BF16-MATMUL-SAME:      outs(%[[EMPTY_BF16]] : tensor<100x500xbf16>)
// BF16-MATMUL:           arith.truncf %{{.+}} : f32 to bf16
// BF16-MATMUL:         util.return %[[TRUNCF]]

// BF16-CONV-LABEL: util.func public @matmul_transpose_b_bf16bf16bf16
// BF16-CONV:         %[[MATMUL:.+]] = linalg.matmul
// BF16-CONV-SAME:      ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<500x250xbf16>)
// BF16-CONV-SAME:      outs(%arg2 : tensor<100x500xbf16>)
// BF16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @matmul_transpose_b_f16f16f16(%arg0 : tensor<100x250xf16>, %arg1 : tensor<500x250xf16>,
    %arg2 : tensor<100x500xf16>) -> tensor<100x500xf16> {
  %0 = linalg.matmul
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ]
      ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<500x250xf16>)
      outs(%arg2 : tensor<100x500xf16>) -> tensor<100x500xf16>
  util.return %0 : tensor<100x500xf16>
}
// F16-MATMUL-LABEL: util.func public @matmul_transpose_b_f16f16f16
// F16-MATMUL:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<100x500xf32>
// F16-MATMUL:         %[[EXTF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%arg2 : tensor<100x500xf16>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F32]] : tensor<100x500xf32>)
// F16-MATMUL:           arith.extf %{{.+}} : f16 to f32
// F16-MATMUL:         %[[MATMUL:.+]] = linalg.matmul
// F16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<500x250xf16>)
// F16-MATMUL-SAME:      outs(%[[EXTF]] : tensor<100x500xf32>)
// F16-MATMUL:         %[[EMPTY_F16:.+]] = tensor.empty() : tensor<100x500xf16>
// F16-MATMUL:         %[[TRUNCF:.+]] = linalg.generic
// F16-MATMUL-SAME:      ins(%[[MATMUL]] : tensor<100x500xf32>)
// F16-MATMUL-SAME:      outs(%[[EMPTY_F16]] : tensor<100x500xf16>)
// F16-MATMUL:           arith.truncf %{{.+}} : f32 to f16
// F16-MATMUL:         util.return %[[TRUNCF]]

// F16-CONV-LABEL: util.func public @matmul_transpose_b_f16f16f16
// F16-CONV:         %[[MATMUL:.+]] = linalg.matmul
// F16-CONV-SAME:      ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<500x250xf16>)
// F16-CONV-SAME:      outs(%arg2 : tensor<100x500xf16>)
// F16-CONV:         util.return %[[MATMUL]]

// -----

util.func public @conv_2d_nchw_fchw_bf16bf16bf16(%arg0 : tensor<1x16x130x130xbf16>, %arg1 : tensor<512x16x3x3xbf16>,
    %arg2 : tensor<1x512x128x128xbf16>) -> tensor<1x512x128x128xbf16> {
    %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
         ins(%arg0, %arg1 : tensor<1x16x130x130xbf16>, tensor<512x16x3x3xbf16>)
         outs(%arg2 : tensor<1x512x128x128xbf16>) -> tensor<1x512x128x128xbf16>
  util.return %0 : tensor<1x512x128x128xbf16>
}
// BF16-MATMUL-LABEL: util.func public @conv_2d_nchw_fchw_bf16bf16bf16
// BF16-MATMUL:         %[[CONV:.+]] = linalg.conv_2d_nchw_fchw
// BF16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<1x16x130x130xbf16>, tensor<512x16x3x3xbf16>)
// BF16-MATMUL-SAME:      outs(%arg2 : tensor<1x512x128x128xbf16>)
// BF16-MATMUL:         util.return %[[CONV]]

// BF16-CONV-LABEL: util.func public @conv_2d_nchw_fchw_bf16bf16bf16
// BF16-CONV:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<1x512x128x128xf32>
// BF16-CONV:         %[[EXTF:.+]] = linalg.generic
// BF16-CONV-SAME:      ins(%arg2 : tensor<1x512x128x128xbf16>)
// BF16-CONV-SAME:      outs(%[[EMPTY_F32]] : tensor<1x512x128x128xf32>)
// BF16-CONV:           arith.extf %{{.+}} : bf16 to f32
// BF16-CONV:         %[[CONV:.+]] = linalg.conv_2d_nchw_fchw
// BF16-CONV-SAME:      ins(%arg0, %arg1 : tensor<1x16x130x130xbf16>, tensor<512x16x3x3xbf16>)
// BF16-CONV-SAME:      outs(%[[EXTF]] : tensor<1x512x128x128xf32>)
// BF16-CONV:         %[[EMPTY_BF16:.+]] = tensor.empty() : tensor<1x512x128x128xbf16>
// BF16-CONV:         %[[TRUNCF:.+]] = linalg.generic
// BF16-CONV-SAME:      ins(%[[CONV]] : tensor<1x512x128x128xf32>)
// BF16-CONV-SAME:      outs(%[[EMPTY_BF16]] : tensor<1x512x128x128xbf16>)
// BF16-CONV:           arith.truncf %{{.+}} : f32 to bf16
// BF16-CONV:         util.return %[[TRUNCF]]

// -----

util.func public @conv_2d_nchw_fchw_f16f16f16(%arg0 : tensor<1x16x130x130xf16>, %arg1 : tensor<512x16x3x3xf16>,
    %arg2 : tensor<1x512x128x128xf16>) -> tensor<1x512x128x128xf16> {
    %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
         ins(%arg0, %arg1 : tensor<1x16x130x130xf16>, tensor<512x16x3x3xf16>)
         outs(%arg2 : tensor<1x512x128x128xf16>) -> tensor<1x512x128x128xf16>
  util.return %0 : tensor<1x512x128x128xf16>
}
// F16-MATMUL-LABEL: util.func public @conv_2d_nchw_fchw_f16f16f16
// F16-MATMUL:         %[[CONV:.+]] = linalg.conv_2d_nchw_fchw
// F16-MATMUL-SAME:      ins(%arg0, %arg1 : tensor<1x16x130x130xf16>, tensor<512x16x3x3xf16>)
// F16-MATMUL-SAME:      outs(%arg2 : tensor<1x512x128x128xf16>)
// F16-MATMUL:         util.return %[[CONV]]

// F16-CONV-LABEL: util.func public @conv_2d_nchw_fchw_f16f16f16
// F16-CONV:         %[[EMPTY_F32:.+]] = tensor.empty() : tensor<1x512x128x128xf32>
// F16-CONV:         %[[EXTF:.+]] = linalg.generic
// F16-CONV-SAME:      ins(%arg2 : tensor<1x512x128x128xf16>)
// F16-CONV-SAME:      outs(%[[EMPTY_F32]] : tensor<1x512x128x128xf32>)
// F16-CONV:           arith.extf %{{.+}} : f16 to f32
// F16-CONV:         %[[CONV:.+]] = linalg.conv_2d_nchw_fchw
// F16-CONV-SAME:      ins(%arg0, %arg1 : tensor<1x16x130x130xf16>, tensor<512x16x3x3xf16>)
// F16-CONV-SAME:      outs(%[[EXTF]] : tensor<1x512x128x128xf32>)
// F16-CONV:         %[[EMPTY_F16:.+]] = tensor.empty() : tensor<1x512x128x128xf16>
// F16-CONV:         %[[TRUNCF:.+]] = linalg.generic
// F16-CONV-SAME:      ins(%[[CONV]] : tensor<1x512x128x128xf32>)
// F16-CONV-SAME:      outs(%[[EMPTY_F16]] : tensor<1x512x128x128xf16>)
// F16-CONV:           arith.truncf %{{.+}} : f32 to f16
// F16-CONV:         util.return %[[TRUNCF]]
