// Regression testcase from https://github.com/openxla/iree/issues/12060

func.func @matmul_i8(%lhs: tensor<?x?xi8>, %rhs: tensor<?x?xi8>, %acc: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %result1 = linalg.matmul ins(%lhs, %rhs: tensor<?x?xi8>, tensor<?x?xi8>) outs(%acc: tensor<?x?xi32>) -> tensor<?x?xi32>
  %result2 = linalg.matmul ins(%lhs, %rhs: tensor<?x?xi8>, tensor<?x?xi8>) outs(%result1: tensor<?x?xi32>) -> tensor<?x?xi32>
  %result3 = linalg.matmul ins(%lhs, %rhs: tensor<?x?xi8>, tensor<?x?xi8>) outs(%result2: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %result3: tensor<?x?xi32>
}
