// Keys [0, 8) and [16, 24) are hidden in every row and query row 2 sees no
// key, so llvm-cpu skips whole K/V tiles and row 2 is zero, as in Torch.
func.func @sdpa_hidden_row() {
  %q = util.unfoldable_constant dense<[[[[-0.54, -0.11, -0.29, 0.16, -0.17, -0.03], [0.87, 0.16, 0.4, 0.45, -0.29, -0.98], [0.47, 0.15, 0.76, 0.78, 0.07, 0.48], [-0.11, 0.9, 0.55, -0.89, -0.52, -0.85]], [[-0.26, 0.65, 0.97, 0.12, -0.46, -0.58], [-0.82, 0.72, 0.41, 0.64, 0.82, 0.61], [-0.3, -0.68, -0.94, 0.73, 0.87, 0.84], [0.91, -0.57, -0.59, 0.26, -0.22, 0.38]], [[-0.28, 0.44, -0.6, 0.41, 0.43, -0.32], [0.33, -0.38, 0.53, 0.36, 0.12, 0.03], [-0.57, 0.73, 0.39, 0.05, 0.93, 0.93], [0.69, 0.83, -0.51, 0.08, -0.75, -0.63]]], [[[0.65, -0.41, 0.54, -0.1, 0.08, 0.21], [-0.17, -0.25, -0.88, 0.42, -0.01, -0.68], [-0.61, 0.35, 0.12, 0.42, -0.23, -0.81], [-0.56, 0.91, -0.8, 0.58, -0.47, -0.41]], [[-0.8, -0.37, -0.95, -0.06, -0.84, 0.33], [-0.45, 0.29, -0.73, 0.99, -0.13, 0.64], [-0.32, -0.43, -0.48, 0.97, -0.84, -0.51], [-0.65, -0.19, -0.63, -0.47, 0.91, 0.0]], [[-0.32, -0.06, 0.62, 0.71, 0.45, 0.24], [0.76, -0.99, -0.74, 0.18, 0.03, 0.23], [0.93, -0.35, -0.04, 0.31, -0.26, -0.04], [-0.76, 0.88, -0.51, -0.17, -0.73, -0.24]]]]> : tensor<2x3x4x6xf32>
  %k = util.unfoldable_constant dense<[[[[-0.96, 0.84, -0.24, -0.07, -0.45, 0.4], [-0.47, 0.69, -0.52, -0.06, 0.57, 0.61], [-0.86, -0.78, -0.07, -0.53, -0.08, -0.93], [0.65, -0.86, 0.45, 0.16, -0.32, 0.14], [-0.29, -0.89, -0.27, -0.99, 0.85, -0.58], [-0.14, 0.98, 0.58, 0.05, 0.13, -1.0], [-0.63, -0.44, -0.44, -0.28, -0.33, 0.52], [0.96, 0.74, -0.71, 0.64, -0.52, 0.97], [0.21, -0.37, 0.49, -0.6, -0.73, 0.02], [-0.74, -0.45, 0.53, 0.54, -0.71, -0.18], [0.88, -0.87, -0.76, 0.18, 0.11, 0.61], [0.28, 0.19, 0.37, 0.49, -0.66, 0.56], [-0.88, 0.6, -0.64, 0.5, -0.56, 0.42], [-0.81, -0.13, -0.04, -0.99, 0.79, -0.11], [-0.43, 0.81, 0.94, 0.75, 0.57, 0.99], [-0.4, -0.5, 0.28, -0.23, 0.46, -0.97], [0.79, 0.05, 0.39, -0.92, 0.8, 0.63], [0.48, 0.49, 0.94, 0.29, -0.23, 0.23], [-0.91, -0.81, -0.8, -0.78, 0.8, 0.63], [0.21, 0.02, 0.21, -0.53, -0.05, 0.98], [0.07, -0.61, -0.76, -0.88, -0.61, 0.34], [0.7, -0.22, -0.77, 0.09, 0.48, 0.85], [-0.28, 0.1, -0.14, 0.3, 0.41, -0.96], [0.23, -0.0, 0.49, 0.79, 0.32, -0.86]], [[-0.23, -0.14, -0.82, 0.49, 0.9, 0.6], [0.97, -0.46, 0.34, 0.6, 0.41, 0.14], [0.56, -0.2, -0.3, 0.34, -0.71, -0.76], [0.1, 0.34, -0.62, 0.7, -0.26, -0.41], [0.53, 0.32, 0.59, -0.22, -0.38, 0.97], [-0.2, 0.19, 0.43, -0.71, 0.31, -0.13], [-0.94, 0.91, -0.67, -0.57, -0.98, 0.97], [-0.89, -0.01, -0.57, 0.2, -0.1, -0.51], [-0.62, 0.12, 0.71, -0.71, -0.93, 0.23], [0.09, 0.91, -0.61, -0.51, -0.32, -0.04], [-0.56, 0.93, -0.01, 0.97, 0.03, 0.84], [0.94, -0.23, -0.98, 0.57, 0.38, -0.13], [0.72, 0.79, 0.02, -0.25, -0.38, 0.45], [0.53, 0.91, -0.03, -0.44, -0.67, -0.56], [0.36, -0.29, -0.55, -0.35, 0.69, -0.65], [-0.82, -0.59, -0.55, -0.49, -0.17, 0.94], [0.14, -0.01, 0.14, -0.85, -0.02, 0.47], [0.44, -0.71, -0.85, 0.96, 0.4, -0.7], [-0.07, -0.35, -0.6, -0.1, 0.66, -0.65], [-0.5, 0.76, -0.35, -0.27, -0.22, -0.32], [-0.73, 0.78, -0.65, 0.82, 0.54, -0.15], [0.58, 0.06, 0.78, 0.24, -0.59, 0.04], [-0.58, 0.18, 0.4, 0.14, -0.13, -0.29], [-0.99, -0.86, 0.76, 0.32, 0.87, 0.44]], [[0.55, 0.21, -0.38, 0.83, -0.31, 0.12], [0.59, -0.03, -0.94, -0.54, 0.86, 0.5], [0.39, 0.79, 0.87, 0.32, 0.2, 0.99], [0.09, -0.92, 0.11, -0.38, -0.24, 0.66], [-0.51, -0.72, 0.73, 0.85, 0.05, 0.31], [0.91, 0.8, 0.1, 0.41, -0.58, 0.23], [0.9, 0.82, 0.51, -0.88, 0.52, 0.28], [-0.45, 0.98, -0.17, 0.77, 0.18, -0.94], [-0.19, -0.56, 0.16, 0.82, -0.41, -0.26], [-0.04, 0.72, 0.12, 0.91, 0.45, -0.33], [0.26, 0.51, -0.89, -0.24, 0.45, -0.79], [-0.11, 0.03, 0.91, -0.84, -0.08, -0.78], [-0.13, 0.35, -0.16, 0.16, -0.79, 0.05], [0.08, 0.79, -0.56, -0.65, 0.64, -0.96], [0.63, -0.7, 0.55, -0.02, 0.93, 0.56], [0.11, -0.1, 0.94, 0.3, -0.59, -0.39], [0.05, -0.63, -0.39, 0.91, -0.13, 0.95], [-0.35, -0.7, 0.83, -0.9, 0.23, -0.77], [-0.58, -0.38, -0.53, -0.47, 0.3, 0.53], [0.51, 0.25, -0.7, 0.86, -0.02, -0.8], [-0.55, -0.64, -0.78, -0.16, -0.05, -0.69], [-0.5, 0.7, -0.17, -0.42, -0.28, 0.58], [-0.84, 0.92, 0.26, 0.93, 0.18, 0.01], [0.99, 0.55, 0.91, 0.84, -0.75, -0.04]]], [[[0.71, 0.58, 0.1, 0.79, -0.79, -0.97], [0.28, 0.25, -0.5, -0.26, -0.41, 0.43], [-0.29, -0.24, -0.2, -0.77, -0.45, -0.32], [0.79, 0.6, 0.96, -0.44, 0.87, 0.78], [0.46, -0.7, 0.14, 0.68, 0.44, 0.4], [0.37, -0.46, 0.18, 0.2, -0.21, -0.55], [-0.67, -0.8, -0.57, 0.98, -0.26, 0.03], [-0.47, 0.26, 0.56, 0.09, 0.37, -0.01], [0.2, 0.97, -0.49, -0.23, 0.52, -0.48], [-0.49, -0.53, -0.75, 0.83, 0.41, -0.68], [-0.61, 0.66, 0.42, -0.7, -0.56, 0.69], [-0.5, 0.53, -0.62, -0.5, 0.66, -0.45], [0.47, 0.49, -0.76, 0.92, 0.05, 0.91], [-0.64, 0.35, -0.47, -0.05, 0.44, -0.14], [0.34, 0.2, 0.02, -0.36, -0.85, -0.63], [-0.86, 0.3, -0.8, 0.27, 0.44, -0.84], [-0.67, 0.99, 0.41, 0.77, 0.84, -0.19], [0.86, -0.8, -0.9, 0.34, -0.33, -0.31], [-0.26, -0.86, -0.82, 0.25, 0.76, 0.84], [-0.58, -0.32, 0.38, -0.59, 0.55, 0.71], [-0.43, 0.99, 0.51, -0.49, -0.45, -0.82], [0.05, 0.01, -0.93, -0.59, -0.43, -0.17], [0.26, 0.48, 0.27, -0.38, -0.09, -0.91], [0.26, -0.51, 0.36, -0.37, 0.65, -0.11]], [[-0.15, 0.77, -0.53, -0.26, -0.79, 0.72], [-0.48, -0.83, 0.12, -0.14, 0.03, -0.55], [-0.77, 0.78, -0.64, 0.77, 0.48, -0.36], [-0.53, -0.04, 0.17, -0.11, 0.57, -0.97], [0.6, 0.35, 0.47, -0.95, 0.06, 0.57], [-0.43, -0.12, -0.59, 0.21, 0.37, -0.61], [-0.22, 0.45, -0.23, 0.52, 0.58, -0.76], [0.43, 0.61, 0.24, -0.54, -0.17, -1.0], [-0.63, 0.26, -0.43, 0.83, 0.97, 0.22], [-0.09, 0.33, 0.82, 0.45, -0.9, -0.71], [-0.8, -0.11, -0.03, 0.96, -0.54, -0.01], [-0.36, 0.48, -0.04, 0.13, -0.57, 0.97], [0.81, 0.81, -0.33, 0.83, -0.21, 0.92], [0.68, 0.27, -0.79, 0.12, -0.26, -0.9], [-0.93, 0.77, -0.63, -0.88, -0.56, -0.8], [0.37, -0.25, 0.64, 0.43, 0.52, -0.98], [-0.44, -0.05, 0.31, -0.65, -0.79, -0.41], [-0.88, 0.95, -0.3, -0.87, 0.97, -0.96], [0.67, -0.9, -0.48, -0.39, -0.53, 0.72], [0.39, -0.45, 0.14, -0.52, 0.48, -0.06], [0.97, -0.0, 0.58, 0.72, 0.8, -0.98], [0.29, 0.61, 0.15, 0.09, -0.72, 0.07], [0.6, 0.87, 0.31, 0.9, 0.0, 0.45], [0.64, -0.4, 0.43, -0.44, 0.71, 0.9]], [[-0.64, -0.6, 0.23, -0.13, 0.73, -0.79], [-0.34, 0.45, -0.53, 0.32, -0.92, 1.0], [0.67, -0.08, -0.49, -0.96, -0.34, 0.9], [-0.14, -0.96, 0.54, 0.79, -0.77, -0.9], [-0.46, -0.56, 0.44, -0.84, -0.21, 0.94], [0.45, 0.66, 0.75, -0.03, 0.34, 0.92], [-0.8, -0.3, 0.87, -0.29, 0.63, -0.51], [0.95, 0.62, 0.74, -0.44, 0.97, 0.18], [-0.49, 0.95, 0.66, 0.59, 0.56, 0.62], [-0.43, 0.41, 0.95, 0.86, -0.54, -0.04], [0.96, -0.46, -0.9, -0.84, -0.71, 0.14], [-0.12, -0.32, -0.57, -0.18, 0.45, -0.46], [0.11, 0.37, 0.21, -0.0, -0.98, -0.16], [-0.36, 0.14, -0.74, 0.53, 0.03, 0.1], [-0.96, 0.94, 0.39, -0.44, -0.53, -0.01], [0.42, -0.28, -0.93, 0.55, 0.69, 0.5], [-0.41, -0.5, 0.61, 0.03, 0.39, -0.82], [-0.38, 0.82, -0.97, 0.89, 0.35, -0.19], [-0.66, 0.89, -0.95, 0.36, 0.48, 0.84], [-0.37, -0.91, -0.02, -0.36, 0.73, -0.96], [-0.27, -0.81, 0.46, -0.01, -0.57, -0.74], [-0.12, 0.17, -0.65, -0.54, -0.03, -0.34], [-0.1, -0.19, 0.38, -0.99, 0.99, 0.27], [0.46, -0.42, 0.22, 0.38, -0.35, 0.93]]]]> : tensor<2x3x24x6xf32>
  %v = util.unfoldable_constant dense<[[[[-0.04, 0.54, 0.82, -0.35, -0.5], [0.68, -0.25, -0.72, 0.97, 0.48], [0.04, 0.62, 0.84, 0.24, -0.61], [-0.91, -0.98, 0.67, -0.67, 0.33], [0.86, -0.94, 0.78, -0.02, 0.19], [-0.39, -0.77, -0.69, -0.78, -0.07], [1.0, 0.72, -0.79, -0.28, -0.33], [0.86, 0.1, 0.4, -0.41, 1.0], [-0.91, -0.87, 0.79, -0.97, -0.44], [-0.32, -0.95, 0.55, 0.58, -0.24], [-0.81, -0.85, -0.18, -0.47, -0.25], [-0.36, -0.18, -0.64, -0.42, -0.28], [-0.61, -0.5, -0.53, 0.34, 0.14], [-0.73, -0.82, -0.33, -0.02, 0.44], [0.32, 0.0, -0.31, 0.63, 0.6], [0.02, -0.95, 0.42, 0.44, 0.92], [0.45, -0.07, -0.06, -0.72, 0.9], [-0.81, -0.11, -0.87, 0.22, -0.93], [-0.7, -0.65, 0.82, 0.46, 0.32], [0.93, -0.72, -0.05, 0.45, -0.29], [0.85, 0.37, 0.26, 0.21, -0.47], [0.31, -0.58, -0.95, -0.76, -0.85], [-0.54, 0.14, 0.54, 0.64, -0.52], [-0.44, -0.98, 0.41, -0.54, 0.9]], [[-0.36, 0.31, 0.71, -0.83, 0.17], [-0.57, 0.59, -0.28, 0.03, -0.16], [-0.71, 0.35, 0.99, -0.03, 0.39], [0.36, 0.95, 0.01, -0.99, -0.73], [0.42, 0.63, -0.87, 0.44, 0.22], [0.44, 0.12, -0.32, -0.99, -0.09], [0.16, 0.73, -0.2, 0.26, -0.24], [0.2, -0.26, 0.01, 0.99, 0.6], [0.09, 0.68, 0.77, -0.19, -0.31], [0.33, 0.45, 0.95, -0.15, -0.15], [0.62, 0.28, -0.62, 0.78, -0.77], [0.18, -0.82, 0.85, 0.21, 0.24], [0.68, -0.67, -0.62, 0.65, 0.51], [0.08, 0.98, -0.4, -0.51, 0.23], [0.14, -0.52, -0.8, 0.48, 0.3], [0.21, -0.28, -0.37, 0.18, 0.69], [-0.35, 0.1, -0.4, -0.37, 0.86], [-0.91, 0.88, 0.03, 0.37, -0.77], [0.44, -0.21, -0.31, -0.76, -0.23], [-0.7, 0.31, -0.66, -0.25, 0.82], [0.78, -0.47, -0.99, -0.26, -0.17], [0.48, -0.39, 0.08, -0.59, -0.03], [-0.58, -0.77, 0.1, -0.19, 0.78], [0.89, 0.41, -0.56, 0.38, -0.21]], [[0.2, 0.37, 0.55, -0.94, 0.65], [-0.94, -0.24, 0.56, -0.8, -0.75], [0.95, -0.25, 0.2, 0.34, -0.51], [0.33, 0.47, -0.74, 0.69, 0.79], [0.47, -0.92, 0.83, -0.51, -0.86], [0.8, 0.54, 0.82, -0.22, 0.73], [-0.58, 0.08, -0.27, -0.7, 0.91], [0.91, 0.46, -0.13, 0.71, 0.38], [-0.99, 0.38, -0.19, 0.39, 0.23], [0.69, 0.47, 0.93, 0.4, -0.95], [-0.4, 0.86, 0.9, -0.57, 0.88], [0.31, 0.01, 0.63, -0.61, 0.21], [0.11, -0.94, -0.32, 0.13, 0.15], [0.53, -0.16, -0.9, 0.42, 0.12], [0.48, 0.33, -0.24, -0.09, 0.35], [0.94, -0.42, -0.51, -0.03, 0.87], [0.69, -0.38, 0.55, -0.09, 0.64], [0.2, -0.62, -0.61, -0.24, -0.08], [0.78, 0.46, 0.11, 0.15, -0.48], [0.61, 0.65, 0.44, -0.48, 0.37], [-0.97, 0.8, -0.58, -0.67, 0.23], [0.6, 0.12, -0.43, -0.95, 0.5], [-0.86, 0.91, -0.94, -0.65, -0.57], [0.91, -0.61, 0.3, -0.08, -0.79]]], [[[-0.07, 0.04, 0.09, -0.44, 0.32], [-0.19, 0.27, 0.66, 0.25, -0.16], [-0.44, -0.97, 0.41, -0.04, 0.2], [-0.76, 0.47, -0.56, 0.84, 0.83], [-0.04, -0.12, -0.69, 0.74, -0.73], [0.19, 0.92, 0.79, -0.46, -0.57], [0.92, -0.8, -0.41, 0.24, 0.44], [-0.59, -0.41, 0.88, -0.67, -0.63], [0.25, -0.82, 0.99, 0.46, -0.86], [-0.84, -0.68, -0.53, -0.25, 0.3], [-0.63, 0.74, 0.1, -0.18, 0.78], [0.47, 0.7, 0.82, -0.77, -0.67], [-0.82, -0.9, 0.41, -0.42, -0.55], [-0.18, 0.74, -0.18, 0.23, 0.38], [0.09, 0.81, 0.96, -0.6, -0.14], [-0.97, -0.53, -0.29, 0.24, -0.7], [0.42, -0.5, 0.95, 0.89, 0.72], [0.97, -0.26, 0.08, -0.47, 0.09], [-0.67, 0.27, 0.6, 0.83, 0.81], [-0.49, 0.53, -0.76, -0.44, 0.07], [0.96, -0.7, 0.83, -0.91, -0.94], [0.91, 0.92, -0.54, -1.0, 0.08], [-0.37, -0.43, 0.23, -0.94, 0.68], [0.24, -0.53, 0.06, 0.02, -0.27]], [[0.25, 0.41, 0.73, -0.56, 0.07], [0.34, -0.43, 0.97, 0.93, 0.43], [-0.99, -0.97, -0.64, 0.07, 0.46], [-0.75, 0.22, -0.05, 0.79, 0.99], [-0.84, 0.49, -0.38, 0.58, 0.43], [-0.53, 0.66, -0.42, 0.83, 0.6], [0.19, -0.33, -0.98, 0.61, 0.74], [-0.24, -0.16, 0.87, -0.53, -0.94], [-0.77, 0.86, 0.41, -0.77, -0.37], [0.93, -0.6, 0.9, -0.04, -0.49], [0.12, 0.06, -0.99, 0.7, -0.67], [-0.96, 0.62, -0.43, -0.16, -0.38], [-0.75, 0.29, -0.26, 0.02, 0.54], [0.27, 0.12, -0.46, 0.22, -0.7], [0.07, 0.02, -0.27, -0.35, -0.94], [0.81, 0.48, -0.11, 0.71, -0.06], [0.27, 0.04, 0.1, 0.74, -0.8], [-0.77, 0.08, -0.18, -0.79, -0.3], [0.36, -0.42, 0.87, 0.48, 0.07], [0.87, -0.74, 0.1, 0.98, 0.13], [-0.84, -0.33, 0.78, -0.83, 0.38], [0.95, -0.88, -0.54, 0.71, -0.68], [-0.65, -0.74, 0.64, -0.37, 0.42], [0.22, -0.6, 0.53, -0.08, 0.93]], [[0.54, 0.69, -0.29, 0.92, 0.28], [-0.66, -0.44, 0.37, -0.68, -0.87], [-0.81, 0.87, -0.46, -0.01, 0.23], [0.99, 0.32, 0.15, -0.29, 0.6], [0.87, 0.86, 0.69, 0.46, 0.76], [0.84, -0.92, 0.54, 0.05, 0.02], [-0.41, -0.07, -0.95, 0.42, -0.69], [0.11, -0.22, 0.51, 0.36, 0.5], [-0.36, -0.13, 0.49, 0.21, -0.11], [-0.2, 0.3, -0.06, 0.06, 0.22], [0.72, 0.28, -0.84, 0.49, -0.79], [-0.61, 0.84, -0.92, -0.43, -0.36], [0.69, -0.19, 0.13, -0.45, -0.44], [0.58, 0.54, 0.06, -0.46, -0.62], [-0.39, 0.29, 0.3, 0.75, -0.11], [0.47, -0.65, 0.5, 0.22, -0.18], [0.16, 0.33, 0.62, 0.74, 0.41], [-0.36, 0.28, 0.06, 0.02, 0.19], [-0.45, -0.23, 0.39, 0.95, 0.1], [-0.38, 0.86, -0.28, 0.4, -0.73], [-0.13, -0.47, 0.37, 0.43, -0.93], [0.19, -0.35, -0.16, 0.19, 1.0], [0.33, -0.47, -0.86, 0.51, -0.89], [-0.24, -0.95, 0.24, 0.12, -0.73]]]]> : tensor<2x3x24x5xf32>
  %mask = util.unfoldable_constant dense<[[false, false, false, false, false, false, false, false, true, true, false, true, true, true, true, true, false, false, false, false, false, false, false, false], [false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false], [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false], [false, false, false, false, false, false, false, false, true, false, true, true, true, false, true, true, false, false, false, false, false, false, false, false]]> : tensor<4x24xi1>
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v, %mask {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_hidden_row_impl, version = 1 : i32
  } : (tensor<2x3x4x6xf32>, tensor<2x3x24x6xf32>, tensor<2x3x24x5xf32>, tensor<4x24xi1>) -> tensor<2x3x4x5xf32>
  check.expect_almost_eq_const(%0, dense<[[[[-0.381007, -0.623348, -0.0180102, 0.120628, 0.161734], [-0.413676, -0.659197, 0.0367323, -0.0196068, 0.082767], [0.0, 0.0, 0.0, 0.0, 0.0], [-0.38884, -0.592936, 0.0395384, -0.0996527, 0.137262]], [[0.236085, 0.179025, 0.085305, 0.00882226, 0.144391], [0.364334, 0.0598521, -0.141506, 0.308993, -0.0738807], [0.0, 0.0, 0.0, 0.0, 0.0], [0.308398, -0.374712, -0.0596949, 0.355467, 0.190129]], [[0.313396, -0.0240583, -0.0801546, 0.159604, 0.0374027], [0.228274, 0.0738112, 0.0199338, 0.00940709, 0.237657], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0254133, 0.0237838, 0.115618, -0.160212, 0.485742]]], [[[-0.269168, -0.0866617, 0.363909, -0.189712, -0.312891], [-0.385616, -0.111194, 0.188841, -0.139963, -0.22779], [0.0, 0.0, 0.0, 0.0, 0.0], [-0.309118, -0.0982828, 0.452891, -0.167222, -0.435376]], [[-0.166569, 0.256563, -0.0871266, -0.13798, -0.427137], [-0.237079, 0.302112, -0.203309, -0.00843084, -0.328699], [0.0, 0.0, 0.0, 0.0, 0.0], [-0.260857, 0.416744, -0.193635, -0.0967381, -0.398968]], [[-0.0207251, 0.11992, 0.116686, 0.0234637, -0.182912], [0.267037, 0.12474, -0.159436, 0.0533358, -0.395079], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0320183, 0.111431, 8.79138e-05, 0.189668, -0.306343]]]]> : tensor<2x3x4x5xf32>) : tensor<2x3x4x5xf32>
  return
}

// Unused: the composite is rewritten to attention before decompositions run.
func.func private @sdpa_hidden_row_impl(%q: tensor<2x3x4x6xf32>, %k: tensor<2x3x24x6xf32>, %v: tensor<2x3x24x5xf32>, %m: tensor<4x24xi1>) -> tensor<2x3x4x5xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>
}

// Causal exclusion must follow score scaling, including zero/negative scales.
func.func @causal_dynamic_zero() {
  %q = flow.tensor.dynamic_constant dense<[[1.0], [2.0]]> : tensor<2x1xf32> -> tensor<?x1xf32>
  %k = util.unfoldable_constant dense<[[1.0], [3.0]]> : tensor<2x1xf32>
  %v = util.unfoldable_constant dense<[[2.0], [10.0]]> : tensor<2x1xf32>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = true, scale = 0.0 : f32},
    decomposition = @causal_dynamic_zero_impl, version = 1 : i32
  } : (tensor<?x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<?x1xf32>
  %barrier = util.optimization_barrier %result : tensor<?x1xf32>
  %checked = tensor.cast %barrier : tensor<?x1xf32> to tensor<2x1xf32>
  check.expect_almost_eq_const(%checked, dense<[[2.0], [6.0000000]]> : tensor<2x1xf32>) : tensor<2x1xf32>
  return
}

// This fallback computes the same two-row causal attention for finite inputs;
// the specialization is required by the compiler-level handler tests.
func.func private @causal_dynamic_zero_impl(%q: tensor<?x1xf32>, %k: tensor<2x1xf32>, %v: tensor<2x1xf32>) -> tensor<?x1xf32> {
  %scale = stablehlo.constant dense<0.0> : tensor<f32>
  %scores = stablehlo.dot_general %q, %k, contracting_dims = [1] x [1] : (tensor<?x1xf32>, tensor<2x1xf32>) -> tensor<?x2xf32>
  %factor = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
  %static_scores = tensor.cast %scores : tensor<?x2xf32> to tensor<2x2xf32>
  %scaled = stablehlo.multiply %static_scores, %factor : tensor<2x2xf32>
  %mask = stablehlo.constant dense<[[true, false], [true, true]]> : tensor<2x2xi1>
  %negative_inf = stablehlo.constant dense<0xFF800000> : tensor<2x2xf32>
  %masked = stablehlo.select %mask, %scaled, %negative_inf : tensor<2x2xi1>, tensor<2x2xf32>
  %exp = stablehlo.exponential %masked : tensor<2x2xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %sums = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
  %denominator = stablehlo.broadcast_in_dim %sums, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
  %probability = stablehlo.divide %exp, %denominator : tensor<2x2xf32>
  %output = stablehlo.dot_general %probability, %v, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %result = tensor.cast %output : tensor<2x1xf32> to tensor<?x1xf32>
  return %result : tensor<?x1xf32>
}

// Causal exclusion must follow score scaling, including zero/negative scales.
func.func @causal_dynamic_negative() {
  %q = flow.tensor.dynamic_constant dense<[[1.0], [2.0]]> : tensor<2x1xf32> -> tensor<?x1xf32>
  %k = util.unfoldable_constant dense<[[1.0], [3.0]]> : tensor<2x1xf32>
  %v = util.unfoldable_constant dense<[[2.0], [10.0]]> : tensor<2x1xf32>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = true, scale = -1.0 : f32},
    decomposition = @causal_dynamic_negative_impl, version = 1 : i32
  } : (tensor<?x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<?x1xf32>
  %barrier = util.optimization_barrier %result : tensor<?x1xf32>
  %checked = tensor.cast %barrier : tensor<?x1xf32> to tensor<2x1xf32>
  check.expect_almost_eq_const(%checked, dense<[[2.0], [2.1438897]]> : tensor<2x1xf32>) : tensor<2x1xf32>
  return
}

// This fallback computes the same two-row causal attention for finite inputs;
// the specialization is required by the compiler-level handler tests.
func.func private @causal_dynamic_negative_impl(%q: tensor<?x1xf32>, %k: tensor<2x1xf32>, %v: tensor<2x1xf32>) -> tensor<?x1xf32> {
  %scale = stablehlo.constant dense<-1.0> : tensor<f32>
  %scores = stablehlo.dot_general %q, %k, contracting_dims = [1] x [1] : (tensor<?x1xf32>, tensor<2x1xf32>) -> tensor<?x2xf32>
  %factor = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
  %static_scores = tensor.cast %scores : tensor<?x2xf32> to tensor<2x2xf32>
  %scaled = stablehlo.multiply %static_scores, %factor : tensor<2x2xf32>
  %mask = stablehlo.constant dense<[[true, false], [true, true]]> : tensor<2x2xi1>
  %negative_inf = stablehlo.constant dense<0xFF800000> : tensor<2x2xf32>
  %masked = stablehlo.select %mask, %scaled, %negative_inf : tensor<2x2xi1>, tensor<2x2xf32>
  %exp = stablehlo.exponential %masked : tensor<2x2xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %sums = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
  %denominator = stablehlo.broadcast_in_dim %sums, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
  %probability = stablehlo.divide %exp, %denominator : tensor<2x2xf32>
  %output = stablehlo.dot_general %probability, %v, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %result = tensor.cast %output : tensor<2x1xf32> to tensor<?x1xf32>
  return %result : tensor<?x1xf32>
}

// Causal exclusion must follow score scaling, including zero/negative scales.
func.func @causal_dynamic_positive() {
  %q = flow.tensor.dynamic_constant dense<[[1.0], [2.0]]> : tensor<2x1xf32> -> tensor<?x1xf32>
  %k = util.unfoldable_constant dense<[[1.0], [3.0]]> : tensor<2x1xf32>
  %v = util.unfoldable_constant dense<[[2.0], [10.0]]> : tensor<2x1xf32>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = true, scale = 1.0 : f32},
    decomposition = @causal_dynamic_positive_impl, version = 1 : i32
  } : (tensor<?x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<?x1xf32>
  %barrier = util.optimization_barrier %result : tensor<?x1xf32>
  %checked = tensor.cast %barrier : tensor<?x1xf32> to tensor<2x1xf32>
  check.expect_almost_eq_const(%checked, dense<[[2.0], [9.8561103]]> : tensor<2x1xf32>) : tensor<2x1xf32>
  return
}

// This fallback computes the same two-row causal attention for finite inputs;
// the specialization is required by the compiler-level handler tests.
func.func private @causal_dynamic_positive_impl(%q: tensor<?x1xf32>, %k: tensor<2x1xf32>, %v: tensor<2x1xf32>) -> tensor<?x1xf32> {
  %scale = stablehlo.constant dense<1.0> : tensor<f32>
  %scores = stablehlo.dot_general %q, %k, contracting_dims = [1] x [1] : (tensor<?x1xf32>, tensor<2x1xf32>) -> tensor<?x2xf32>
  %factor = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
  %static_scores = tensor.cast %scores : tensor<?x2xf32> to tensor<2x2xf32>
  %scaled = stablehlo.multiply %static_scores, %factor : tensor<2x2xf32>
  %mask = stablehlo.constant dense<[[true, false], [true, true]]> : tensor<2x2xi1>
  %negative_inf = stablehlo.constant dense<0xFF800000> : tensor<2x2xf32>
  %masked = stablehlo.select %mask, %scaled, %negative_inf : tensor<2x2xi1>, tensor<2x2xf32>
  %exp = stablehlo.exponential %masked : tensor<2x2xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %sums = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
  %denominator = stablehlo.broadcast_in_dim %sums, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
  %probability = stablehlo.divide %exp, %denominator : tensor<2x2xf32>
  %output = stablehlo.dot_general %probability, %v, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %result = tensor.cast %output : tensor<2x1xf32> to tensor<?x1xf32>
  return %result : tensor<?x1xf32>
}

// Causal exclusion must follow score scaling, including zero/negative scales.
func.func @causal_static_negative() {
  %q = util.unfoldable_constant dense<[[1.0], [2.0]]> : tensor<2x1xf32>
  %k = util.unfoldable_constant dense<[[1.0], [3.0]]> : tensor<2x1xf32>
  %v = util.unfoldable_constant dense<[[2.0], [10.0]]> : tensor<2x1xf32>
  %result = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = true, scale = -1.0 : f32},
    decomposition = @causal_static_negative_impl, version = 1 : i32
  } : (tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  check.expect_almost_eq_const(%result, dense<[[2.0], [2.1438897]]> : tensor<2x1xf32>) : tensor<2x1xf32>
  return
}

// This fallback computes the same two-row causal attention for finite inputs;
// the specialization is required by the compiler-level handler tests.
func.func private @causal_static_negative_impl(%q: tensor<2x1xf32>, %k: tensor<2x1xf32>, %v: tensor<2x1xf32>) -> tensor<2x1xf32> {
  %scale = stablehlo.constant dense<-1.0> : tensor<f32>
  %scores = stablehlo.dot_general %q, %k, contracting_dims = [1] x [1] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  %factor = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
  %static_scores = tensor.cast %scores : tensor<2x2xf32> to tensor<2x2xf32>
  %scaled = stablehlo.multiply %static_scores, %factor : tensor<2x2xf32>
  %mask = stablehlo.constant dense<[[true, false], [true, true]]> : tensor<2x2xi1>
  %negative_inf = stablehlo.constant dense<0xFF800000> : tensor<2x2xf32>
  %masked = stablehlo.select %mask, %scaled, %negative_inf : tensor<2x2xi1>, tensor<2x2xf32>
  %exp = stablehlo.exponential %masked : tensor<2x2xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %sums = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
  %denominator = stablehlo.broadcast_in_dim %sums, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
  %probability = stablehlo.divide %exp, %denominator : tensor<2x2xf32>
  %output = stablehlo.dot_general %probability, %v, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %result = tensor.cast %output : tensor<2x1xf32> to tensor<2x1xf32>
  return %result : tensor<2x1xf32>
}
