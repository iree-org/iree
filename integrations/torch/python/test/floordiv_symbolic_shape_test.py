import sys

import pytest
import torch
from iree.importers.torch.exporter import export


def test_floordiv_symbolic_shape():
    """Test that FloorDiv in symbolic shapes is handled correctly.

    Conv1d with stride > 1 creates output shape: (input_len - kernel + 1) // stride
    which involves FloorDiv when the input length is dynamic.

    Currently raises NotImplementedError for FloorDiv sympy expressions.
    """

    class Conv1dModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # stride=2 causes floor division in output shape calculation
            self.conv = torch.nn.Conv1d(
                in_channels=1, out_channels=1, kernel_size=3, stride=2
            )

        def forward(self, x):
            return self.conv(x)

    model = Conv1dModel()
    # Input shape: (batch, channels, sequence_length)
    example_args = (torch.randn(1, 1, 100),)

    # Make the sequence length dynamic
    seq_len = torch.export.Dim("seq_len", min=3)
    dynamic_shapes = {"x": {2: seq_len}}

    export(
        model,
        args=example_args,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "-vv"] + args)
    sys.exit(ret_code)
