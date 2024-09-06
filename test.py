import torch
import numpy as np

# softmax(Q * K^t) * V
if __name__ == "__main__":
    pass

def gen_attention_no_mask():
    m = 4096
    k1 = 64
    k2 = 64
    n = 64
    b0 = 1
    b1 = 1
    q = (torch.rand(b0,b1,m,n).to(torch.float32) - 0.5) * 1
    k = (torch.rand(b0,b1,k1,n).to(torch.float32) - 0.5) * 1
    v = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * 1
    mask = (torch.rand(b0,b1,m,k1).to(torch.float32) - 0.5) * 1

    np.save("attn_q.npy", q.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_k.npy", k.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_v.npy", v.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_mask.npy", mask.detach().to(dtype=torch.float16, device="cpu").numpy())

    # Post attention func
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    np.save("attn_ref.npy", out.detach().to(device="cpu").numpy())

def gen_attention_float_mask():
    m = 4096
    k2 = 64
    n = 64
    b0 = 1
    b1 = 1
    q = (torch.rand(b0,b1,m,n).to(torch.float32) - 0.5) * 1
    k = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * 1
    v = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * 1
    mask = (torch.rand(b0,b1,m,k2).to(torch.float32) - 0.5) * 1

    np.save("attn_q.npy", q.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_k.npy", k.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_v.npy", v.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_mask.npy", mask.detach().to(dtype=torch.float16, device="cpu").numpy())

    # Post attention func
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    np.save("attn_ref.npy", out.detach().to(device="cpu").numpy())

def gen_attention_bool_mask():
    m = 4096
    k2 = 64
    n = 64
    b0 = 1
    b1 = 1
    q = (torch.rand(b0,b1,m,n).to(torch.float32) - 0.5) * 1
    k = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * 1
    v = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * 1
    mask = (torch.rand(b0,b1,m,k2).to(torch.float32) - 0.5) > 0

    np.save("attn_q.npy", q.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_k.npy", k.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_v.npy", v.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_mask.npy", mask.detach().to(dtype=torch.bool, device="cpu").numpy())

    # Post attention func
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    np.save("attn_ref.npy", out.detach().to(device="cpu").numpy())

def gen_attention_causal_mask():
    m = 4096
    k2 = 64
    n = 64
    b0 = 1
    b1 = 1
    q = (torch.rand(b0,b1,m,n).to(torch.float32) - 0.5) * 1
    k = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * 1
    v = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * 1
    mask = (torch.rand(b0,b1,m,k2).to(torch.float32) - 0.5) > 0

    np.save("attn_q.npy", q.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_k.npy", k.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_v.npy", v.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_mask.npy", mask.detach().to(dtype=torch.bool, device="cpu").numpy())

    # Post attention func
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    np.save("attn_ref.npy", out.detach().to(device="cpu").numpy())

gen_attention_no_mask()