import torch
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", torch.float32)

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    print("Successfully imported apply_rotary_pos_emb, repeat_kv from qwen3_5")
except Exception as e:
    print("Failed from qwen3_5:", e)
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
        print("Successfully imported apply_rotary_pos_emb, repeat_kv from qwen2")
    except Exception as e2:
        print("Failed from qwen2:", e2)
