import torch

try:
    import flash_attn
    from flash_attn import flash_attn_func

    print(f"✅ FlashAttention version: {flash_attn.__version__}")

    if torch.cuda.is_available():
        print(f"✅ CUDA is available. Device: {torch.cuda.get_device_name(0)}")

        # Dữ liệu cần ép kiểu sang float16 (fp16)
        q = torch.randn(2, 8, 32, 64, device='cuda', dtype=torch.float16)
        k = torch.randn(2, 8, 32, 64, device='cuda', dtype=torch.float16)
        v = torch.randn(2, 8, 32, 64, device='cuda', dtype=torch.float16)

        # Gọi FlashAttention
        out = flash_attn_func(q, k, v)

        print("✅ FlashAttention test successful! Output shape:", out.shape)
    else:
        print("❌ CUDA device not available!")
except ImportError as e:
    print("❌ Import Error:", e)
except RuntimeError as e:
    print("❌ Runtime Error:", e)
