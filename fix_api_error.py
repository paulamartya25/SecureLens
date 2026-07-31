#!/usr/bin/env python3
with open("test_true_fhe.py", 'r') as f:
    content = f.read()

content = content.replace(
    "enc_server = server_pub_ctx.ckks_vector_from(ct_bytes)",
    "enc_server = ts.ckks_vector_from(server_pub_ctx, ct_bytes)"
)

with open("test_true_fhe.py", 'w') as f:
    f.write(content)

print("✅ Fixed!")
