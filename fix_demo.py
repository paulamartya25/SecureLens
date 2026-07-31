# Fix the demo file
with open('demo_true_fhe.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic lines
content = content.replace(
    'print(f\"  - Secret key size: {len(ckks.context.secret_key().save())} bytes\")\nprint(f\"  - Public context size: {len(ckks.public_context.save())} bytes\")',
    'print(\"  - Secret key generated (kept on client only)\")\nprint(\"  - Public context created (can be shared with server)\")'
)

with open('demo_true_fhe.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Fixed demo_true_fhe.py')
