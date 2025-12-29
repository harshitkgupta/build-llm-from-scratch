# Build LLM from Scratch

## Notebooks
1. [Word Level Tokenizer](notebooks/Tokenizer.ipynb)
2. [Byte Pair Encoding(BPE)](notebooks/Byte_Pair_Encoding.ipynb)

#### Check CPU
```
import os
print(f"CPU count: {os.cpu_count()}")
```

#### Check GPU usage
```
!nvidia-smi
```

#### Check TPU status
```
import jax
print(jax.devices())

devices = jax.devices()
print(f"Detected {len(devices)} TPU device(s)")
print(f"Device type: {devices[0].platform}")
print(f"TPU Kind: {devices[0].device_kind}")

```

### Check memory
```
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
```