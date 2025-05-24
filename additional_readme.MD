# Tail-Optimized LRU for VLLM

## Installation from Source

To install this vLLM variant from source:

1. Clone the repository:
   ```bash
   git clone  https://github.com/TianyiPeng/vllm-tail-optimized-caching.git
   cd vllm-tail-optimized-caching
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv vllm_env
   source vllm_env/bin/activate
   ```

3. Install vLLM in editable mode:
   ```bash
   VLLM_USE_PRECOMPILED=1 pip install -v --editable .
   ```
