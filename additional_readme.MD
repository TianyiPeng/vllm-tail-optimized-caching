# Tail-Optimized LRU for VLLM

## What's New in This Fork

- **Tail-Optimized (Low-Priority) Caching:**
  - Implements a caching strategy that prioritizes eviction of low-priority (tail) tokens, improving cache efficiency for long-context and multi-user scenarios.
  - New configuration option: `caching_low_priority_last_num_tokens` (set in `CacheConfig` or via LLM init) to control how many tokens in the end of a sequence are considered low-priority for eviction.
- **Tests:**
  - Added tests for low-priority caching logic in `tests/core/block/test_caching_low_priority.py`.

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
