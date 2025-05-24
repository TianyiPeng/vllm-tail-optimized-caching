import pytest
from vllm.config import CacheConfig
from vllm.core.block_manager import SelfAttnBlockSpaceManager

def test_caching_low_priority_last_num_tokens_propagation():
    block_size = 4
    num_gpu_blocks = 16
    num_cpu_blocks = 8
    caching_low_priority_last_num_tokens = 8

    # Create a CacheConfig with the new parameter
    cache_config = CacheConfig(
        block_size=block_size,
        sliding_window=None,
        enable_prefix_caching=True,
        caching_low_priority_last_num_tokens=caching_low_priority_last_num_tokens,
    )
    # Simulate the block manager creation as in the engine/scheduler
    block_manager = SelfAttnBlockSpaceManager(
        block_size=cache_config.block_size,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        sliding_window=cache_config.sliding_window,
        enable_caching=cache_config.enable_prefix_caching,
        caching_low_priority_last_num_tokens=cache_config.caching_low_priority_last_num_tokens,
    )

    # Assert the value is set correctly
    assert block_manager.caching_low_priority_last_num_tokens == caching_low_priority_last_num_tokens

if __name__ == "__main__":
    pytest.main([__file__]) 