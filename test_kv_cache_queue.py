#!/usr/bin/env python3
"""
Test script for KVCacheBlockQueue debugging.
Run this in Colab to test the low priority block insertion logic.
"""

import sys
import os

# Add the vllm directory to the path if needed
# sys.path.append('/path/to/vllm')

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

def test_queue_operations():
    """Test various queue operations with detailed output."""
    
    print("=" * 60)
    print("TESTING KV CACHE QUEUE OPERATIONS")
    print("=" * 60)
    
    # Test 1: Start with a queue with some initial unhashed blocks
    print("\n" + "="*40)
    print("TEST 1: Initialize queue with unhashed blocks")
    print("="*40)
    
    # Create some initial unhashed blocks like vllm does in practice
    initial_blocks = [
        FreeKVCacheBlockQueue.create_test_block(i, is_cached=False) 
        for i in range(3)
    ]
    queue = FreeKVCacheBlockQueue(initial_blocks)
    print("Initialized queue with unhashed blocks:")
    queue._print_queue_state()
    
    # Test 2: Insert low priority block into empty queue
    print("\n" + "="*40)
    print("TEST 2: Insert low priority block into empty queue")
    print("="*40)
    
    low_block_1 = FreeKVCacheBlockQueue.create_test_block(100, is_cached=True, low_priority=True)
    queue.append(low_block_1)
    
    # Test 3: Insert another low priority block
    print("\n" + "="*40)
    print("TEST 3: Insert another low priority block")
    print("="*40)
    
    low_block_2 = FreeKVCacheBlockQueue.create_test_block(101, is_cached=True, low_priority=True)
    queue.append(low_block_2)
    
    # Test 4: Insert high priority block
    print("\n" + "="*40)
    print("TEST 4: Insert high priority block")
    print("="*40)
    
    high_block_1 = FreeKVCacheBlockQueue.create_test_block(200, is_cached=True, low_priority=False)
    queue.append(high_block_1)
    
    # Test 5: Remove head block (should be low priority)
    print("\n" + "="*40)
    print("TEST 5: Remove head block")
    print("="*40)
    
    removed_block = queue.popleft()
    print(f"Removed block: {removed_block.block_id}")
    
    # Test 6: Insert another low priority block (should go to head)
    print("\n" + "="*40)
    print("TEST 6: Insert low priority block after removal")
    print("="*40)
    
    low_block_3 = FreeKVCacheBlockQueue.create_test_block(102, is_cached=True, low_priority=True)
    queue.append(low_block_3)
    
    # Test 7: Remove the last low priority block
    print("\n" + "="*40)
    print("TEST 7: Remove remaining low priority blocks")
    print("="*40)
    
    queue.remove(low_block_2)  # Remove the current last_low_priority_cached_block
    
    # Test 8: Insert low priority block when pointer is None
    print("\n" + "="*40)
    print("TEST 8: Insert low priority block when no low priority blocks exist")
    print("="*40)
    
    low_block_4 = FreeKVCacheBlockQueue.create_test_block(103, is_cached=True, low_priority=True)
    queue.append(low_block_4)

def test_empty_queue_scenario():
    """Test what happens when queue becomes empty during operation."""
    
    print("\n" + "="*60)
    print("TESTING EMPTY QUEUE SCENARIO")
    print("="*60)
    
    # Start with a small queue
    initial_blocks = [FreeKVCacheBlockQueue.create_test_block(i, is_cached=False) for i in range(2)]
    queue = FreeKVCacheBlockQueue(initial_blocks)
    
    print("Initial state:")
    queue._print_queue_state()
    
    # Remove all blocks to make it empty
    print("\n" + "-"*40)
    print("Removing all blocks to make queue empty:")
    print("-"*40)
    
    while queue.num_free_blocks > 0:
        removed = queue.popleft()
        print(f"Removed block {removed.block_id}")
    
    # Now add a low priority block to empty queue
    print("\n" + "-"*40)
    print("Adding low priority block to empty queue:")
    print("-"*40)
    
    low_block = FreeKVCacheBlockQueue.create_test_block(100, is_cached=True, low_priority=True)
    queue.append(low_block)

def test_with_initial_blocks():
    """Test with initial unhashed blocks."""
    
    print("\n" + "="*60)
    print("TESTING WITH INITIAL UNHASHED BLOCKS")
    print("="*60)
    
    # Create some initial unhashed blocks
    initial_blocks = [
        FreeKVCacheBlockQueue.create_test_block(i, is_cached=False) 
        for i in range(5)
    ]
    
    queue = FreeKVCacheBlockQueue(initial_blocks)
    print("Initialized queue with unhashed blocks:")
    queue._print_queue_state()
    
    # Add some low priority blocks
    print("\n" + "-"*40)
    print("Adding low priority blocks:")
    print("-"*40)
    
    for i in range(10, 12):
        low_block = FreeKVCacheBlockQueue.create_test_block(i, is_cached=True, low_priority=True)
        queue.append(low_block)
    
    # Add some high priority blocks
    print("\n" + "-"*40)
    print("Adding high priority blocks:")
    print("-"*40)
    
    for i in range(20, 22):
        high_block = FreeKVCacheBlockQueue.create_test_block(i, is_cached=True, low_priority=False)
        queue.append(high_block)
    
    # Remove some blocks from head
    print("\n" + "-"*40)
    print("Removing blocks from head:")
    print("-"*40)
    
    for _ in range(3):
        removed = queue.popleft()
        print(f"Removed block {removed.block_id}")

if __name__ == "__main__":
    test_queue_operations()
    test_empty_queue_scenario()
    test_with_initial_blocks()
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60) 