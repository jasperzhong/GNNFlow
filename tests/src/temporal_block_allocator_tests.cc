#include <gtest/gtest.h>

#include "temporal_block_allocator.h"

class TemporalBlockAllocatorTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    std::size_t GB = 1024 * 1024 * 1024;
    std::size_t alignment = 16;
    allocator_.reset(new dgnn::TemporalBlockAllocator(GB, alignment));
  }

  std::unique_ptr<dgnn::TemporalBlockAllocator> allocator_;
};

TEST_F(TemporalBlockAllocatorTest, SingleAllocateAndDeallocate) {
  std::size_t size = 1024;
  auto block = allocator_->Allocate(size);
  EXPECT_EQ(block->size, 0);
  EXPECT_EQ(block->capacity, 1024);

  EXPECT_EQ(allocator_->num_blocks_on_device(), 1);
  EXPECT_EQ(allocator_->num_blocks_on_host(), 0);
  EXPECT_EQ(allocator_->used_space_on_device(),
            allocator_->AlignUp(size) * dgnn::kBlockSpaceSize);
  EXPECT_EQ(allocator_->used_space_on_host(), 0);

  allocator_->Deallocate(block);

  EXPECT_EQ(allocator_->num_blocks_on_device(), 0);
  EXPECT_EQ(allocator_->num_blocks_on_host(), 0);
  EXPECT_EQ(allocator_->used_space_on_device(), 0);
  EXPECT_EQ(allocator_->used_space_on_host(), 0);
}

TEST_F(TemporalBlockAllocatorTest, MultipleAllocateAndDeallocate) {
  std::vector<std::size_t> sizes = {1024, 2048, 4096, 8192};
  std::vector<std::shared_ptr<dgnn::TemporalBlock>> blocks;
  std::size_t total_size = 0;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    auto block = allocator_->Allocate(sizes[i]);
    blocks.push_back(block);

    total_size += allocator_->AlignUp(sizes[i]) * dgnn::kBlockSpaceSize;

    EXPECT_EQ(allocator_->num_blocks_on_device(), i + 1);
    EXPECT_EQ(allocator_->num_blocks_on_host(), 0);
    EXPECT_EQ(allocator_->used_space_on_device(), total_size);
    EXPECT_EQ(allocator_->used_space_on_host(), 0);
  }

  for (auto block : blocks) {
    allocator_->Deallocate(block);
  }

  EXPECT_EQ(allocator_->num_blocks_on_device(), 0);
  EXPECT_EQ(allocator_->num_blocks_on_host(), 0);
  EXPECT_EQ(allocator_->used_space_on_device(), 0);
  EXPECT_EQ(allocator_->used_space_on_host(), 0);
}

TEST_F(TemporalBlockAllocatorTest, SwapBlockToHost) {
  std::size_t size = 1024;
  auto block = allocator_->Allocate(size);
  block->size = size;

  block = allocator_->SwapBlockToHost(block);
  EXPECT_EQ(block->size, 1024);
  EXPECT_EQ(block->capacity, 1024);
  EXPECT_NE(block->dst_nodes, nullptr);
  EXPECT_NE(block->timestamps, nullptr);
  EXPECT_NE(block->eids, nullptr);

  EXPECT_EQ(allocator_->num_blocks_on_device(), 0);
  EXPECT_EQ(allocator_->num_blocks_on_host(), 1);
  EXPECT_EQ(allocator_->used_space_on_device(), 0);
  EXPECT_EQ(allocator_->used_space_on_host(), size * dgnn::kBlockSpaceSize);
}
