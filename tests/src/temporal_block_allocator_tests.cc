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

TEST_F(TemporalBlockAllocatorTest, AllocateAndDeallocate) {
  std::size_t size = 1024;
  auto block = allocator_->AllocateTemporalBlock(size);
  EXPECT_EQ(block->size, 0);
  EXPECT_EQ(block->capacity, size);
  allocator_->Print();

  allocator_->DeallocateTemporalBlock(block);
  allocator_->Print();
}

TEST_F(TemporalBlockAllocatorTest, Reallocate) {
  std::size_t size = 1024;
  auto block = allocator_->AllocateTemporalBlock(size);
  allocator_->Print();

  std::size_t new_size = 2048;
  auto new_block = allocator_->ReallocateTemporalBlock(block, new_size);
  EXPECT_EQ(new_block->size, 0);
  EXPECT_EQ(new_block->capacity, new_size);
  allocator_->Print();

  allocator_->DeallocateTemporalBlock(new_block);
  allocator_->Print();
}
