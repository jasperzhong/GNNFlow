#include <gtest/gtest.h>

#include <random>

#include "dynamic_graph.h"

class DynamicGraphTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    std::size_t GB = 1024 * 1024 * 1024;
    graph_.reset(new dgnn::DynamicGraph(GB));
  }

  std::unique_ptr<dgnn::DynamicGraph> graph_;
};

TEST_F(DynamicGraphTest, AddEdges) {
  std::vector<dgnn::NIDType> src_nodes = {0, 0, 0, 1, 1, 1, 2, 2, 2};
  std::vector<dgnn::NIDType> dst_nodes = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<dgnn::TimestampType> timestamps = {0, 1, 2, 0, 1, 2, 0, 1, 2};

  graph_->AddEdges(src_nodes, dst_nodes, timestamps, false);

  auto node_table = graph_->h_copy_of_d_node_table();

  std::vector<dgnn::NIDType> dst_nodes_on_device(3);
  std::vector<dgnn::TimestampType> timestamps_on_device(3);
  std::vector<dgnn::EIDType> eids_on_device(3);

  for (std::size_t i = 0; i < 3; ++i) {
    auto block = node_table[i].head.next;
    EXPECT_EQ(block->size, 3);
    EXPECT_EQ(block->capacity, 16);
    thrust::copy(
        thrust::device_ptr<dgnn::NIDType>(block->dst_nodes),
        thrust::device_ptr<dgnn::NIDType>(block->dst_nodes) + block->size,
        dst_nodes_on_device.begin());

    thrust::copy(thrust::device_ptr<dgnn::TimestampType>(block->timestamps),
                 thrust::device_ptr<dgnn::TimestampType>(block->timestamps) +
                     block->size,
                 timestamps_on_device.begin());

    thrust::copy(thrust::device_ptr<dgnn::EIDType>(block->eids),
                 thrust::device_ptr<dgnn::EIDType>(block->eids) + block->size,
                 eids_on_device.begin());

    std::vector<dgnn::NIDType> expected_dst_nodes_on_device = {1, 2, 3};
    EXPECT_EQ(dst_nodes_on_device, expected_dst_nodes_on_device);

    std::vector<dgnn::TimestampType> expected_timestamps_on_device = {0, 1, 2};
    EXPECT_EQ(timestamps_on_device, expected_timestamps_on_device);

    std::vector<dgnn::EIDType> expected_eids_on_device = {0 + i * 3, 1 + i * 3,
                                                          2 + i * 3};
    EXPECT_EQ(eids_on_device, expected_eids_on_device);
  }
}

TEST_F(DynamicGraphTest, AddEdgesWithReverseEdges) {
  std::vector<dgnn::NIDType> src_nodes = {0, 0, 0, 1, 1, 1, 2, 2, 2};
  std::vector<dgnn::NIDType> dst_nodes = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<dgnn::TimestampType> timestamps = {0, 1, 2, 0, 1, 2, 0, 1, 2};

  graph_->AddEdges(src_nodes, dst_nodes, timestamps, true);

  auto node_table = graph_->h_copy_of_d_node_table();

  {
    auto block = node_table[0].head.next;
    EXPECT_EQ(block->size, 3);
    EXPECT_EQ(block->capacity, 16);
    std::vector<dgnn::NIDType> dst_nodes_on_device(3);
    std::vector<dgnn::TimestampType> timestamps_on_device(3);
    std::vector<dgnn::EIDType> eids_on_device(3);

    thrust::copy(
        thrust::device_ptr<dgnn::NIDType>(block->dst_nodes),
        thrust::device_ptr<dgnn::NIDType>(block->dst_nodes) + block->size,
        dst_nodes_on_device.begin());
    thrust::copy(thrust::device_ptr<dgnn::TimestampType>(block->timestamps),
                 thrust::device_ptr<dgnn::TimestampType>(block->timestamps) +
                     block->size,
                 timestamps_on_device.begin());

    thrust::copy(thrust::device_ptr<dgnn::EIDType>(block->eids),
                 thrust::device_ptr<dgnn::EIDType>(block->eids) + block->size,
                 eids_on_device.begin());

    std::vector<dgnn::NIDType> expected_dst_nodes_on_device = {1, 2, 3};
    EXPECT_EQ(dst_nodes_on_device, expected_dst_nodes_on_device);

    std::vector<dgnn::TimestampType> expected_timestamps_on_device = {0, 1, 2};
    EXPECT_EQ(timestamps_on_device, expected_timestamps_on_device);

    std::vector<dgnn::EIDType> expected_eids_on_device = {0, 1, 2};
    EXPECT_EQ(eids_on_device, expected_eids_on_device);
  }

  {
    auto block = node_table[1].head.next;
    EXPECT_EQ(block->size, 6);
    EXPECT_EQ(block->capacity, 16);
    std::vector<dgnn::NIDType> dst_nodes_on_device(6);
    std::vector<dgnn::TimestampType> timestamps_on_device(6);
    std::vector<dgnn::EIDType> eids_on_device(6);

    thrust::copy(
        thrust::device_ptr<dgnn::NIDType>(block->dst_nodes),
        thrust::device_ptr<dgnn::NIDType>(block->dst_nodes) + block->size,
        dst_nodes_on_device.begin());

    thrust::copy(thrust::device_ptr<dgnn::TimestampType>(block->timestamps),
                 thrust::device_ptr<dgnn::TimestampType>(block->timestamps) +
                     block->size,
                 timestamps_on_device.begin());

    thrust::copy(thrust::device_ptr<dgnn::EIDType>(block->eids),
                 thrust::device_ptr<dgnn::EIDType>(block->eids) + block->size,
                 eids_on_device.begin());

    std::vector<dgnn::NIDType> expected_dst_nodes_on_device = {1, 0, 1,
                                                               2, 2, 3};

    EXPECT_EQ(dst_nodes_on_device, expected_dst_nodes_on_device);

    std::vector<dgnn::TimestampType> expected_timestamps_on_device = {0, 0, 0,
                                                                      0, 1, 2};
    EXPECT_EQ(timestamps_on_device, expected_timestamps_on_device);

    std::vector<dgnn::EIDType> expected_eids_on_device = {3, 0, 3, 6, 4, 5};
    EXPECT_EQ(eids_on_device, expected_eids_on_device);
  }

  {
    auto block = node_table[2].head.next;
    EXPECT_EQ(block->size, 6);
    EXPECT_EQ(block->capacity, 16);
    std::vector<dgnn::NIDType> dst_nodes_on_device(6);
    std::vector<dgnn::TimestampType> timestamps_on_device(6);
    std::vector<dgnn::EIDType> eids_on_device(6);

    thrust::copy(
        thrust::device_ptr<dgnn::NIDType>(block->dst_nodes),
        thrust::device_ptr<dgnn::NIDType>(block->dst_nodes) + block->size,
        dst_nodes_on_device.begin());

    thrust::copy(thrust::device_ptr<dgnn::TimestampType>(block->timestamps),
                 thrust::device_ptr<dgnn::TimestampType>(block->timestamps) +
                     block->size,
                 timestamps_on_device.begin());

    thrust::copy(thrust::device_ptr<dgnn::EIDType>(block->eids),
                 thrust::device_ptr<dgnn::EIDType>(block->eids) + block->size,
                 eids_on_device.begin());

    std::vector<dgnn::NIDType> expected_dst_nodes_on_device = {1, 2, 0,
                                                               1, 2, 3};

    EXPECT_EQ(dst_nodes_on_device, expected_dst_nodes_on_device);

    std::vector<dgnn::TimestampType> expected_timestamps_on_device = {0, 1, 1,
                                                                      1, 1, 2};
    EXPECT_EQ(timestamps_on_device, expected_timestamps_on_device);

    std::vector<dgnn::EIDType> expected_eids_on_device = {6, 7, 1, 4, 7, 8};
    EXPECT_EQ(eids_on_device, expected_eids_on_device);
  }

  {
    auto block = node_table[3].head.next;
    EXPECT_EQ(block->size, 3);
    EXPECT_EQ(block->capacity, 16);
    std::vector<dgnn::NIDType> dst_nodes_on_device(3);
    std::vector<dgnn::TimestampType> timestamps_on_device(3);
    std::vector<dgnn::EIDType> eids_on_device(3);

    thrust::copy(
        thrust::device_ptr<dgnn::NIDType>(block->dst_nodes),
        thrust::device_ptr<dgnn::NIDType>(block->dst_nodes) + block->size,
        dst_nodes_on_device.begin());

    thrust::copy(thrust::device_ptr<dgnn::TimestampType>(block->timestamps),
                 thrust::device_ptr<dgnn::TimestampType>(block->timestamps) +
                     block->size,
                 timestamps_on_device.begin());

    thrust::copy(thrust::device_ptr<dgnn::EIDType>(block->eids),
                 thrust::device_ptr<dgnn::EIDType>(block->eids) + block->size,
                 eids_on_device.begin());

    std::vector<dgnn::NIDType> expected_dst_nodes_on_device = {0, 1, 2};

    EXPECT_EQ(dst_nodes_on_device, expected_dst_nodes_on_device);

    std::vector<dgnn::TimestampType> expected_timestamps_on_device = {2, 2, 2};
    EXPECT_EQ(timestamps_on_device, expected_timestamps_on_device);

    std::vector<dgnn::EIDType> expected_eids_on_device = {2, 5, 8};
    EXPECT_EQ(eids_on_device, expected_eids_on_device);
  }
}
