#include "spatiotemporal_memory.hpp"
#include <iostream>
#include <thread>
#include <chrono>

namespace elysia::emulation {

VirtualTierPipeline::VirtualTierPipeline(double cache_latency_ns, double ram_latency_ns, double ssd_latency_ns)
    : cache_latency_(cache_latency_ns), ram_latency_(ram_latency_ns), ssd_latency_(ssd_latency_ns), dma_queue_count_(0) {}

VirtualTierPipeline::~VirtualTierPipeline() = default;

void VirtualTierPipeline::trigger_async_dma_imprint(uint32_t chart_id, const MetricTensor3x3& metric_delta) {
    (void)chart_id;
    (void)metric_delta;
    dma_queue_count_++;
    // Asynchronous DMA dispatch logic simulation
    std::thread([this]() {
        std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long long>(ssd_latency_ / 1000.0)));
        if (dma_queue_count_ > 0) {
            dma_queue_count_--;
        }
    }).detach();
}

double VirtualTierPipeline::simulate_tier_transfer(size_t data_bytes, int source_tier, int target_tier) {
    double base_lat = cache_latency_;
    if (source_tier == 1 || target_tier == 1) base_lat = ram_latency_;
    if (source_tier == 2 || target_tier == 2) base_lat = ssd_latency_;

    double bandwidth_bytes_per_ns = 100.0; // 100 GB/s nominal simulation
    double transfer_time = static_cast<double>(data_bytes) / bandwidth_bytes_per_ns;
    return base_lat + transfer_time;
}

} // namespace elysia::emulation
