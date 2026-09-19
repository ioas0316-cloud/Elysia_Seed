#include "spatiotemporal_memory.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace elysia::emulation {

AtlasManager::AtlasManager(size_t initial_chart_count) {
    charts_.reserve(initial_chart_count);
    active_mask_buffer_.resize(initial_chart_count, 0);

    // Grid layout for initial charts in 3D
    size_t side = static_cast<size_t>(std::ceil(std::cbrt(initial_chart_count)));
    if (side == 0) side = 1;
    float step = 1.0f / side;
    float r = step * 1.2f; // Overlapping radius

    size_t count = 0;
    for (size_t x = 0; x < side && count < initial_chart_count; ++x) {
        for (size_t y = 0; y < side && count < initial_chart_count; ++y) {
            for (size_t z = 0; z < side && count < initial_chart_count; ++z) {
                LocalChart chart;
                chart.chart_id = static_cast<uint32_t>(count);
                chart.center[0] = (x + 0.5f) * step;
                chart.center[1] = (y + 0.5f) * step;
                chart.center[2] = (z + 0.5f) * step;
                chart.radius = r;
                chart.is_active = false;
                charts_.push_back(chart);
                count++;
            }
        }
    }
}

AtlasManager::~AtlasManager() = default;

std::vector<LocalChart> AtlasManager::query_active_charts(const float current_pos[3]) {
    std::vector<LocalChart> active;
    std::fill(active_mask_buffer_.begin(), active_mask_buffer_.end(), 0);

    for (size_t i = 0; i < charts_.size(); ++i) {
        float dx = current_pos[0] - charts_[i].center[0];
        float dy = current_pos[1] - charts_[i].center[1];
        float dz = current_pos[2] - charts_[i].center[2];
        float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq <= charts_[i].radius * charts_[i].radius) {
            charts_[i].is_active = true;
            active_mask_buffer_[i] = 1;
            active.push_back(charts_[i]);
        } else {
            charts_[i].is_active = false;
        }
    }

    return active;
}

Rotor AtlasManager::compute_chart_transition_rotor(uint32_t src_chart_id, uint32_t dst_chart_id) {
    if (src_chart_id >= charts_.size() || dst_chart_id >= charts_.size()) {
        return { 1.0f, 0.0f, 0.0f, 0.0f };
    }

    const auto& src = charts_[src_chart_id];
    const auto& dst = charts_[dst_chart_id];

    float dx = dst.center[0] - src.center[0];
    float dy = dst.center[1] - src.center[1];
    float dz = dst.center[2] - src.center[2];

    float theta = std::sqrt(dx * dx + dy * dy + dz * dz) * 0.5f;
    float s = std::sin(theta);
    float c = std::cos(theta);

    return { c, dx * s, dy * s, dz * s };
}

void AtlasManager::update_chart_active_states(const std::vector<uint32_t>& active_ids) {
    for (auto& chart : charts_) {
        chart.is_active = false;
    }
    std::fill(active_mask_buffer_.begin(), active_mask_buffer_.end(), 0);

    for (uint32_t id : active_ids) {
        if (id < charts_.size()) {
            charts_[id].is_active = true;
            active_mask_buffer_[id] = 1;
        }
    }
}

} // namespace elysia::emulation
