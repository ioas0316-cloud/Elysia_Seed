#include "spatiotemporal_flux_scaling.hpp"
#include <cmath>
#include <algorithm>
#include <array>

namespace elysia::emulation {

#ifndef __CUDACC__

static inline float compute_determinant_3x3_cpu(const MetricTensor3x3& m) {
    return m.g[0][0] * (m.g[1][1] * m.g[2][2] - m.g[1][2] * m.g[2][1])
         - m.g[0][1] * (m.g[1][0] * m.g[2][2] - m.g[1][2] * m.g[2][0])
         + m.g[0][2] * (m.g[1][0] * m.g[2][1] - m.g[1][1] * m.g[2][0]);
}

void MetricFieldEngineFluxExtension::launch_compute_flux_divergence_kernel(
    const MetricTensor3x3* d_g_field,
    const float* d_density,
    const float* d_velocity,
    float* d_bottleneck_out,
    size_t dim_x, size_t dim_y, size_t dim_z,
    float grid_spacing,
    cudaStream_t stream)
{
    (void)stream;
    for (size_t gz = 0; gz < dim_z; ++gz) {
        for (size_t gy = 0; gy < dim_y; ++gy) {
            for (size_t gx = 0; gx < dim_x; ++gx) {
                size_t center_idx = gx + gy * dim_x + gz * dim_x * dim_y;

                auto fetch_weighted_flux = [&](int x, int y, int z) {
                    x = std::min(std::max(x, 0), static_cast<int>(dim_x) - 1);
                    y = std::min(std::max(y, 0), static_cast<int>(dim_y) - 1);
                    z = std::min(std::max(z, 0), static_cast<int>(dim_z) - 1);
                    size_t idx = x + y * dim_x + z * dim_x * dim_y;

                    float det_g = std::max(compute_determinant_3x3_cpu(d_g_field[idx]), 1e-6f);
                    float sqrt_det = std::sqrt(det_g);
                    float rho = d_density[idx];

                    std::array<float, 3> res;
                    res[0] = sqrt_det * rho * d_velocity[idx * 3 + 0];
                    res[1] = sqrt_det * rho * d_velocity[idx * 3 + 1];
                    res[2] = sqrt_det * rho * d_velocity[idx * 3 + 2];
                    return res;
                };

                float center_det = std::max(compute_determinant_3x3_cpu(d_g_field[center_idx]), 1e-6f);
                float inv_sqrt_det_center = 1.0f / std::sqrt(center_det);

                auto flux_px = fetch_weighted_flux(static_cast<int>(gx) + 1, static_cast<int>(gy), static_cast<int>(gz));
                auto flux_nx = fetch_weighted_flux(static_cast<int>(gx) - 1, static_cast<int>(gy), static_cast<int>(gz));
                auto flux_py = fetch_weighted_flux(static_cast<int>(gx), static_cast<int>(gy) + 1, static_cast<int>(gz));
                auto flux_ny = fetch_weighted_flux(static_cast<int>(gx), static_cast<int>(gy) - 1, static_cast<int>(gz));
                auto flux_pz = fetch_weighted_flux(static_cast<int>(gx), static_cast<int>(gy), static_cast<int>(gz) + 1);
                auto flux_nz = fetch_weighted_flux(static_cast<int>(gx), static_cast<int>(gy), static_cast<int>(gz) - 1);

                float dJ_dx = (flux_px[0] - flux_nx[0]) / (2.0f * grid_spacing);
                float dJ_dy = (flux_py[1] - flux_ny[1]) / (2.0f * grid_spacing);
                float dJ_dz = (flux_pz[2] - flux_nz[2]) / (2.0f * grid_spacing);

                float riemannian_div = inv_sqrt_det_center * (dJ_dx + dJ_dy + dJ_dz);
                d_bottleneck_out[center_idx] = riemannian_div;
            }
        }
    }
}

void MetricFieldEngineFluxExtension::launch_apply_bottleneck_metric_stress_kernel(
    MetricTensor3x3* d_g_field,
    const float* d_bottleneck_index,
    float eta_stress_coefficient,
    size_t total_cells,
    cudaStream_t stream)
{
    (void)stream;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        float p_stress = d_bottleneck_index[idx];
        if (p_stress > 0.0f) {
            float stress_delta = eta_stress_coefficient * p_stress;
            d_g_field[idx].g[0][0] += stress_delta;
            d_g_field[idx].g[1][1] += stress_delta;
            d_g_field[idx].g[2][2] += stress_delta;
        }
    }
}

#endif // __CUDACC__

AtlasManagerScalingExtension::AtlasManagerScalingExtension(size_t max_depth)
    : max_depth_(max_depth), next_chart_id_(1000) {}

uint32_t AtlasManagerScalingExtension::generate_next_chart_id() {
    return ++next_chart_id_;
}

float AtlasManagerScalingExtension::compute_max_bottleneck_in_chart(
    const LocalChart& chart,
    const float* bottleneck_map,
    size_t dim_x, size_t dim_y, size_t dim_z)
{
    float max_p = 0.0f;
    int min_x = std::max(0, static_cast<int>((chart.center[0] - chart.radius) * dim_x));
    int max_x = std::min(static_cast<int>(dim_x) - 1, static_cast<int>((chart.center[0] + chart.radius) * dim_x));
    int min_y = std::max(0, static_cast<int>((chart.center[1] - chart.radius) * dim_y));
    int max_y = std::min(static_cast<int>(dim_y) - 1, static_cast<int>((chart.center[1] + chart.radius) * dim_y));
    int min_z = std::max(0, static_cast<int>((chart.center[2] - chart.radius) * dim_z));
    int max_z = std::min(static_cast<int>(dim_z) - 1, static_cast<int>((chart.center[2] + chart.radius) * dim_z));

    for (int z = min_z; z <= max_z; ++z) {
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                size_t idx = x + y * dim_x + z * dim_x * dim_y;
                max_p = std::max(max_p, bottleneck_map[idx]);
            }
        }
    }
    return max_p;
}

void AtlasManagerScalingExtension::evaluate_merge_candidates(
    const std::unordered_map<uint32_t, ScaledLocalChartNode>& chart_registry,
    std::vector<ChartScaleTrigger>& triggers,
    float merge_threshold)
{
    std::unordered_map<uint32_t, std::vector<uint32_t>> parent_to_children_map;

    for (const auto& trigger : triggers) {
        auto it = chart_registry.find(trigger.chart_id);
        if (it != chart_registry.end() && it->second.parent_id != 0xFFFFFFFF) {
            if (trigger.max_bottleneck_pressure < merge_threshold) {
                parent_to_children_map[it->second.parent_id].push_back(trigger.chart_id);
            }
        }
    }

    for (auto& trigger : triggers) {
        auto it = chart_registry.find(trigger.chart_id);
        if (it != chart_registry.end()) {
            uint32_t p_id = it->second.parent_id;
            if (p_id != 0xFFFFFFFF && parent_to_children_map[p_id].size() == 8) {
                trigger.requires_merge = true;
            }
        }
    }
}

std::vector<AtlasManagerScalingExtension::ChartScaleTrigger> AtlasManagerScalingExtension::evaluate_chart_bottlenecks(
    const std::vector<ScaledLocalChartNode>& chart_nodes,
    const float* host_bottleneck_map,
    size_t grid_dim_x, size_t grid_dim_y, size_t grid_dim_z,
    float split_threshold,
    float merge_threshold)
{
    std::vector<ChartScaleTrigger> triggers;
    std::unordered_map<uint32_t, ScaledLocalChartNode> registry;

    for (const auto& node : chart_nodes) {
        registry[node.chart_data.chart_id] = node;
        if (!node.chart_data.is_active || !node.is_leaf) continue;

        float max_p = compute_max_bottleneck_in_chart(
            node.chart_data, host_bottleneck_map, grid_dim_x, grid_dim_y, grid_dim_z);

        ChartScaleTrigger trigger{};
        trigger.chart_id = node.chart_data.chart_id;
        trigger.max_bottleneck_pressure = max_p;
        trigger.requires_subdivision = (max_p > split_threshold) && (node.depth < max_depth_);
        trigger.requires_merge = false;

        triggers.push_back(trigger);
    }

    evaluate_merge_candidates(registry, triggers, merge_threshold);
    return triggers;
}

void AtlasManagerScalingExtension::subdivide_chart(
    uint32_t parent_id,
    std::unordered_map<uint32_t, ScaledLocalChartNode>& chart_registry)
{
    auto it = chart_registry.find(parent_id);
    if (it == chart_registry.end() || !it->second.is_leaf) return;

    ScaledLocalChartNode& parent = it->second;
    parent.is_leaf = false;
    parent.chart_data.is_active = false;

    float parent_r = parent.chart_data.radius;
    float child_r = parent_r * 0.5f;
    float offset = child_r;

    const float offsets[8][3] = {
        {-offset, -offset, -offset}, { offset, -offset, -offset},
        {-offset,  offset, -offset}, { offset,  offset, -offset},
        {-offset, -offset,  offset}, { offset, -offset,  offset},
        {-offset,  offset,  offset}, { offset,  offset,  offset}
    };

    for (int i = 0; i < 8; ++i) {
        uint32_t child_id = generate_next_chart_id();
        parent.children_ids[i] = child_id;

        ScaledLocalChartNode child_node{};
        child_node.chart_data.chart_id = child_id;
        child_node.chart_data.center[0] = parent.chart_data.center[0] + offsets[i][0];
        child_node.chart_data.center[1] = parent.chart_data.center[1] + offsets[i][1];
        child_node.chart_data.center[2] = parent.chart_data.center[2] + offsets[i][2];
        child_node.chart_data.radius = child_r;
        child_node.chart_data.is_active = true;

        child_node.parent_id = parent_id;
        child_node.is_leaf = true;
        child_node.depth = parent.depth + 1;
        std::fill(std::begin(child_node.children_ids), std::end(child_node.children_ids), 0xFFFFFFFF);

        chart_registry[child_id] = child_node;
    }
}

void AtlasManagerScalingExtension::merge_charts(
    uint32_t parent_id,
    std::unordered_map<uint32_t, ScaledLocalChartNode>& chart_registry)
{
    auto it = chart_registry.find(parent_id);
    if (it == chart_registry.end() || it->second.is_leaf) return;

    ScaledLocalChartNode& parent = it->second;

    for (int i = 0; i < 8; ++i) {
        uint32_t child_id = parent.children_ids[i];
        chart_registry.erase(child_id);
        parent.children_ids[i] = 0xFFFFFFFF;
    }

    parent.is_leaf = true;
    parent.chart_data.is_active = true;
}

} // namespace elysia::emulation
