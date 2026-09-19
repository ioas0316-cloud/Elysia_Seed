#ifndef SPATIOTEMPORAL_FLUX_SCALING_HPP
#define SPATIOTEMPORAL_FLUX_SCALING_HPP

#include "spatiotemporal_memory.hpp"
#include <vector>
#include <unordered_map>

namespace elysia::emulation {

/**
 * @brief 유량 및 유속 모니터링 셀 구조체
 */
struct FluxCell {
    float density;       ///< 현재 정보/상태 밀도 (rho)
    float velocity[3];   ///< 유속 벡터 (v)
    float divergence;    ///< 리만 발산 및 병목 누적치 (P_bottleneck)
};

/**
 * @brief 옥트리(Octree) 구조를 지원하는 확장 국소 차트 노드
 */
struct ScaledLocalChartNode {
    LocalChart chart_data;          ///< 기본 차트 ID, 중심 좌표, 반경, 활성 상태
    uint32_t parent_id{0xFFFFFFFF}; ///< 부모 차트 ID (최상위 노드일 경우 0xFFFFFFFF)
    uint32_t children_ids[8]{};     ///< 자식 차트 ID (리프 노드일 경우 모두 0xFFFFFFFF)
    bool is_leaf{true};             ///< 리프 노드 여부
    uint8_t depth{0};               ///< 차트 분할 깊이 (Max Depth 제어용)
};

/**
 * @brief MetricFieldEngine 동적 병목 연산 확장 인터페이스
 */
class MetricFieldEngineFluxExtension {
public:
    /**
     * @brief CUDA 커널: 리만 발산 \nabla_g \cdot (rho * v) 및 병목 지수 계산
     */
    static void launch_compute_flux_divergence_kernel(
        const MetricTensor3x3* d_g_field,
        const float* d_density,
        const float* d_velocity,
        float* d_bottleneck_out,
        size_t dim_x, size_t dim_y, size_t dim_z,
        float grid_spacing,
        cudaStream_t stream = 0);

    /**
     * @brief 병목 지수에 따른 계량 텐서 자동 동적 팽창/축소
     */
    static void launch_apply_bottleneck_metric_stress_kernel(
        MetricTensor3x3* d_g_field,
        const float* d_bottleneck_index,
        float eta_stress_coefficient,
        size_t total_cells,
        cudaStream_t stream = 0);
};

/**
 * @brief AtlasManager 동적 차트 재구성 확장 인터페이스
 */
class AtlasManagerScalingExtension {
public:
    struct ChartScaleTrigger {
        uint32_t chart_id;
        float max_bottleneck_pressure;
        bool requires_subdivision;
        bool requires_merge;
    };

    explicit AtlasManagerScalingExtension(size_t max_depth = 4);

    /**
     * @brief 차트별 병목 압력을 스캔하여 분할/병합 대상 차트 탐지
     */
    std::vector<ChartScaleTrigger> evaluate_chart_bottlenecks(
        const std::vector<ScaledLocalChartNode>& chart_nodes,
        const float* host_bottleneck_map,
        size_t grid_dim_x, size_t grid_dim_y, size_t grid_dim_z,
        float split_threshold,
        float merge_threshold);

    /**
     * @brief 병목 차트 8분할 (Subdivide Chart)
     */
    void subdivide_chart(
        uint32_t parent_id,
        std::unordered_map<uint32_t, ScaledLocalChartNode>& chart_registry);

    /**
     * @brief 비활성/희소 차트 병합 (Merge Charts)
     */
    void merge_charts(
        uint32_t parent_id,
        std::unordered_map<uint32_t, ScaledLocalChartNode>& chart_registry);

private:
    size_t max_depth_;
    uint32_t next_chart_id_;

    uint32_t generate_next_chart_id();

    float compute_max_bottleneck_in_chart(
        const LocalChart& chart,
        const float* bottleneck_map,
        size_t dim_x, size_t dim_y, size_t dim_z);

    void evaluate_merge_candidates(
        const std::unordered_map<uint32_t, ScaledLocalChartNode>& chart_registry,
        std::vector<ChartScaleTrigger>& triggers,
        float merge_threshold);
};

} // namespace elysia::emulation

#endif // SPATIOTEMPORAL_FLUX_SCALING_HPP
