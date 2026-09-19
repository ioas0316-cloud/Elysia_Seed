#ifndef SPATIOTEMPORAL_MEMORY_HPP
#define SPATIOTEMPORAL_MEMORY_HPP

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include "cuda_stub.hpp"
#endif

#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace elysia::emulation {

// ============================================================================
// 0. 기본 기하 구조체 (Geometric Primitive Types)
// ============================================================================

/**
 * @brief Cl(3,0) 기하 대수 기반 스피너 / 회전자 (Rotor)
 * R = a + b(e12) + c(e23) + d(e31)
 */
struct alignas(16) Rotor {
    float scalar;       ///< 스칼라 성분 (a)
    float bivector_xy;  ///< e12 이중벡터 성분
    float bivector_yz;  ///< e23 이중벡터 성분
    float bivector_zx;  ///< e31 이중벡터 성분

    __host__ __device__ inline Rotor reverse() const {
        return { scalar, -bivector_xy, -bivector_yz, -bivector_zx };
    }

    /**
     * @brief 스피너 곱 R3 = R1 * R2 (Cl3,0 이중벡터 외적/내적 결합법칙)
     */
    __host__ __device__ inline Rotor multiply(const Rotor& other) const {
        Rotor out;
        out.scalar      = scalar * other.scalar
                        - bivector_xy * other.bivector_xy
                        - bivector_yz * other.bivector_yz
                        - bivector_zx * other.bivector_zx;

        out.bivector_xy = scalar * other.bivector_xy + bivector_xy * other.scalar
                        - bivector_yz * other.bivector_zx + bivector_zx * other.bivector_yz;

        out.bivector_yz = scalar * other.bivector_yz + bivector_yz * other.scalar
                        - bivector_zx * other.bivector_xy + bivector_xy * other.bivector_zx;

        out.bivector_zx = scalar * other.bivector_zx + bivector_zx * other.scalar
                        - bivector_xy * other.bivector_yz + bivector_yz * other.bivector_xy;
        return out;
    }
};

/**
 * @brief 국소 시공간 계량 텐서 g_mem (3x3 대칭 텐서)
 */
struct alignas(16) MetricTensor3x3 {
    float g[3][3];
};

/**
 * @brief 국소 차트(Chart) 식별 및 바운딩 림
 */
struct LocalChart {
    uint32_t chart_id;
    float center[3];
    float radius;
    bool is_active;
};

// ============================================================================
// 1. StaticRotorUnit (SRU) - 정적 로터 보조 메모리 인터페이스
// ============================================================================

/**
 * @brief Cache-RAM 스와핑 시 위상 차이(Delta Rotor)를 SRAM 레지스터에 보존 및 복원
 */
class StaticRotorUnit {
public:
    struct RotorTagEntry {
        Rotor delta_r;           ///< 보존된 위상 오프셋 (Delta Rotor)
        float bg_drift_accum;    ///< 백그라운드 위상 드리프트 축적량
        bool is_pinned;          ///< SRAM Pinning 여부
    };

    explicit StaticRotorUnit(size_t max_tag_capacity);
    ~StaticRotorUnit();

    /**
     * @brief Eviction 발생 시: Delta Rotor 계산 및 SRAM Tag Register에 Pinning
     * @param cache_line_id 캐시 라인 태그 ID
     * @param r_curr 방출 시점의 실시간 스피너
     * @param r_base 전역 기저 프레임 스피너
     */
    __host__ Rotor pin_evicted_phase(uint64_t cache_line_id, const Rotor& r_curr, const Rotor& r_base);

    /**
     * @brief Fetch 발생 시: Pinned Delta 기반 위상 프레임 즉시 복원
     * @param cache_line_id 캐시 라인 태그 ID
     * @param r_global_now 현재 스케일의 전역 위상
     */
    __host__ Rotor restore_active_phase(uint64_t cache_line_id, const Rotor& r_global_now);

    /**
     * @brief 하위 계층 대기 중 백그라운드 위상 드리프트 연속성 업데이트
     */
    void update_background_drift(float delta_t, float omega_bg);

    size_t get_pinned_count() const { return tag_array_.size(); }

private:
    size_t capacity_;
    std::unordered_map<uint64_t, RotorTagEntry> tag_array_;
};

// ============================================================================
// 2. MetricFieldEngine - 리만 계량 텐서 및 가소성 CUDA 연산 엔진
// ============================================================================

/**
 * @brief GPU 기반 g_mem 텐서 필드 연산, 경사 흐름 계산 및 SSD 가소성 각인 엔진
 */
class MetricFieldEngine {
public:
    MetricFieldEngine(size_t grid_dim_x, size_t grid_dim_y, size_t grid_dim_z);
    ~MetricFieldEngine();

    /**
     * @brief 비동기 CUDA 커널: 리만 경사 흐름 (\nabla V) 계산
     */
    void launch_riemannian_gradient_kernel(
        const float* d_pos_x,
        float* d_grad_out,
        size_t num_particles,
        cudaStream_t stream = 0);

    /**
     * @brief 비동기 CUDA 커널: 신규 궤적 및 회전 토크 기반 g_mem 변형 각인 (가소성)
     */
    void launch_plasticity_update_kernel(
        const float* d_pos_x,
        const float* d_velocity,
        const Rotor* d_torques,
        float alpha, float beta,
        size_t num_particles,
        cudaStream_t stream = 0);

    /**
     * @brief 백그라운드 자연 풍화 확산 커널 (Entropy Decay & Diffusion)
     */
    void launch_entropy_decay_kernel(
        float gamma, float diffusion_D,
        float delta_t, cudaStream_t stream = 0);

    // Device Memory Pointers
    MetricTensor3x3* get_device_metric_ptr() const { return d_g_mem_field_; }
    size_t get_grid_dim_x() const { return grid_x_; }
    size_t get_grid_dim_y() const { return grid_y_; }
    size_t get_grid_dim_z() const { return grid_z_; }
    size_t get_total_cells() const { return total_cells_; }

private:
    size_t grid_x_, grid_y_, grid_z_;
    size_t total_cells_;
    MetricTensor3x3* d_g_mem_field_;  ///< CUDA GPU 메인 계량 텐서 버퍼
};

// ============================================================================
// 3. AtlasManager - 시공간 아틀라스 및 국소 차트 파편화 제어기
// ============================================================================

/**
 * @brief 전역 공간을 국소 차트로 구획화하여 연산 불필요 영역을 Freeze 처리
 */
class AtlasManager {
public:
    explicit AtlasManager(size_t initial_chart_count);
    ~AtlasManager();

    /**
     * @brief 현재 궤적 위치 x(t)에 기인한 활성 차트(Active Charts) 쿼리
     */
    std::vector<LocalChart> query_active_charts(const float current_pos[3]);

    /**
     * @brief 차트 간 이행 지도 (Transition Map) 위상 연산
     */
    Rotor compute_chart_transition_rotor(uint32_t src_chart_id, uint32_t dst_chart_id);

    /**
     * @brief 인과적 영향권 밖 차트들의 계산 정지 (Freeze) 제어
     */
    void update_chart_active_states(const std::vector<uint32_t>& active_ids);

    const std::vector<LocalChart>& get_all_charts() const { return charts_; }

private:
    std::vector<LocalChart> charts_;
    std::vector<uint8_t> active_mask_buffer_;
};

// ============================================================================
// 4. VirtualTierPipeline - Cache-RAM-SSD 레이턴시 및 비동기 각인 에뮬레이션
// ============================================================================

class VirtualTierPipeline {
public:
    VirtualTierPipeline(double cache_latency_ns, double ram_latency_ns, double ssd_latency_ns);
    ~VirtualTierPipeline();

    /**
     * @brief Phase-Lock 발동 시 SSD 구역으로 g_mem 변형 패킷 비동기 DMA 전송
     */
    void trigger_async_dma_imprint(uint32_t chart_id, const MetricTensor3x3& metric_delta);

    /**
     * @brief 지연 모사 및 파이프라인 처리 시뮬레이션
     */
    double simulate_tier_transfer(size_t data_bytes, int source_tier, int target_tier);

    size_t get_pending_dma_count() const { return dma_queue_count_; }

private:
    double cache_latency_;
    double ram_latency_;
    double ssd_latency_;
    size_t dma_queue_count_;
};

} // namespace elysia::emulation

#endif // SPATIOTEMPORAL_MEMORY_HPP
