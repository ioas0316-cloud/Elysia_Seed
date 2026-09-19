#ifndef GEOMETRIC_ISA_HPP
#define GEOMETRIC_ISA_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

namespace elysia::isa {

/**
 * @brief 기하 대수 전용 명령어 Opcode (Geometric Algebra ISA)
 */
enum class Opcode : uint8_t {
    PLOCK       = 0x10,  ///< 스피너 위상 고정 (Phase-Lock)
    ROTOR_PIN   = 0x11,  ///< 로터 오프셋 레지스터 Pinning (Cache Eviction 연동)
    GEOM_SWAP   = 0x12,  ///< 기하학적 연속성 보존 스왑
    METRIC_W    = 0x20,  ///< g_mem 계량 텐서 변형 각인
    METRIC_R    = 0x21,  ///< g_mem 계량 텐서 역행렬 읽기
    SPIN_ROT    = 0x30,  ///< Cl(3,0) 이중벡터 회전 토크 적용
    CHART_ACT   = 0x40,  ///< 국소 차트 동적 활성화
    CHART_FREEZE= 0x41   ///< 국소 차트 연산 정지 (Freeze)
};

/**
 * @brief 32-bit ISA Instruction Word Encoding
 * Format: [Opcode (8bit) | RegDest (8bit) | RegSrc1 (8bit) | RegSrc2 / Flags (8bit)]
 */
struct Instruction {
    Opcode op;
    uint8_t r_dst;
    uint8_t r_src1;
    uint8_t r_src2_or_flags;

    uint32_t encode() const {
        return (static_cast<uint32_t>(op) << 24) |
               (static_cast<uint32_t>(r_dst) << 16) |
               (static_cast<uint32_t>(r_src1) << 8) |
               static_cast<uint32_t>(r_src2_or_flags);
    }

    static Instruction decode(uint32_t raw) {
        Instruction inst;
        inst.op = static_cast<Opcode>((raw >> 24) & 0xFF);
        inst.r_dst = static_cast<uint8_t>((raw >> 16) & 0xFF);
        inst.r_src1 = static_cast<uint8_t>((raw >> 8) & 0xFF);
        inst.r_src2_or_flags = static_cast<uint8_t>(raw & 0xFF);
        return inst;
    }

    std::string disassemble() const {
        std::ostringstream ss;
        switch (op) {
            case Opcode::PLOCK:        ss << "PLOCK       "; break;
            case Opcode::ROTOR_PIN:    ss << "ROTOR_PIN   "; break;
            case Opcode::GEOM_SWAP:    ss << "GEOM_SWAP   "; break;
            case Opcode::METRIC_W:     ss << "METRIC_W    "; break;
            case Opcode::METRIC_R:     ss << "METRIC_R    "; break;
            case Opcode::SPIN_ROT:     ss << "SPIN_ROT    "; break;
            case Opcode::CHART_ACT:    ss << "CHART_ACT   "; break;
            case Opcode::CHART_FREEZE: ss << "CHART_FREEZE"; break;
            default:                   ss << "UNKNOWN     "; break;
        }
        ss << "R" << (int)r_dst << ", R" << (int)r_src1 << ", R" << (int)r_src2_or_flags;
        return ss.str();
    }
};

/**
 * @brief 위상 인식 컴파일러 패스 (Phase-Aware Compiler Pass)
 * High-level memory transfer ops -> Insert ROTOR_PIN & GEOM_SWAP
 */
class PhaseAwareCompilerPass {
public:
    struct HighLevelOp {
        enum class Type { MEM_EVICT, MEM_FETCH, COMPUTE_GRADIENT, CHART_TRANSITION };
        Type type;
        uint64_t memory_tag;
        uint32_t chart_id;
    };

    std::vector<Instruction> compile(const std::vector<HighLevelOp>& ops) {
        std::vector<Instruction> program;

        for (const auto& op : ops) {
            switch (op.type) {
                case HighLevelOp::Type::MEM_EVICT:
                    // Automatically insert ROTOR_PIN before eviction
                    program.push_back({ Opcode::ROTOR_PIN, 1, 0, static_cast<uint8_t>(op.memory_tag & 0xFF) });
                    program.push_back({ Opcode::GEOM_SWAP, 2, 1, 0 });
                    break;

                case HighLevelOp::Type::MEM_FETCH:
                    // Automatically insert PLOCK and restore phase
                    program.push_back({ Opcode::PLOCK, 1, 0, static_cast<uint8_t>(op.memory_tag & 0xFF) });
                    program.push_back({ Opcode::GEOM_SWAP, 0, 1, 2 });
                    break;

                case HighLevelOp::Type::COMPUTE_GRADIENT:
                    program.push_back({ Opcode::METRIC_R, 3, 0, 0 });
                    program.push_back({ Opcode::SPIN_ROT, 4, 3, 0 });
                    break;

                case HighLevelOp::Type::CHART_TRANSITION:
                    program.push_back({ Opcode::CHART_ACT, static_cast<uint8_t>(op.chart_id & 0xFF), 0, 0 });
                    break;
            }
        }

        return program;
    }
};

} // namespace elysia::isa

#endif // GEOMETRIC_ISA_HPP
