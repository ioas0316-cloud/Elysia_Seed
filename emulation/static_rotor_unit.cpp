#include "spatiotemporal_memory.hpp"
#include <iostream>
#include <stdexcept>

namespace elysia::emulation {

StaticRotorUnit::StaticRotorUnit(size_t max_tag_capacity)
    : capacity_(max_tag_capacity) {}

StaticRotorUnit::~StaticRotorUnit() = default;

Rotor StaticRotorUnit::pin_evicted_phase(uint64_t cache_line_id, const Rotor& r_curr, const Rotor& r_base) {
    if (tag_array_.size() >= capacity_ && tag_array_.find(cache_line_id) == tag_array_.end()) {
        // Evict unpinned or oldest if full (simplified boundary check)
        auto it = tag_array_.begin();
        while (it != tag_array_.end() && it->second.is_pinned) {
            ++it;
        }
        if (it != tag_array_.end()) {
            tag_array_.erase(it);
        }
    }

    // Delta R = R_curr * R_base_rev
    Rotor r_base_rev = r_base.reverse();
    Rotor delta_r = r_curr.multiply(r_base_rev);

    RotorTagEntry entry;
    entry.delta_r = delta_r;
    entry.bg_drift_accum = 0.0f;
    entry.is_pinned = true;

    tag_array_[cache_line_id] = entry;
    return delta_r;
}

Rotor StaticRotorUnit::restore_active_phase(uint64_t cache_line_id, const Rotor& r_global_now) {
    auto it = tag_array_.find(cache_line_id);
    if (it == tag_array_.end()) {
        // If not found in tag array, return r_global_now
        return r_global_now;
    }

    // R_active = R_global_now * Delta R
    Rotor delta_r = it->second.delta_r;

    // Apply background drift if any
    if (it->second.bg_drift_accum != 0.0f) {
        float angle = it->second.bg_drift_accum;
        Rotor drift = { std::cos(angle * 0.5f), 0.0f, 0.0f, std::sin(angle * 0.5f) };
        delta_r = delta_r.multiply(drift);
    }

    Rotor restored = r_global_now.multiply(delta_r);
    return restored;
}

void StaticRotorUnit::update_background_drift(float delta_t, float omega_bg) {
    for (auto& pair : tag_array_) {
        if (pair.second.is_pinned) {
            pair.second.bg_drift_accum += omega_bg * delta_t;
        }
    }
}

} // namespace elysia::emulation
