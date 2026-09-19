"""
Elysia Spatiotemporal Memory Core Python Wrapper
Interoperability bridge connecting elysia_core Python Spine with C++ Emulator/Benchmarks
"""

import subprocess
import os
import sys

class SpatiotemporalEmulatorBridge:
    def __init__(self, src_path=None, binary_path=None):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if src_path is None:
            src_path = os.path.join(root_dir, "benchmarks", "benchmark_suite.cpp")
        if binary_path is None:
            binary_path = os.path.join(root_dir, "benchmarks", "benchmark_suite")

        self.root_dir = root_dir
        self.src_path = src_path
        self.binary_path = binary_path

    def compile_if_needed(self):
        if not os.path.exists(self.binary_path):
            cmd = [
                "g++", "-std=c++17", "-Wall", "-Wextra",
                "-I" + os.path.join(self.root_dir, "emulation"),
                "-I" + os.path.join(self.root_dir, "isa_compiler"),
                self.src_path,
                os.path.join(self.root_dir, "emulation", "static_rotor_unit.cpp"),
                os.path.join(self.root_dir, "emulation", "atlas_manager.cpp"),
                os.path.join(self.root_dir, "emulation", "virtual_tier_pipeline.cpp"),
                os.path.join(self.root_dir, "emulation", "metric_field_engine.cpp"),
                os.path.join(self.root_dir, "emulation", "spatiotemporal_flux_scaling.cpp"),
                "-o", self.binary_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Compilation failed: {res.stderr}")

    def run_benchmarks(self):
        self.compile_if_needed()
        result = subprocess.run([self.binary_path], capture_output=True, text=True)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

if __name__ == "__main__":
    bridge = SpatiotemporalEmulatorBridge()
    res = bridge.run_benchmarks()
    print("Benchmark execution status code:", res["returncode"])
    print(res["stdout"])
