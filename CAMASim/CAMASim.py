# The CAMASim class is designed to facilitate operations and performance evaluation of CAM-based accelerators systems.
# It provides functionalities for both function simulation and performance evaluatiob based on user-defined configurations.

from CAMASim.arch.ArchEstimator import ArchEstimator
from CAMASim.function.FunctionSimulator import FunctionSimulator
from CAMASim.performance.PerformanceEvaluator import PerformanceEvaluator


class CAMASim:
    def __init__(self, config):
        # Initialize the CAMASim class with configuration settings.

        # Extract configuration settings
        query_config = config["query"]
        arch_config = config["arch"]
        array_config = config["array"]
        cell_config = config["cell"]
        noise_config = config["cell"]["writeNoise"]

        self.query_config = query_config

        # Initialize Function Simulator, CAM Architecture Estimator, and Performance Evaluator
        self.func_sim = FunctionSimulator(
            array_config, query_config, cell_config, noise_config
        )
        self.cam_arch = ArchEstimator(arch_config, array_config)
        self.perf_eval = PerformanceEvaluator(arch_config, array_config, cell_config)

    def write(self, data):
        # Perform a write operation.
        print("*** Write Data to CAM Arrays ***")

        latency, energy = None, None

        # Function Simulation
        if self.query_config["FuncSim"] == 1:
            self.func_sim.write(data)

        # Performance Evaluation
        if self.query_config["PerfEval"] == 1:
            # Estimate CAM architecture based on data
            cam_arch = self.cam_arch.estimate_cam_architecture(data)
            self.perf_eval.initialize(cam_arch)
            # Measure latency and energy consumption during write operation
            latency, energy = self.perf_eval.write(data)
            print("--> Write Latency (ns), Energy (J):", latency, energy)

        return latency, energy

    def query(self, data):
        # Perform a query operation.
        print("*** Query CAM Arrays ***")

        # Function Simulation
        results, latency, energy = None, None, None
        if self.query_config["FuncSim"] == 1:
            # Simulate query operation and obtain results
            results = self.func_sim.query(data)

        # Performance Evaluation
        if self.query_config["PerfEval"] == 1:
            # Measure latency and energy consumption during query operation
            latency, energy = self.perf_eval.query(data)
            print("--> Query Latency (ps), Energy (J):", latency, energy)

        return results, latency, energy
