class LatencyEval:
    def __init__(
        self,
        arch_config,
        array_config,
        array_cost,
        peripheral_cost,
        num_array,
        num_mat,
        num_bank,
        array_peripherals,
        mat_peripherals,
        bank_peripherals,
    ):
        """
        Initialize the LatencyEval component.
        """
        self.arch_config = arch_config
        self.array_config = array_config
        self.array_cost = array_cost
        self.peripheral_cost = peripheral_cost
        self.num_array = num_array
        self.num_mat = num_mat
        self.num_bank = num_bank
        self.array_peripherals = array_peripherals
        self.mat_peripherals = mat_peripherals
        self.bank_peripherals = bank_peripherals

    def calculate_write_latency(self):
        """
        Calculate the latency for a write operation.

        Returns:
            float: Total write latency (in nanoseconds).

        This method calculates the total latency for a write operation, including array, interconnect, and peripheral latencies.
        """
        array_latency = (
            self.array_config["row"] * self.array_cost["write"]["latency"]
            + self.array_cost["interconnect"]["latency"]
            + self.array_cost["peripheral"]["latency"]
        )

        array_peripheral_latency = self.estimate_peripheral_latency(
            self.array_peripherals
        )
        mat_peripheral_latency = self.estimate_peripheral_latency(self.mat_peripherals)
        bank_peripheral_latency = self.estimate_peripheral_latency(
            self.bank_peripherals
        )

        total_latency = (
            array_latency
            + array_peripheral_latency
            + mat_peripheral_latency
            + bank_peripheral_latency
        )
        return total_latency

    def calculate_query_latency(self):
        """
        Calculate the latency for a query operation.

        Returns:
            float: Total query latency (in nanoseconds).

        This method calculates the total latency for a query operation, including array, interconnect, and peripheral latencies. Additional communication cost is added.
        """
        array_latency = (
            self.array_cost["subarray"]["latency"]
            + self.array_cost["interconnect"]["latency"]
            + self.array_cost["peripheral"]["latency"]
        )
        array_peripheral_latency = self.estimate_peripheral_latency(
            self.array_peripherals
        )
        mat_peripheral_latency = self.estimate_peripheral_latency(
            self.mat_peripherals
        ) * (self.arch_config["ArraysPerMat"] - 1)
        bank_peripheral_latency = self.estimate_peripheral_latency(
            self.bank_peripherals
        ) * (self.arch_config["MatsPerBank"] - 1)
        total_latency = (
            array_latency
            + array_peripheral_latency
            + mat_peripheral_latency
            + bank_peripheral_latency
        )
        icnt_latency = 0.1 * total_latency
        total_latency += icnt_latency

        print("Query Latency Breakdown:")
        print(f" - array latency: {array_latency}ps")
        print(
            f" - peripheral latency: {array_peripheral_latency + mat_peripheral_latency + bank_peripheral_latency}ps"
        )
        print(f'- interconnect latency: {icnt_latency}ps')

        
        return total_latency

    def estimate_peripheral_latency(self, peripherals):
        """
        Estimate the total peripheral latency.

        Args:
            peripherals (dict): Dictionary of peripherals with their parameters.

        Returns:
            float: Total peripheral latency (in nanoseconds).

        This method estimates the total peripheral latency based on the given peripherals and their parameters.
        """
        total_latency = 0
        for peripheral, params in peripherals.items():
            num = params["num"]
            size = params["size"]
            # latency = self.peripheral_cost[peripheral]["latency"] * size
            latency = self.peripheral_cost[peripheral]["latency"]
            total_latency += latency
        return total_latency
