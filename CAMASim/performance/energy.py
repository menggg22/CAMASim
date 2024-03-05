class EnergyEval:
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
        Initialize the EnergyEval component.
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

    def calculate_query_energy(self):
        """
        Calculate the energy consumption for a query operation.

        Returns:
            float: Total query energy consumption (in pJ).

        This method calculates the total energy consumption for a query operation, including array, interconnect, and peripheral energy.
        """
        array_energy = (
            (
                self.array_cost["interconnect"]["energy"]
                + self.array_cost["subarray"]["energy"]
            )
            * self.arch_config["SubarraysPerArray"]
            * min(self.arch_config["ArraysPerMat"], self.num_array)
            * min(self.arch_config["MatsPerBank"], self.num_mat)
            * self.num_bank
        )
        array_energy += (
            (self.array_cost["peripheral"]["energy"])
            * min(self.arch_config["ArraysPerMat"], self.num_array)
            * min(self.arch_config["MatsPerBank"], self.num_mat)
            * self.num_bank
        )

        array_peripheral_energy = self.estimate_peripheral_energy(
            self.array_peripherals
        )
        mat_peripheral_energy = self.estimate_peripheral_energy(self.mat_peripherals)
        bank_peripheral_energy = self.estimate_peripheral_energy(self.bank_peripherals)

        total_energy = (
            array_energy
            + array_peripheral_energy
            + mat_peripheral_energy
            + bank_peripheral_energy
        )
        icnt_energy = 0.1 * total_energy
        total_energy += icnt_energy

        print("Query Energy Breakdown:")
        print(f" - array energy: {array_energy}J")
        print(f" - peripheral energy: {array_peripheral_energy + mat_peripheral_energy + bank_peripheral_energy}J")
        # print(f" - peripheral energy: {array_peripheral_energy}J")
        print(f" - interconnect energy: {icnt_energy}J")
        return total_energy

    def calculate_write_energy(self):
        """
        Calculate the energy consumption for a write operation.

        Returns:
            float: Total write energy consumption (in pJ).

        This method calculates the total energy consumption for a write operation, including array, interconnect, and peripheral energy.
        """
        array_energy = (
            self.array_cost["write"]["energy"]
            * self.num_array
            * self.num_mat
            * self.num_bank
            + self.array_cost["interconnect"]["energy"]
            + self.array_cost["peripheral"]["energy"]
        )

        array_peripheral_energy = self.estimate_peripheral_energy(
            self.array_peripherals
        )
        mat_peripheral_energy = self.estimate_peripheral_energy(self.mat_peripherals)
        bank_peripheral_energy = self.estimate_peripheral_energy(self.bank_peripherals)

        total_energy = (
            array_energy
            + array_peripheral_energy
            + mat_peripheral_energy
            + bank_peripheral_energy
        )
        return total_energy

    def estimate_peripheral_energy(self, peripherals):
        """
        Estimate the total energy consumption of peripherals.

        Args:
            peripherals (dict): Dictionary of peripherals with their parameters.

        Returns:
            float: Total energy consumption of peripherals (in pJ).

        This method estimates the total energy consumption of peripherals based on the given peripherals and their parameters.
        """
        total_energy = 0
        for peripheral, params in peripherals.items():
            num = params["num"]
            size = params["size"]
            energy = self.peripheral_cost[peripheral]["energy"]
            total_energy += num * size * energy
        return total_energy
