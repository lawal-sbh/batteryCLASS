# src/battery_model.py

class Battery:
    def __init__(self, name, max_power, capacity, efficiency=0.95, initial_soc=0.5):
        self.name = name
        self.max_power = max_power
        self.capacity = capacity
        self.efficiency = efficiency
        self.soc = initial_soc

    def charge(self, power_mw, duration_hours):
        actual_power = min(power_mw, self.max_power)
        energy_input = actual_power * duration_hours * self.efficiency
        new_energy = (self.soc * self.capacity) + energy_input
        self.soc = min(new_energy / self.capacity, 1.0)
        return actual_power

    def discharge(self, power_mw, duration_hours):
        actual_power = min(power_mw, self.max_power)
        energy_output = actual_power * duration_hours
        new_energy = (self.soc * self.capacity) - energy_output
        self.soc = max(new_energy / self.capacity, 0)
        return actual_power

    def get_status(self):
        return {
            'soc': self.soc,
            'available_energy_mwh': self.soc * self.capacity,
            'max_charge_mw': self.max_power,
            'max_discharge_mw': self.max_power
        }

# Test code at the bottom
if __name__ == "__main__":
    print("Testing Battery class directly...")
    b = Battery("Direct_Test", 50, 100)
    b.charge(30, 2)
    print(f"Direct test SOC: {b.soc}")