import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
K = 0.307075
m_muon = 105.658
m_e = 0.511
LAYER_THICKNESS = 0.05
muon_energy = 2000
max_depth = 1500

thicknesses_mm = [1, 2, 4, 6, 8, 10]
thicknesses_cm = [t / 10.0 for t in thicknesses_mm]
N_muons = 100

def sternheimer_delta(beta_gamma, x0, x1, C, a, m):
    if beta_gamma <= 0:
        return 0.0
    x = np.log10(beta_gamma)
    if x < x0:
        return 0.0
    elif x < x1:
        return 2 * np.log(10) * x - C
    else:
        return 2 * np.log(10) * x - C + a * (x1 - x)**m

def bb(E, material):
    if E <= 0:
        return 0.0
    total_E = E + m_muon
    gamma = total_E / m_muon
    beta2 = 1 - 1 / gamma**2
    if beta2 <= 0:
        return 0.0
    beta = np.sqrt(beta2)
    beta_gamma = beta * gamma

    rho = material["rho"]
    Z_over_A = material["Z_over_A"]
    I = material["I"]

    mass_ratio = m_e / m_muon
    Tmax = (2 * m_e * beta2 * gamma**2) / (1 + 2*gamma*mass_ratio + mass_ratio**2)
    arg = (2 * m_e * beta2 * gamma**2 * Tmax) / I**2
    if arg <= 1:
        return 0.0

    delta = sternheimer_delta(beta_gamma,
                              material["x0"], material["x1"],
                              material["C"], material["a"], material["m"])

    stopping = (K * rho * Z_over_A / beta2 *
                (0.5 * np.log(arg) - beta2 - delta / 2))
    return max(stopping, 0.0)

def fluctuate(mean_dedx):
    """Gaussian approximation to Landau fluctuations.
    Applied to dE/dx BEFORE Birks law (correct physics)."""
    sigma = 0.25 * mean_dedx
    return max(np.random.normal(mean_dedx, sigma), 0.0)

def simulate(material_name, energy, max_depth_local, fluctuate_energy_loss=True):
    material = materials[material_name]
    E = energy
    x = 0.0
    depths, energies, dedx_list, dose, photons = [], [], [], [], []

    while x < max_depth_local and E > 0:
        dedx_mean = bb(E, material)

        if fluctuate_energy_loss:
            dedx = fluctuate(dedx_mean)
        else:
            dedx = dedx_mean

        dE = dedx * LAYER_THICKNESS

        if dedx > 0:
            effective_yield = photon_yield[material_name] / (1.0 + material["kB"] * dedx)
            photons_step = effective_yield * dE
        else:
            photons_step = 0.0

        E -= dE
        if E < 0:
            E = 0.0

        depths.append(x)
        energies.append(E)
        dedx_list.append(dedx)
        dose.append(dE)
        photons.append(photons_step)
        x += LAYER_THICKNESS

    return (np.array(depths), np.array(energies), np.array(dedx_list),
            np.array(dose), np.array(photons))

materials = {
    "CaF2": {
        "rho": 3.18,
        "Z_over_A": 38 / 78.07,
        "I": 166e-6,
        "kB": 0.0080,
        "x0": 0.201, "x1": 2.871, "C": 3.20, "a": 0.080, "m": 3.0 
    },
    "Hydrogel": {
        "rho": 1.10,
        "Z_over_A": 0.555,
        "I": 75e-6,
        "kB": 0.0200,
        "x0": 0.240, "x1": 2.800, "C": 3.50, "a": 0.091, "m": 3.0 
    },
    "ZnO_Epoxy_PPO": {
        "rho": 1.19,
        "Z_over_A": (0.10 * 38 / 81.38) + (0.89 * 6 / 12) + (0.01 * 8 / 16),
        "I": 80e-6,
        "kB": 0.0150,
        "x0": 0.201, "x1": 2.871, "C": 3.20, "a": 0.080, "m": 3.0  
    },
    "Plastic": {
        "rho": 1.03,
        "Z_over_A": 0.541,
        "I": 64.7e-6,
        "kB": 0.0120,
        "x0": 0.201, "x1": 2.871, "C": 3.20, "a": 0.080, "m": 3.0 
    }
}

photon_yield = {
    "CaF2": 24000,
    "Hydrogel": 200,
    "ZnO_Epoxy_PPO": 420,
    "Plastic": 9100
}

plt.figure(figsize=(10, 6))
for mat in materials:
    depths, _, dedx, _, _ = simulate(mat, muon_energy, 1.0, False)
    plt.plot(depths*10, dedx/10, label=mat)
plt.title("Stopping Power vs Depth")
plt.xlabel("Depth (mm)")
plt.ylabel("dE/dx (MeV/mm)")
plt.legend()
plt.grid()
plt.show()

print("\nEstimated muon ranges (2 GeV):")
for mat_name in materials:
    depths, E_arr, _, _, _ = simulate(mat_name, muon_energy, max_depth, False)
    if E_arr[-1] <= 0:
        print(f" {mat_name}: ~ {round(depths[-1], 2)} mm (stopped)")
    else:
        print(f" {mat_name}: > {max_depth} mm (not stopped)")

deposited_data = {}
deposited_std = {}
photon_data = {}
photon_std = {}

print("\nThickness dependence:")
for mat_name in materials:
    print(f"\n{mat_name} (MIP dE/dx = {bb(muon_energy, materials[mat_name])/10:.4f} MeV/mm)")
    avg_deps, std_deps, avg_photons, std_photons = [], [], [], []
    for thick_mm in thicknesses_mm:
        total_deps = []
        total_photons_list = []
        for _ in range(N_muons):
            _, _, _, dose, photons = simulate(mat_name, muon_energy, thick_mm, True)
            total_deps.append(np.sum(dose))
            total_photons_list.append(np.sum(photons))
        avg_deps.append(np.mean(total_deps))
        std_deps.append(np.std(total_deps))
        avg_photons.append(np.mean(total_photons_list))
        std_photons.append(np.std(total_photons_list))
    deposited_data[mat_name] = np.array(avg_deps)
    deposited_std[mat_name] = np.array(std_deps)
    photon_data[mat_name] = np.array(avg_photons)
    photon_std[mat_name] = np.array(std_photons)

plt.figure(figsize=(10, 6))
for mat in materials:
    plt.errorbar(thicknesses_mm, deposited_data[mat], yerr=deposited_std[mat],
                 marker='o', capsize=4, label=mat)
plt.title("Energy Deposition vs Thickness")
plt.xlabel("Thickness (mm)")
plt.ylabel("Deposited Energy (MeV)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
for mat in materials:
    plt.errorbar(thicknesses_mm, photon_data[mat], yerr=photon_std[mat],
                 marker='o', capsize=4, label=mat)
plt.title("Quenched Photon Production vs Thickness")
plt.xlabel("Thickness (mm)")
plt.ylabel("Photons produced")
plt.legend()
plt.grid()
plt.show()

relative_photon_yield = {}
relative_photon_yield_std = {}
plastic_ph = photon_data["Plastic"]
plastic_ph_std = photon_std["Plastic"]

for mat in materials:
    rel = photon_data[mat] / plastic_ph
    rel_std = rel * np.sqrt((photon_std[mat]/photon_data[mat])**2 +
                            (plastic_ph_std/plastic_ph)**2)
    relative_photon_yield[mat] = rel
    relative_photon_yield_std[mat] = rel_std

plt.figure(figsize=(10, 6))
for mat in materials:
    plt.errorbar(thicknesses_mm, relative_photon_yield[mat],
                 yerr=relative_photon_yield_std[mat], marker='o', capsize=4, label=mat)

exp_mm = [1, 2, 4, 6, 8]
plt.scatter(exp_mm, [0.90, 1.003, 1.003, 1.003, 1.003], color='darkred', s=80,
            label='Exp. Nanocomposite')
plt.scatter(exp_mm, [1.15, 0.95, 1.05, 1.000, 1.000], color='darkblue', s=80,
            label='Exp. Hydrogel')
plt.title("Relative Photon Yield")
plt.xlabel("Thickness (mm)")
plt.ylabel("Relative Yield")
plt.legend()
plt.grid()
plt.show()

effective_yield = {}
for mat in materials:
    effective_yield[mat] = photon_data[mat] / deposited_data[mat]

plt.figure(figsize=(10, 6))
for mat in materials:
    plt.plot(thicknesses_mm, effective_yield[mat], 'o-', label=mat)
plt.title("Effective Quenched Light Yield")
plt.xlabel("Thickness (mm)")
plt.ylabel("ph/MeV")
plt.legend()
plt.grid()
plt.show()

print("Mean relative photon yield (quenched, Plastic=1):")
for mat in ["ZnO_Epoxy_PPO", "Hydrogel", "Plastic", "CaF2"]:
    mean_rel = np.mean(relative_photon_yield[mat])
    print(f"  {mat}: {mean_rel:.4f}")

print("\nMean effective yield (ph/MeV):")
for mat in ["ZnO_Epoxy_PPO", "Hydrogel", "Plastic", "CaF2"]:
    print(f"  {mat}: {np.mean(effective_yield[mat]):.0f}")