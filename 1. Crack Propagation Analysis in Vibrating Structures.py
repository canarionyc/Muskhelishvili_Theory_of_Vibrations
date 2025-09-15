# %%
import numpy as np
import sympy as sp
from scipy import integrate

# Define symbolic variables
z, t = sp.symbols('z t', complex=True)
omega = sp.symbols('omega', real=True, positive=True)


# Muskhelishvili complex potentials for a crack
def complex_potentials_crack(z, a, load_amplitude, omega, t):
	"""
	Muskhelishvili complex potentials for a crack of length 2a under dynamic loading

	Parameters:
	- z: complex coordinate
	- a: half-length of crack
	- load_amplitude: amplitude of applied load
	- omega: circular frequency of vibration
	- t: time

	Returns:
	- phi, psi: Muskhelishvili complex potentials
	"""
	# Complex potentials for crack problem
	phi = load_amplitude * sp.exp(sp.I * omega * t) * z / sp.sqrt(z ** 2 - a ** 2)
	psi = -load_amplitude * sp.exp(sp.I * omega * t) * a ** 2 / (sp.sqrt(z ** 2 - a ** 2) * (z ** 2 - a ** 2))

	return phi, psi


# Calculate dynamic stress intensity factor
def dynamic_stress_intensity_factor(a, load_amplitude, omega, t):
	"""
	Calculates the dynamic stress intensity factor for mode I crack

	Parameters:
	- a: half-length of crack
	- load_amplitude: amplitude of applied load
	- omega: circular frequency of vibration
	- t: time

	Returns:
	- K_I: Mode I stress intensity factor
	"""
	# For mode I, K_I is related to the limit of phi as z approaches a
	K_I = load_amplitude * np.sqrt(np.pi * a) * np.cos(omega * t)
	return K_I


# %% Example calculation
crack_length = 0.01  # 1 cm crack (half-length = 0.005 m)
load_amplitude = 1e6  # 1 MPa
freq = 100  # 100 Hz
omega_val = 2 * np.pi * freq

# Calculate K_I over one vibration cycle
time_points = np.linspace(0, 1 / freq, 100)
K_I_values = [dynamic_stress_intensity_factor(crack_length / 2, load_amplitude,
                                              omega_val, t) for t in time_points]


# Energy release rate calculation (G = K_I^2/E for plane stress)
def energy_release_rate(K_I, E):
	"""Calculate energy release rate for mode I"""
	return K_I ** 2 / E


# Young's modulus for steel
E_steel = 200e9  # 200 GPa
G_values = [energy_release_rate(k, E_steel) for k in K_I_values]

#%% plot Dynamic Stress Intensity Factor vs Time
import matplotlib.pyplot as plt

# Plot stress intensity factor over time
plt.figure(figsize=(10, 6))
plt.plot(time_points, K_I_values, 'b-', linewidth=2)
plt.title('Dynamic Stress Intensity Factor vs Time')
plt.xlabel('Time (s)')
plt.ylabel('K_I (Pa·m^(1/2))')
plt.grid(True)
plt.show()

#%% Plot energy release rate over time
plt.figure(figsize=(10, 6))
plt.plot(time_points, G_values, 'r-', linewidth=2)
plt.title('Energy Release Rate vs Time')
plt.xlabel('Time (s)')
plt.ylabel('G (J/m²)')
plt.grid(True)
plt.show()

#%% Visualize stress field around the crack
def calculate_stress_field(x_range, y_range, crack_length, load_amplitude, omega, t):
    """Calculate stress field around a crack using complex potentials"""
    a = crack_length / 2
    K_I = load_amplitude * np.sqrt(np.pi * a) * np.cos(omega * t)

    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    sigma_y = np.zeros_like(X)

    for i in range(len(y)):
        for j in range(len(x)):
            # Skip points on the crack
            if abs(Y[i, j]) < 1e-10 and abs(X[i, j]) <= a:
                sigma_y[i, j] = np.nan
            else:
                # Calculate r and theta relative to right crack tip
                r_right = np.sqrt((X[i, j] - a)**2 + Y[i, j]**2)
                theta_right = np.arctan2(Y[i, j], (X[i, j] - a))

                # Calculate r and theta relative to left crack tip
                r_left = np.sqrt((X[i, j] + a)**2 + Y[i, j]**2)
                theta_left = np.arctan2(Y[i, j], (X[i, j] + a))

                # Use singular solution near crack tips
                if r_right < 0.25 * crack_length:
                    sigma_y[i, j] = K_I / np.sqrt(2*np.pi*r_right) * np.cos(theta_right/2) * (1 + np.sin(theta_right/2) * np.sin(3*theta_right/2))
                elif r_left < 0.25 * crack_length:
                    sigma_y[i, j] = K_I / np.sqrt(2*np.pi*r_left) * np.cos(theta_left/2) * (1 + np.sin(theta_left/2) * np.sin(3*theta_left/2))
                else:
                    # Far-field stress
                    sigma_y[i, j] = load_amplitude * np.cos(omega * t)

    return X, Y, sigma_y

#%% Plot stress field at t=0 (maximum loading)
t = 0
X, Y, sigma_y = calculate_stress_field(
    x_range=[-0.02, 0.02],
    y_range=[-0.01, 0.01],
    crack_length=crack_length,
    load_amplitude=load_amplitude,
    omega=omega_val,
    t=t
)

plt.figure(figsize=(12, 8))
levels = np.linspace(0, 3*load_amplitude, 20)
contour = plt.contourf(X, Y, sigma_y, levels=levels, cmap='jet')
plt.colorbar(label='σ_y (Pa)')
plt.title(f'Normal Stress σ_y Around Crack at t = {t:.4f}s')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.grid(True)
plt.plot([-crack_length/2, crack_length/2], [0, 0], 'k-', linewidth=3)  # Draw the crack
plt.show()

#%% Visualize stress singularity at the crack tip
r_values = np.logspace(-4, -2, 100)  # Distance from crack tip in meters
theta_values = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]  # Different angles

plt.figure(figsize=(10, 8))
for theta in theta_values:
    K_I = dynamic_stress_intensity_factor(crack_length/2, load_amplitude, omega_val, 0)
    sigma_y = K_I / np.sqrt(2*np.pi*r_values) * np.cos(theta/2) * (1 + np.sin(theta/2) * np.sin(3*theta/2))
    plt.loglog(r_values, sigma_y, linewidth=2, label=f'θ = {theta:.2f} rad')

plt.title('Stress Singularity Near Crack Tip')
plt.xlabel('Distance from Crack Tip (m)')
plt.ylabel('σ_y (Pa)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

#%% Visualize stress distribution around crack tip in polar coordinates
theta = np.linspace(0, 2*np.pi, 100)
r = 0.001  # Fixed distance from crack tip (1 mm)
K_I = dynamic_stress_intensity_factor(crack_length/2, load_amplitude, omega_val, 0)

# Calculate stress components using Williams' expansion
sigma_r = K_I / np.sqrt(2*np.pi*r) * np.cos(theta/2) * (1 - np.sin(theta/2) * np.sin(3*theta/2))
sigma_theta = K_I / np.sqrt(2*np.pi*r) * np.cos(theta/2) * (1 + np.sin(theta/2) * np.sin(3*theta/2))
tau_r_theta = K_I / np.sqrt(2*np.pi*r) * np.sin(theta/2) * np.cos(theta/2) * np.cos(3*theta/2)

# Convert to MPa for better readability
sigma_r_MPa = sigma_r / 1e6
sigma_theta_MPa = sigma_theta / 1e6
tau_r_theta_MPa = tau_r_theta / 1e6

#%% Create polar plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
ax.plot(theta, sigma_r_MPa, label='σ_r')
ax.plot(theta, sigma_theta_MPa, label='σ_θ')
ax.plot(theta, tau_r_theta_MPa, label='τ_rθ')
ax.set_title(f'Stress Components Around Crack Tip at r = {r*1000:.1f} mm')
ax.set_theta_zero_location('E')
ax.set_theta_direction(-1)
ax.set_rlabel_position(45)
ax.legend(loc='upper right')
ax.grid(True)
plt.show()

#%% Demonstrate frequency effects on stress intensity factor
def frequency_response_analysis(crack_length, load_amplitude, freq_range, n_points=50):
    """Analyze how frequency affects the maximum stress intensity factor"""
    frequencies = np.linspace(freq_range[0], freq_range[1], n_points)
    max_K_values = []

    for freq in frequencies:
        omega = 2 * np.pi * freq
        # Find maximum K_I over one cycle
        times = np.linspace(0, 1/freq, 20)
        k_values = [dynamic_stress_intensity_factor(crack_length/2, load_amplitude, omega, t)
                   for t in times]
        max_K_values.append(max(k_values))

    return frequencies, np.array(max_K_values)

#%% Plot frequency response
freq_range = (10, 1000)  # Hz
frequencies, max_K_values = frequency_response_analysis(
    crack_length=crack_length,
    load_amplitude=load_amplitude,
    freq_range=freq_range
)

plt.figure(figsize=(10, 6))
plt.plot(frequencies, max_K_values, 'g-', linewidth=2)
plt.title('Maximum Stress Intensity Factor vs Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Max K_I (Pa·m^(1/2))')
plt.grid(True)
plt.show()

#%% Visualize the real and imaginary parts of complex potentials
def visualize_complex_potentials(crack_length, load_amplitude, omega, t):
    """Visualize Muskhelishvili's complex potentials around a crack"""
    a = crack_length / 2

    # Create a grid excluding points on the crack
    x = np.linspace(-3*a, 3*a, 200)
    y = np.linspace(-2*a, 2*a, 150)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    # Initialize arrays for potentials
    phi_real = np.zeros_like(X)
    phi_imag = np.zeros_like(X)

    for i in range(len(y)):
        for j in range(len(x)):
            z_val = Z[i, j]
            # Skip points on the crack to avoid singularity
            if abs(Y[i, j]) < 1e-10 and abs(X[i, j]) <= a:
                phi_real[i, j] = np.nan
                phi_imag[i, j] = np.nan
            else:
                # Calculate phi potential (simplified analytical form)
                phi = load_amplitude * np.exp(1j * omega * t) * z_val / np.sqrt(z_val**2 - a**2)
                phi_real[i, j] = np.real(phi)
                phi_imag[i, j] = np.imag(phi)

    return X, Y, phi_real, phi_imag

#%%  Plot the real part of phi potential
X, Y, phi_real, phi_imag = visualize_complex_potentials(
    crack_length=crack_length,
    load_amplitude=load_amplitude,
    omega=omega_val,
    t=0
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot real part
levels1 = np.linspace(-2e6, 2e6, 20)
c1 = ax1.contourf(X, Y, phi_real, levels=levels1, cmap='RdBu')
ax1.set_title('Real Part of Complex Potential φ(z)')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_aspect('equal')
ax1.grid(True)
ax1.plot([-crack_length/2, crack_length/2], [0, 0], 'k-', linewidth=3)  # Draw the crack
fig.colorbar(c1, ax=ax1)

# Plot imaginary part
levels2 = np.linspace(-2e6, 2e6, 20)
c2 = ax2.contourf(X, Y, phi_imag, levels=levels2, cmap='RdBu')
ax2.set_title('Imaginary Part of Complex Potential φ(z)')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_aspect('equal')
ax2.grid(True)
ax2.plot([-crack_length/2, crack_length/2], [0, 0], 'k-', linewidth=3)  # Draw the crack
fig.colorbar(c2, ax=ax2)

plt.tight_layout()
plt.show()