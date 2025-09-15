# %% Complex Potentials Visualization
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def visualize_complex_potentials(geometry_type, params, material_props, omega):
	"""
	Visualize complex potentials for different geometries

	Parameters:
	- geometry_type: 'crack', 'hole', 'v_notch'
	- params: Dictionary of geometry parameters
	- material_props: Dictionary of material properties
	- omega: Angular frequency
	"""
	# Set up grid for visualization
	x = np.linspace(-5, 5, 200)
	y = np.linspace(-5, 5, 200)
	X, Y = np.meshgrid(x, y)
	Z = X + 1j * Y

	# Initialize potential arrays
	phi_real = np.zeros_like(X)
	phi_imag = np.zeros_like(X)
	psi_real = np.zeros_like(X)
	psi_imag = np.zeros_like(X)

	# Define mapping function based on geometry
	if geometry_type == 'crack':
		a = params.get('length', 2.0) / 2
		# Create mask for points inside the crack
		mask = (np.abs(Y) < 1e-10) & (np.abs(X) < a)

		# Define mapping function: z → z / √(z² - a²)
		def mapping_func(z_val):
			return z_val / np.sqrt(z_val ** 2 - a ** 2)

		# Define complex potentials for crack
		def phi_func(z_val):
			sigma = material_props.get('sigma_infinity', 1.0)
			return sigma * np.sqrt(z_val ** 2 - a ** 2)

		def psi_func(z_val):
			sigma = material_props.get('sigma_infinity', 1.0)
			return -sigma * z_val ** 2 / np.sqrt(z_val ** 2 - a ** 2)

	elif geometry_type == 'hole':
		a = params.get('radius', 1.0)
		# Create mask for points inside the hole
		mask = np.sqrt(X ** 2 + Y ** 2) < a

		# Define mapping function: z → a²/z
		def mapping_func(z_val):
			return a ** 2 / z_val

		# Define complex potentials for hole
		def phi_func(z_val):
			sigma = material_props.get('sigma_infinity', 1.0)
			return sigma / 2 * z_val + sigma * a ** 2 / (2 * z_val)

		def psi_func(z_val):
			sigma = material_props.get('sigma_infinity', 1.0)
			return -sigma / 2 * z_val - sigma * a ** 2 / z_val - sigma * a ** 4 / (2 * z_val ** 3)

	elif geometry_type == 'v_notch':
		alpha = params.get('angle', np.pi / 2)  # V-notch angle in radians
		# Create mask for points inside the V-notch
		theta = np.arctan2(Y, X)
		mask = (np.abs(theta) < alpha / 2) & (np.sqrt(X ** 2 + Y ** 2) < 1e-10)

		# Exponent for the mapping
		gamma = np.pi / alpha

		# Define mapping function: z → z^γ
		def mapping_func(z_val):
			return z_val ** gamma

		# Define complex potentials for V-notch
		def phi_func(z_val):
			# Use Williams expansion coefficients for the leading term
			lambda_value = williams_expansion_coefficients(alpha, 'mode1')[0]
			return z_val ** lambda_value

		def psi_func(z_val):
			lambda_value = williams_expansion_coefficients(alpha, 'mode1')[0]
			return -lambda_value * z_val ** lambda_value
	else:
		raise ValueError(f"Unsupported geometry type: {geometry_type}")

	# Calculate potentials at each point
	valid_points = ~mask
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if valid_points[i, j]:
				z_val = Z[i, j]
				try:
					phi = phi_func(z_val)
					psi = psi_func(z_val)

					phi_real[i, j] = np.real(phi)
					phi_imag[i, j] = np.imag(phi)
					psi_real[i, j] = np.real(psi)
					psi_imag[i, j] = np.imag(psi)
				except:
					# Handle numerical issues
					valid_points[i, j] = False

	# Mask invalid points
	phi_real[~valid_points] = np.nan
	phi_imag[~valid_points] = np.nan
	psi_real[~valid_points] = np.nan
	psi_imag[~valid_points] = np.nan

	# Create plots
	fig, axes = plt.subplots(2, 2, figsize=(16, 12))

	# Plot real part of phi
	im1 = axes[0, 0].contourf(X, Y, phi_real, levels=20, cmap='RdBu')
	axes[0, 0].set_title(r'Real Part of $\phi(z)$')
	axes[0, 0].set_aspect('equal')
	fig.colorbar(im1, ax=axes[0, 0])

	# Plot imaginary part of phi
	im2 = axes[0, 1].contourf(X, Y, phi_imag, levels=20, cmap='RdBu')
	axes[0, 1].set_title(r'Imaginary Part of $\phi(z)$')
	axes[0, 1].set_aspect('equal')
	fig.colorbar(im2, ax=axes[0, 1])

	# Plot real part of psi
	im3 = axes[1, 0].contourf(X, Y, psi_real, levels=20, cmap='RdBu')
	axes[1, 0].set_title(r'Real Part of $\psi(z)$')
	axes[1, 0].set_aspect('equal')
	fig.colorbar(im3, ax=axes[1, 0])

	# Plot imaginary part of psi
	im4 = axes[1, 1].contourf(X, Y, psi_imag, levels=20, cmap='RdBu')
	axes[1, 1].set_title(r'Imaginary Part of $\psi(z)$')
	axes[1, 1].set_aspect('equal')
	fig.colorbar(im4, ax=axes[1, 1])

	# Draw geometry
	if geometry_type == 'crack':
		for ax in axes.flatten():
			ax.plot([-a, a], [0, 0], 'k-', linewidth=3)
	elif geometry_type == 'hole':
		for ax in axes.flatten():
			circle = plt.Circle((0, 0), a, fill=False, color='k', linewidth=2)
			ax.add_patch(circle)
	elif geometry_type == 'v_notch':
		for ax in axes.flatten():
			r_max = 3
			ax.plot([0, r_max * np.cos(-alpha / 2)], [0, r_max * np.sin(-alpha / 2)], 'k-', linewidth=2)
			ax.plot([0, r_max * np.cos(alpha / 2)], [0, r_max * np.sin(alpha / 2)], 'k-', linewidth=2)

	plt.tight_layout()
	plt.show()


# %% Williams Expansion Coefficients Implementation
def williams_expansion_coefficients(alpha, mode='mode1', n_terms=5):
	"""
	Calculate Williams' expansion coefficients for V-notch problems

	Parameters:
	- alpha: V-notch angle in radians
	- mode: 'mode1' (symmetric) or 'mode2' (antisymmetric)
	- n_terms: Number of terms to return

	Returns:
	- Array of eigenvalues
	"""

	# Define the characteristic equation based on mode
	def char_eq(lambda_val, alpha, mode):
		if mode == 'mode1':
			# Mode I (symmetric) characteristic equation
			return np.sin(lambda_val * alpha) + lambda_val * np.sin(alpha)
		else:
			# Mode II (antisymmetric) characteristic equation
			return np.sin(lambda_val * alpha) - lambda_val * np.sin(alpha)

	# Find roots (eigenvalues) of the characteristic equation
	lambda_values = []

	# Initial guess for eigenvalues
	if mode == 'mode1':
		# Mode I eigenvalues often start near these values
		initial_guesses = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
	else:
		# Mode II eigenvalues often start near these values
		initial_guesses = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

	# Use numerical root finding
	from scipy.optimize import fsolve

	for guess in initial_guesses:
		if len(lambda_values) >= n_terms:
			break

		# Find root near the guess
		root = fsolve(lambda x: char_eq(x, alpha, mode), guess)[0]

		# Check if root is valid and not a duplicate
		if root > 0 and not np.isclose(root, 0) and not any(np.isclose(root, val) for val in lambda_values):
			lambda_values.append(root)

	# Sort eigenvalues
	lambda_values.sort()

	# For 90° V-notch in Mode I, the first eigenvalue should be around 0.5
	# This is a validation check
	if mode == 'mode1' and np.isclose(alpha, np.pi / 2, rtol=1e-2) and not np.isclose(lambda_values[0], 0.5, rtol=1e-1):
		print("Warning: First eigenvalue for 90° V-notch should be approximately 0.5")

	return np.array(lambda_values[:n_terms])


# %% Stress Field Visualization
def visualize_stress_field(geometry_type, params, material_props, omega=0):
	"""
	Visualize stress fields around different geometries

	Parameters:
	- geometry_type: 'crack', 'hole', 'v_notch'
	- params: Dictionary of geometry parameters
	- material_props: Dictionary of material properties
	- omega: Angular frequency (default=0 for static case)
	"""
	# Set up grid for visualization
	x = np.linspace(-5, 5, 200)
	y = np.linspace(-5, 5, 200)
	X, Y = np.meshgrid(x, y)
	Z = X + 1j * Y

	# Initialize stress arrays
	sigma_xx = np.zeros_like(X)
	sigma_yy = np.zeros_like(X)
	tau_xy = np.zeros_like(X)

	# Create mask based on geometry
	if geometry_type == 'crack':
		a = params.get('length', 2.0) / 2
		mask = (np.abs(Y) < 1e-10) & (np.abs(X) < a)
	elif geometry_type == 'hole':
		a = params.get('radius', 1.0)
		mask = np.sqrt(X ** 2 + Y ** 2) < a
	elif geometry_type == 'v_notch':
		alpha = params.get('angle', np.pi / 2)
		theta = np.arctan2(Y, X)
		mask = (np.abs(theta) < alpha / 2) & (np.sqrt(X ** 2 + Y ** 2) < 1e-10)
	else:
		raise ValueError(f"Unsupported geometry type: {geometry_type}")

	# Calculate stresses at each point
	valid_points = ~mask
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if valid_points[i, j]:
				z_val = Z[i, j]
				try:
					# Calculate stresses based on geometry
					if geometry_type == 'crack':
						stresses = calculate_crack_stresses(z_val, a, material_props)
					elif geometry_type == 'hole':
						stresses = calculate_hole_stresses(z_val, a, material_props)
					elif geometry_type == 'v_notch':
						stresses = calculate_vnotch_stresses(z_val, alpha, material_props)

					sigma_xx[i, j] = stresses[0]
					sigma_yy[i, j] = stresses[1]
					tau_xy[i, j] = stresses[2]
				except:
					# Handle numerical issues
					valid_points[i, j] = False

	# Mask invalid points
	sigma_xx[~valid_points] = np.nan
	sigma_yy[~valid_points] = np.nan
	tau_xy[~valid_points] = np.nan

	# Calculate von Mises stress
	von_mises = np.sqrt(sigma_xx ** 2 + sigma_yy ** 2 - sigma_xx * sigma_yy + 3 * tau_xy ** 2)

	# Create plots
	fig, axes = plt.subplots(2, 2, figsize=(16, 12))

	# Plot sigma_xx
	im1 = axes[0, 0].contourf(X, Y, sigma_xx, levels=20, cmap='jet')
	axes[0, 0].set_title(r'$\sigma_{xx}$')
	axes[0, 0].set_aspect('equal')
	fig.colorbar(im1, ax=axes[0, 0])

	# Plot sigma_yy
	im2 = axes[0, 1].contourf(X, Y, sigma_yy, levels=20, cmap='jet')
	axes[0, 1].set_title(r'$\sigma_{yy}$')
	axes[0, 1].set_aspect('equal')
	fig.colorbar(im2, ax=axes[0, 1])

	# Plot tau_xy
	im3 = axes[1, 0].contourf(X, Y, tau_xy, levels=20, cmap='jet')
	axes[1, 0].set_title(r'$\tau_{xy}$')
	axes[1, 0].set_aspect('equal')
	fig.colorbar(im3, ax=axes[1, 0])

	# Plot von Mises stress
	im4 = axes[1, 1].contourf(X, Y, von_mises, levels=20, cmap='jet')
	axes[1, 1].set_title('von Mises Stress')
	axes[1, 1].set_aspect('equal')
	fig.colorbar(im4, ax=axes[1, 1])

	# Draw geometry
	if geometry_type == 'crack':
		for ax in axes.flatten():
			ax.plot([-a, a], [0, 0], 'k-', linewidth=3)
	elif geometry_type == 'hole':
		for ax in axes.flatten():
			circle = plt.Circle((0, 0), a, fill=False, color='k', linewidth=2)
			ax.add_patch(circle)
	elif geometry_type == 'v_notch':
		for ax in axes.flatten():
			r_max = 3
			ax.plot([0, r_max * np.cos(-alpha / 2)], [0, r_max * np.sin(-alpha / 2)], 'k-', linewidth=2)
			ax.plot([0, r_max * np.cos(alpha / 2)], [0, r_max * np.sin(alpha / 2)], 'k-', linewidth=2)

	plt.tight_layout()
	plt.show()


# %% Helper functions for stress calculations
def calculate_crack_stresses(z, a, material_props):
	"""Calculate stresses around a crack"""
	sigma_inf = material_props.get('sigma_infinity', 1.0)

	# Convert to polar coordinates
	r = np.abs(z)
	theta = np.angle(z)

	# For points very close to crack tip, use asymptotic solution
	if np.abs(z - a) < 1e-10 or np.abs(z + a) < 1e-10:
		K_I = sigma_inf * np.sqrt(np.pi * a)  # Stress intensity factor

		# Asymptotic stress field
		sigma_xx = K_I / np.sqrt(2 * np.pi * r) * np.cos(theta / 2) * (1 - np.sin(theta / 2) * np.sin(3 * theta / 2))
		sigma_yy = K_I / np.sqrt(2 * np.pi * r) * np.cos(theta / 2) * (1 + np.sin(theta / 2) * np.sin(3 * theta / 2))
		tau_xy = K_I / np.sqrt(2 * np.pi * r) * np.cos(theta / 2) * np.sin(theta / 2) * np.cos(3 * theta / 2)
	else:
		# Use complex potentials
		z_term = np.sqrt(z ** 2 - a ** 2)
		phi = sigma_inf * z_term
		phi_prime = sigma_inf * z / z_term
		psi_prime = -sigma_inf * z ** 2 / (z_term * (z ** 2 - a ** 2))

		# Calculate stresses
		sigma_xx = np.real(phi_prime + phi_prime.conjugate() - z.conjugate() * psi_prime)
		sigma_yy = np.real(phi_prime + phi_prime.conjugate() + z.conjugate() * psi_prime)
		tau_xy = np.imag(z.conjugate() * psi_prime)

	return sigma_xx, sigma_yy, tau_xy


def calculate_hole_stresses(z, a, material_props):
	"""Calculate stresses around a circular hole"""
	sigma_inf = material_props.get('sigma_infinity', 1.0)

	# Convert to polar coordinates
	r = np.abs(z)
	theta = np.angle(z)

	# Complex potentials for circular hole
	phi = sigma_inf / 2 * z + sigma_inf * a ** 2 / (2 * z)
	psi = -sigma_inf / 2 * z - sigma_inf * a ** 2 / z - sigma_inf * a ** 4 / (2 * z ** 3)

	# Derivatives
	phi_prime = sigma_inf / 2 - sigma_inf * a ** 2 / (2 * z ** 2)
	psi_prime = -sigma_inf / 2 + sigma_inf * a ** 2 / z ** 2 + 3 * sigma_inf * a ** 4 / (2 * z ** 4)

	# Calculate stresses
	sigma_xx = np.real(phi_prime + phi_prime.conjugate() - z.conjugate() * psi_prime)
	sigma_yy = np.real(phi_prime + phi_prime.conjugate() + z.conjugate() * psi_prime)
	tau_xy = np.imag(z.conjugate() * psi_prime)

	return sigma_xx, sigma_yy, tau_xy


def calculate_vnotch_stresses(z, alpha, material_props):
	"""Calculate stresses around a V-notch"""
	sigma_inf = material_props.get('sigma_infinity', 1.0)

	# Get eigenvalue for V-notch
	lambda1 = williams_expansion_coefficients(alpha, 'mode1')[0]

	# Convert to polar coordinates
	r = np.abs(z)
	theta = np.angle(z)

	# Williams expansion for stresses (Mode I, first term only)
	K = sigma_inf  # Generalized stress intensity factor

	# Angular functions for Mode I
	f_theta_r = lambda1 * ((2 - lambda1) * np.cos((lambda1 - 1) * theta) -
	                       (lambda1 - 1) * np.cos((lambda1 - 3) * theta))
	f_theta_theta = lambda1 * ((2 + lambda1) * np.cos((lambda1 - 1) * theta) +
	                           (lambda1 - 1) * np.cos((lambda1 - 3) * theta))
	f_theta_r_theta = lambda1 * ((lambda1 - 1) * np.sin((lambda1 - 3) * theta) -
	                             (lambda1 - 1) * np.sin((lambda1 - 1) * theta))

	# Calculate stresses
	sigma_r = K * r ** (lambda1 - 1) * f_theta_r
	sigma_theta = K * r ** (lambda1 - 1) * f_theta_theta
	tau_r_theta = K * r ** (lambda1 - 1) * f_theta_r_theta

	# Convert to Cartesian
	sigma_xx = sigma_r * np.cos(theta) ** 2 + sigma_theta * np.sin(theta) ** 2 - 2 * tau_r_theta * np.sin(
		theta) * np.cos(theta)
	sigma_yy = sigma_r * np.sin(theta) ** 2 + sigma_theta * np.cos(theta) ** 2 + 2 * tau_r_theta * np.sin(
		theta) * np.cos(theta)
	tau_xy = (sigma_r - sigma_theta) * np.sin(theta) * np.cos(theta) + tau_r_theta * (
				np.cos(theta) ** 2 - np.sin(theta) ** 2)

	return sigma_xx, sigma_yy, tau_xy


# %% Vibration Modes Visualization
def visualize_vibration_modes(geometry_type, params, material_props, n_modes=4):
	"""
	Visualize vibration modes for different geometries

	Parameters:
	- geometry_type: 'crack', 'hole', 'v_notch'
	- params: Dictionary of geometry parameters
	- material_props: Dictionary of material properties
	- n_modes: Number of modes to display
	"""
	# Set up grid for visualization
	n_elements = 30
	x = np.linspace(-5, 5, n_elements)
	y = np.linspace(-5, 5, n_elements)
	X, Y = np.meshgrid(x, y)

	# Material properties
	E = material_props.get('E', 200e9)  # Young's modulus
	rho = material_props.get('rho', 7800)  # Density
	nu = material_props.get('nu', 0.3)  # Poisson's ratio

	# Create mask based on geometry
	if geometry_type == 'crack':
		a = params.get('length', 2.0) / 2
		mask = (np.abs(Y) < 1e-10) & (np.abs(X) < a)
	elif geometry_type == 'hole':
		a = params.get('radius', 1.0)
		mask = np.sqrt(X ** 2 + Y ** 2) < a
	elif geometry_type == 'v_notch':
		alpha = params.get('angle', np.pi / 2)
		theta = np.arctan2(Y, X)
		r = np.sqrt(X ** 2 + Y ** 2)
		mask = (np.abs(theta) < alpha / 2) & (r < 1e-10)
	else:
		raise ValueError(f"Unsupported geometry type: {geometry_type}")

	# Calculate natural frequencies and mode shapes
	frequencies, mode_shapes = calculate_vibration_modes(geometry_type, params, material_props, n_elements, n_modes)

	# Create plots
	fig, axes = plt.subplots(2, 2, figsize=(14, 12))
	axes = axes.flatten()

	# Plot each mode
	for i in range(min(n_modes, 4)):
		# Reshape mode shape to 2D grid
		mode = mode_shapes[:, i].reshape(n_elements, n_elements)

		# Apply mask
		mode_masked = mode.copy()
		mode_masked[mask] = np.nan

		# Plot mode shape
		contour = axes[i].contourf(X, Y, mode_masked, levels=20, cmap='viridis')
		axes[i].set_title(f'Mode {i + 1}: {frequencies[i]:.2f} Hz')
		axes[i].set_aspect('equal')
		fig.colorbar(contour, ax=axes[i], label='Displacement')

		# Draw geometry
		if geometry_type == 'crack':
			axes[i].plot([-a, a], [0, 0], 'k-', linewidth=3)
		elif geometry_type == 'hole':
			circle = plt.Circle((0, 0), a, fill=False, color='k', linewidth=2)
			axes[i].add_patch(circle)
		elif geometry_type == 'v_notch':
			r_max = 3
			axes[i].plot([0, r_max * np.cos(-alpha / 2)], [0, r_max * np.sin(-alpha / 2)], 'k-', linewidth=2)
			axes[i].plot([0, r_max * np.cos(alpha / 2)], [0, r_max * np.sin(alpha / 2)], 'k-', linewidth=2)

	plt.tight_layout()
	plt.show()


# %%  Frequency Response Analysis
def frequency_response_analysis(geometry_type, params, material_props, freq_range=(1, 1000), n_points=100):
	"""
	Perform frequency response analysis for different geometries

	Parameters:
	- geometry_type: 'crack', 'hole', 'v_notch'
	- params: Dictionary of geometry parameters
	- material_props: Dictionary of material properties
	- freq_range: Tuple of (min, max) frequency in Hz
	- n_points: Number of frequency points to analyze
	"""
	# Material properties
	E = material_props.get('E', 200e9)  # Young's modulus
	rho = material_props.get('rho', 7800)  # Density
	nu = material_props.get('nu', 0.3)  # Poisson's ratio

	# Create frequency array
	frequencies = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)

	# Calculate characteristic dimension based on geometry
	if geometry_type == 'crack':
		a = params.get('length', 2.0) / 2
		char_dim = a
	elif geometry_type == 'hole':
		a = params.get('radius', 1.0)
		char_dim = a
	elif geometry_type == 'v_notch':
		alpha = params.get('angle', np.pi / 2)
		char_dim = 1.0  # Arbitrary reference dimension
	else:
		raise ValueError(f"Unsupported geometry type: {geometry_type}")

	# Calculate wave speeds
	c_p = np.sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2 * nu)))  # P-wave speed
	c_s = np.sqrt(E / (2 * rho * (1 + nu)))  # S-wave speed

	# Calculate wavelengths and dimensionless frequency
	wavelengths_p = c_p / frequencies
	ka_values = 2 * np.pi * char_dim / wavelengths_p

	# Calculate response metrics
	scf_values = np.zeros_like(frequencies)  # Stress concentration factor
	displacement_amp = np.zeros_like(frequencies)  # Displacement amplitude

	# Calculate frequency response based on geometry
	for i, freq in enumerate(frequencies):
		omega = 2 * np.pi * freq
		ka = ka_values[i]

		if geometry_type == 'crack':
			# SCF for a crack (simplified model)
			scf_values[i] = crack_dynamic_scf(ka)
			displacement_amp[i] = 1.0 / (1 + 0.1 * ka ** 2)  # Simplified model

		elif geometry_type == 'hole':
			# SCF for a circular hole (simplified model)
			scf_values[i] = hole_dynamic_scf(ka)
			displacement_amp[i] = 1.0 / (1 + 0.2 * ka ** 2)  # Simplified model

		elif geometry_type == 'v_notch':
			# SCF for a V-notch (simplified model)
			alpha = params.get('angle', np.pi / 2)
			lambda1 = williams_expansion_coefficients(alpha, 'mode1')[0]
			scf_values[i] = vnotch_dynamic_scf(ka, lambda1)
			displacement_amp[i] = 1.0 / (1 + 0.15 * ka ** 2)  # Simplified model

	# Plot results
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

	# Plot stress concentration factor
	ax1.semilogx(frequencies, scf_values, 'b-', linewidth=2)
	ax1.set_title('Dynamic Stress Concentration Factor')
	ax1.set_xlabel('Frequency (Hz)')
	ax1.set_ylabel('SCF')
	ax1.grid(True, which='both', ls='--')

	# Add reference line for static SCF
	if geometry_type == 'crack':
		static_scf = crack_dynamic_scf(0)
	elif geometry_type == 'hole':
		static_scf = hole_dynamic_scf(0)
	elif geometry_type == 'v_notch':
		alpha = params.get('angle', np.pi / 2)
		lambda1 = williams_expansion_coefficients(alpha, 'mode1')[0]
		static_scf = vnotch_dynamic_scf(0, lambda1)

	ax1.axhline(y=static_scf, color='r', linestyle='--', label=f'Static SCF = {static_scf:.2f}')
	ax1.legend()

	# Add top x-axis with ka values
	ax1_top = ax1.twiny()
	ax1_top.set_xscale('log')
	ax1_top.set_xlim(ax1.get_xlim())
	ax1_top.set_xlabel('Dimensionless Frequency (ka)')

	# Set specific ka ticks
	ka_ticks = [0.1, 1, 10]
	freq_ticks = [ka_tick * c_p / (2 * np.pi * char_dim) for ka_tick in ka_ticks]
	ax1_top.set_xticks(freq_ticks)
	ax1_top.set_xticklabels([f'{ka_tick:.1f}' for ka_tick in ka_ticks])

	# Plot displacement amplitude
	ax2.semilogx(frequencies, displacement_amp, 'g-', linewidth=2)
	ax2.set_title('Displacement Amplitude')
	ax2.set_xlabel('Frequency (Hz)')
	ax2.set_ylabel('Normalized Amplitude')
	ax2.grid(True, which='both', ls='--')

	# Add top x-axis with ka values
	ax2_top = ax2.twiny()
	ax2_top.set_xscale('log')
	ax2_top.set_xlim(ax2.get_xlim())
	ax2_top.set_xlabel('Dimensionless Frequency (ka)')

	# Set specific ka ticks
	ax2_top.set_xticks(freq_ticks)
	ax2_top.set_xticklabels([f'{ka_tick:.1f}' for ka_tick in ka_ticks])

	plt.tight_layout()
	plt.show()


# %%
# Helper functions for dynamic SCF calculations
def crack_dynamic_scf(ka):
	"""Calculate dynamic SCF for a crack"""
	# Simplified model based on dynamic fracture mechanics
	if ka < 0.01:  # Static case
		return 2.0  # Static SCF for crack tip (simplified)
	else:
		# SCF increases with frequency until wave interaction becomes significant
		return 2.0 * (1 + 0.3 * ka * np.exp(-0.2 * ka))


def hole_dynamic_scf(ka):
	"""Calculate dynamic SCF for a circular hole"""
	# Simplified model based on dynamic elasticity
	if ka < 0.01:  # Static case
		return 3.0  # Static SCF for circular hole
	else:
		# SCF varies with frequency due to wave scattering
		return 3.0 * (1 + 0.2 * np.sin(ka) * np.exp(-0.1 * ka))


def vnotch_dynamic_scf(ka, lambda1):
	"""Calculate dynamic SCF for a V-notch"""
	# Simplified model based on singularity strength
	# lambda1 is the first eigenvalue from Williams' expansion

	# Static SCF depends on singularity strength
	static_scf = 1.0 / (lambda1 - 0.5) if lambda1 > 0.5 else 4.0

	if ka < 0.01:  # Static case
		return static_scf
	else:
		# SCF varies with frequency
		return static_scf * (1 + 0.25 * ka * np.exp(-0.15 * ka))


# %% 6. Comparative Analysis Across Geometries
def compare_geometries():
	"""Compare stress fields and frequency responses across different geometries"""
	# Define material properties
	material_props = {
		'E': 200e9,  # Young's modulus (Pa)
		'nu': 0.3,  # Poisson's ratio
		'rho': 7800,  # Density (kg/m³)
		'sigma_infinity': 1.0  # Remote stress
	}

	# Define geometry parameters
	crack_params = {'length': 2.0}  # Crack length
	hole_params = {'radius': 1.0}  # Hole radius
	vnotch_params = {'angle': np.pi / 2}  # 90° V-notch

	# 1. Compare stress fields
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))

	# Setup visualization grid
	x = np.linspace(-3, 3, 200)
	y = np.linspace(-3, 3, 200)
	X, Y = np.meshgrid(x, y)

	# Crack stress field
	Z = X + 1j * Y
	a = crack_params['length'] / 2

	sigma_yy_crack = np.zeros_like(X)
	mask_crack = (np.abs(Y) < 1e-10) & (np.abs(X) < a)

	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if not mask_crack[i, j]:
				z_val = Z[i, j]
				try:
					_, sigma_yy, _ = calculate_crack_stresses(z_val, a, material_props)
					sigma_yy_crack[i, j] = sigma_yy
				except:
					sigma_yy_crack[i, j] = np.nan

	# Circular hole stress field
	a_hole = hole_params['radius']
	sigma_yy_hole = np.zeros_like(X)
	mask_hole = np.sqrt(X ** 2 + Y ** 2) < a_hole

	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if not mask_hole[i, j]:
				z_val = Z[i, j]
				try:
					_, sigma_yy, _ = calculate_hole_stresses(z_val, a_hole, material_props)
					sigma_yy_hole[i, j] = sigma_yy
				except:
					sigma_yy_hole[i, j] = np.nan

	# V-notch stress field
	alpha = vnotch_params['angle']
	sigma_yy_vnotch = np.zeros_like(X)
	theta = np.arctan2(Y, X)
	r = np.sqrt(X ** 2 + Y ** 2)
	mask_vnotch = (np.abs(theta) < alpha / 2) & (r < 1e-10)

	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if not mask_vnotch[i, j]:
				z_val = Z[i, j]
				try:
					_, sigma_yy, _ = calculate_vnotch_stresses(z_val, alpha, material_props)
					sigma_yy_vnotch[i, j] = sigma_yy
				except:
					sigma_yy_vnotch[i, j] = np.nan

	# Plot stress fields
	contour1 = axes[0].contourf(X, Y, sigma_yy_crack, levels=20, cmap='jet', extend='both')
	axes[0].set_title('Crack')
	axes[0].set_aspect('equal')
	fig.colorbar(contour1, ax=axes[0], label=r'$\sigma_{yy}$')
	axes[0].plot([-a, a], [0, 0], 'k-', linewidth=3)

	contour2 = axes[1].contourf(X, Y, sigma_yy_hole, levels=20, cmap='jet', extend='both')
	axes[1].set_title('Circular Hole')
	axes[1].set_aspect('equal')
	fig.colorbar(contour2, ax=axes[1], label=r'$\sigma_{yy}$')
	circle = plt.Circle((0, 0), a_hole, fill=False, color='k', linewidth=2)
	axes[1].add_patch(circle)

	contour3 = axes[2].contourf(X, Y, sigma_yy_vnotch, levels=20, cmap='jet', extend='both')
	axes[2].set_title('V-notch')
	axes[2].set_aspect('equal')
	fig.colorbar(contour3, ax=axes[2], label=r'$\sigma_{yy}$')
	r_max = 3
	axes[2].plot([0, r_max * np.cos(-alpha / 2)], [0, r_max * np.sin(-alpha / 2)], 'k-', linewidth=2)
	axes[2].plot([0, r_max * np.cos(alpha / 2)], [0, r_max * np.sin(alpha / 2)], 'k-', linewidth=2)

	plt.tight_layout()
	plt.show()

	# 2. Compare frequency responses
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

	# Frequency range
	frequencies = np.logspace(1, 4, 100)  # 10 Hz to 10 kHz

	# Material properties for wave speed
	E = material_props['E']
	rho = material_props['rho']
	nu = material_props['nu']

	# Calculate wave speeds
	c_p = np.sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2 * nu)))  # P-wave speed

	# Calculate dimensionless frequencies
	ka_crack = 2 * np.pi * a * frequencies / c_p
	ka_hole = 2 * np.pi * a_hole * frequencies / c_p

	# Reference dimension for V-notch
	ref_dim = 1.0
	ka_vnotch = 2 * np.pi * ref_dim * frequencies / c_p

	# Calculate dynamic SCFs
	scf_crack = [crack_dynamic_scf(ka) for ka in ka_crack]
	scf_hole = [hole_dynamic_scf(ka) for ka in ka_hole]

	# Get eigenvalue for V-notch
	lambda1 = williams_expansion_coefficients(alpha, 'mode1')[0]
	scf_vnotch = [vnotch_dynamic_scf(ka, lambda1) for ka in ka_vnotch]

	# Plot SCFs
	ax1.semilogx(frequencies, scf_crack, 'b-', linewidth=2, label='Crack')
	ax1.semilogx(frequencies, scf_hole, 'r-', linewidth=2, label='Circular Hole')
	ax1.semilogx(frequencies, scf_vnotch, 'g-', linewidth=2, label='V-notch')

	ax1.set_title('Dynamic Stress Concentration Factors')
	ax1.set_xlabel('Frequency (Hz)')
	ax1.set_ylabel('SCF')
	ax1.grid(True, which='both', ls='--')
	ax1.legend()

	# Calculate singularity orders
	r_values = np.logspace(-3, 0, 100)

	# Crack: r^(-0.5) singularity
	stress_crack = r_values ** (-0.5)

	# Hole: no singularity
	stress_hole = np.ones_like(r_values) * 3

	# V-notch: r^(λ-1) singularity
	stress_vnotch = r_values ** (lambda1 - 1)

	# Plot stress singularities
	ax2.loglog(r_values, stress_crack, 'b-', linewidth=2,
	           label=f'Crack: $r^{{-0.5}}$')
	ax2.loglog(r_values, stress_hole, 'r-', linewidth=2,
	           label=f'Hole: No singularity (SCF=3)')
	ax2.loglog(r_values, stress_vnotch, 'g-', linewidth=2,
	           label=f'V-notch: $r^{{{lambda1 - 1:.3f}}}$')

	ax2.set_title('Stress Singularity Comparison')
	ax2.set_xlabel('Normalized Distance from Feature (r/a)')
	ax2.set_ylabel('Normalized Stress')
	ax2.grid(True, which='both', ls='--')
	ax2.legend()

	plt.tight_layout()
	plt.show()


# %%
def calculate_vibration_modes(geometry_type, params, material_props, n_elements, n_modes):
	"""
	Simplified calculation of vibration modes using finite differences

	Note: This is a simplified implementation for visualization purposes.
	In a real application, you would use a proper FEM solver.
	"""
	# Material properties
	E = material_props.get('E', 200e9)  # Young's modulus
	rho = material_props.get('rho', 7800)  # Density
	nu = material_props.get('nu', 0.3)  # Poisson's ratio

	# Create grid
	x = np.linspace(-5, 5, n_elements)
	y = np.linspace(-5, 5, n_elements)
	X, Y = np.meshgrid(x, y)

	# Create mask based on geometry
	if geometry_type == 'crack':
		a = params.get('length', 2.0) / 2
		mask = (np.abs(Y) < 1e-10) & (np.abs(X) < a)
	elif geometry_type == 'hole':
		a = params.get('radius', 1.0)
		mask = np.sqrt(X ** 2 + Y ** 2) < a
	elif geometry_type == 'v_notch':
		alpha = params.get('angle', np.pi / 2)
		theta = np.arctan2(Y, X)
		r = np.sqrt(X ** 2 + Y ** 2)
		mask = (np.abs(theta) < alpha / 2) & (r < 1)
	else:
		raise ValueError(f"Unsupported geometry type: {geometry_type}")

	# Set up stiffness and mass matrices
	n_dof = n_elements * n_elements  # Number of degrees of freedom
	K = np.zeros((n_dof, n_dof))  # Stiffness matrix
	M = np.zeros((n_dof, n_dof))  # Mass matrix

	# Grid spacing
	dx = x[1] - x[0]
	dy = y[1] - y[0]

	# Fill matrices (simplified approach)
	for i in range(n_elements):
		for j in range(n_elements):
			# Node index in flattened array
			node = i * n_elements + j

			# Skip nodes in the masked area
			if mask[i, j]:
				continue

			# Add mass
			M[node, node] = rho * dx * dy

			# Add stiffness (simplified Laplacian stencil)
			K[node, node] = 4 * E / ((1 - nu ** 2) * dx ** 2)

			# Connect to neighbors
			neighbors = []
			if i > 0:
				neighbors.append((i - 1) * n_elements + j)
			if i < n_elements - 1:
				neighbors.append((i + 1) * n_elements + j)
			if j > 0:
				neighbors.append(i * n_elements + (j - 1))
			if j < n_elements - 1:
				neighbors.append(i * n_elements + (j + 1))

			for neighbor in neighbors:
				if neighbor < 0 or neighbor >= n_dof:
					continue

				# Skip connections to masked nodes
				ni, nj = neighbor // n_elements, neighbor % n_elements
				if mask[ni, nj]:
					continue

				K[node, neighbor] = -E / ((1 - nu ** 2) * dx ** 2)

	# Apply boundary conditions (simplified)
	# Fix outer edges
	for i in range(n_elements):
		for j in range(n_elements):
			node = i * n_elements + j

			# Check if node is on boundary
			if i == 0 or i == n_elements - 1 or j == 0 or j == n_elements - 1:
				K[node, :] = 0
				K[:, node] = 0
				K[node, node] = 1
				M[node, :] = 0
				M[:, node] = 0
				M[node, node] = 1e-10  # Small non-zero value for numerical stability

	# Solve generalized eigenvalue problem
	from scipy.sparse.linalg import eigsh

	# Convert to sparse matrices for efficiency
	from scipy.sparse import csr_matrix
	K_sparse = csr_matrix(K)
	M_sparse = csr_matrix(M)

	# Solve for eigenvalues and eigenvectors
	try:
		eigenvalues, eigenvectors = eigsh(K_sparse, k=n_modes + 5, M=M_sparse, sigma=1.0)

		# Calculate frequencies from eigenvalues
		frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

		# Sort by frequency
		idx = np.argsort(frequencies)
		frequencies = frequencies[idx]
		eigenvectors = eigenvectors[:, idx]

		# Return first n_modes
		return frequencies[:n_modes], eigenvectors[:, :n_modes]
	except:
		# Fallback: Generate example modes
		print("Warning: Eigenvalue calculation failed. Generating example modes.")

		# Create example frequencies
		if geometry_type == 'crack':
			frequencies = np.array([100, 250, 450, 700]) * np.sqrt(E / (rho * a ** 2))
		elif geometry_type == 'hole':
			frequencies = np.array([150, 300, 500, 800]) * np.sqrt(E / (rho * a ** 2))
		elif geometry_type == 'v_notch':
			frequencies = np.array([120, 280, 480, 750]) * np.sqrt(E / (rho * 5 ** 2))

		# Create example mode shapes
		mode_shapes = np.zeros((n_dof, n_modes))

		# Mode 1: First bending
		mode_shapes[:, 0] = np.sin(np.pi * X.flatten() / 10) * np.sin(np.pi * Y.flatten() / 10)

		# Mode 2: Second bending
		mode_shapes[:, 1] = np.sin(2 * np.pi * X.flatten() / 10) * np.sin(np.pi * Y.flatten() / 10)

		# Mode 3: Torsional
		mode_shapes[:, 2] = np.sin(np.pi * X.flatten() / 10) * np.sin(2 * np.pi * Y.flatten() / 10)

		# Mode 4: Higher order
		mode_shapes[:, 3] = np.sin(2 * np.pi * X.flatten() / 10) * np.sin(2 * np.pi * Y.flatten() / 10)

		return frequencies[:n_modes], mode_shapes[:, :n_modes]


# %%
import numpy as np
import sympy as sp
from scipy import integrate

# Define symbolic variables
z, zeta = sp.symbols('z zeta', complex=True)
t, omega = sp.symbols('t omega', real=True)


# Muskhelishvili complex potentials for general case
def muskhelishvili_potentials(z, mapping_func, phi_func, psi_func):
	"""
	Calculate Muskhelishvili complex potentials with conformal mapping

	Parameters:
	- z: Complex coordinate
	- mapping_func: Conformal mapping function
	- phi_func, psi_func: Functions defining the complex potentials

	Returns:
	- phi, psi: Complex potentials at point z
	"""
	# Apply conformal mapping
	zeta = mapping_func(z)

	# Calculate potentials in mapped domain
	phi_zeta = phi_func(zeta)
	psi_zeta = psi_func(zeta)

	# Transform back to original domain
	dz_dzeta = 1 / sp.diff(mapping_func(z), z).subs(z, z)

	phi = phi_zeta
	psi = psi_zeta + z * sp.conjugate(sp.diff(phi_zeta, zeta)) * dz_dzeta

	return phi, psi


# Calculate stresses from complex potentials
def calculate_stresses_from_potentials(z, phi, psi):
	"""
	Calculate stresses from Muskhelishvili complex potentials

	Parameters:
	- z: Complex coordinate
	- phi, psi: Complex potentials

	Returns:
	- sigma_x, sigma_y, tau_xy: Stress components
	"""
	# Derivatives
	phi_prime = sp.diff(phi, z)
	phi_bar_prime = sp.conjugate(phi_prime)

	# Stress components (Muskhelishvili's formulas)
	sigma_x = 2 * sp.re(phi_prime) - sp.re(z * sp.conjugate(phi_prime) + psi)
	sigma_y = 2 * sp.re(phi_prime) + sp.re(z * sp.conjugate(phi_prime) + psi)
	tau_xy = sp.im(z * sp.conjugate(phi_prime) + psi)

	return sigma_x, sigma_y, tau_xy


# %% Vibration analysis with complex variable method
def vibration_with_complex_variable(geometry_type, params, material_props, freq_range, n_points=100):
	"""
	Perform vibration analysis using Muskhelishvili's complex variable method

	Parameters:
	- geometry_type: 'crack', 'hole', 'v_notch', etc.
	- params: Dictionary of geometry parameters
	- material_props: Dictionary of material properties
	- freq_range: Tuple of (min_freq, max_freq)
	- n_points: Number of frequency points

	Returns:
	- frequencies: Array of frequencies
	- response: Array of maximum stress responses
	"""
	E = material_props['E']
	nu = material_props['nu']
	rho = material_props['rho']

	# Frequency range
	frequencies = np.linspace(freq_range[0], freq_range[1], n_points)

	# Response array
	response = np.zeros(n_points)

	# For each frequency
	for i, freq in enumerate(frequencies):
		omega_val = 2 * np.pi * freq

		# Calculate wavelength
		c_p = np.sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2 * nu)))  # P-wave speed
		wavelength = c_p / freq

		# Different analysis based on geometry type
		if geometry_type == 'crack':
			a = params['crack_length'] / 2
			load_amplitude = params['load_amplitude']

			# Dynamic stress intensity factor
			K_I = load_amplitude * np.sqrt(np.pi * a)

			# Dynamic amplification factor (simplified)
			daf = 1.0
			if wavelength < 10 * params['crack_length']:
				# For shorter wavelengths, dynamic effects become important
				ka = 2 * np.pi * a / wavelength
				daf = 1 + 0.5 * np.sin(ka)

			response[i] = K_I * daf

		elif geometry_type == 'hole':
			a = params['hole_radius']
			sigma_infinity = params['remote_stress']

			# Stress concentration factor
			scf = 3.0  # For circular hole under tension

			# Dynamic amplification factor (simplified)
			daf = 1.0
			if wavelength < 10 * a:
				ka = 2 * np.pi * a / wavelength
				daf = 1 + 0.3 * np.sin(ka)

			response[i] = scf * sigma_infinity * daf

		elif geometry_type == 'v_notch':
			alpha = params['notch_angle']
			depth = params['notch_depth']

			# Get eigenvalues from Williams expansion
			lambda_values = williams_expansion_coefficients(alpha, 'mode1')
			lambda_1 = lambda_values[0]

			# Singularity strength
			stress_singularity = depth ** (-lambda_1)

			# Dynamic amplification factor (simplified)
			daf = 1.0
			if wavelength < 10 * depth:
				kd = 2 * np.pi * depth / wavelength
				daf = 1 + 0.4 * np.sin(kd)

			response[i] = stress_singularity * daf

	return frequencies, response


# %% Example Usage
# Define material properties
material_props = {
	'E': 200e9,  # Young's modulus (Pa)
	'nu': 0.3,  # Poisson's ratio
	'rho': 7800,  # Density (kg/m³)
	'sigma_infinity': 1.0  # Remote stress
}

# Define geometry parameters for different cases
crack_params = {
	'length': 2.0  # Crack length
}

hole_params = {
	'radius': 1.0  # Hole radius
}

vnotch_params = {
	'angle': np.pi / 2  # 90° V-notch
}

# %%
# Visualize complex potentials
print("Visualizing complex potentials for crack...")
visualize_complex_potentials('crack', crack_params, material_props, omega=0)

# %%
print("Visualizing stress field for circular hole...")
visualize_stress_field('hole', hole_params, material_props)

# %%
print("Visualizing vibration modes for V-notch...")
visualize_vibration_modes('v_notch', vnotch_params, material_props)

# %%
print("Performing frequency response analysis for crack...")
frequency_response_analysis('crack', crack_params, material_props)

# %%
print("Comparing stress fields across different geometries...")
compare_geometries()
