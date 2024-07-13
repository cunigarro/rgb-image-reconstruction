import numpy as np
from scipy.optimize import least_squares

# Load spectral reflectance data of color samples (including NIR range)
reflectance_data = np.loadtxt('reflectance_data.txt')  # Shape (num_samples, num_wavelengths)

# Load captured RGB values of the color samples (including NIR channel if available)
captured_rgb = np.loadtxt('captured_rgb.txt')  # Shape (num_samples, 4) if NIR included

# Load illuminant SPD data (including NIR range)
illuminant_spd = np.loadtxt('illuminant_spd.txt')  # Shape (num_wavelengths,)

# Define num_wavelengths from the shape of reflectance_data or illuminant_spd
num_wavelengths = reflectance_data.shape[1]

# Objective function for optimization
def residuals(sensitivity, reflectance_data, captured_rgb, illuminant_spd):
    num_samples, num_wavelengths = reflectance_data.shape
    S_R, S_G, S_B, S_NIR = np.split(sensitivity, 4)  # Assuming 4 channels: R, G, B, NIR

    model_rgb = np.zeros_like(captured_rgb)
    for i in range(num_samples):
        model_rgb[i, 0] = np.sum(S_R * illuminant_spd * reflectance_data[i, :])
        model_rgb[i, 1] = np.sum(S_G * illuminant_spd * reflectance_data[i, :])
        model_rgb[i, 2] = np.sum(S_B * illuminant_spd * reflectance_data[i, :])
        model_rgb[i, 3] = np.sum(S_NIR * illuminant_spd * reflectance_data[i, :])

    return (model_rgb - captured_rgb).ravel()

# Initial guess for the sensitivities (assuming 4 sensors, each with num_wavelengths values)
initial_guess = np.ones(4 * num_wavelengths)

# Perform the optimization
result = least_squares(residuals, initial_guess, args=(reflectance_data, captured_rgb, illuminant_spd))

# Extract the estimated sensitivities
S_R_est, S_G_est, S_B_est, S_NIR_est = np.split(result.x, 4)

# Save or plot the estimated spectral sensitivities
np.savetxt('S_R_est.txt', S_R_est)
np.savetxt('S_G_est.txt', S_G_est)
np.savetxt('S_B_est.txt', S_B_est)
np.savetxt('S_NIR_est.txt', S_NIR_est)
