import matplotlib.pyplot as plt
import numpy as np

# Sample data: 4 objects, each with 4 points (toe/camber, 2 methods)
objects = ['1D', '2A', '2D', 'Cold Roll']
x_base = np.array([1.00, 1.12, 1.24, 1.36, 1.48, 1.60])  # X positions for each group

# y_toe_CMM =      np.array([-0.3135, 0.1595, -0.1472,  0.0278])
# y_camber_CMM =   np.array([ 0.5373, 0.0899,  0.3859, -0.0526])
# Swapped toe and camber
y_toe_CMM =      np.array([ 0.5373, 0.0899,  0.3859, -0.0526])
y_camber_CMM =   np.array([-0.3135, 0.1595, -0.1472,  0.0278])
y_misalign_CMM = np.array([ 0.6221, 0.1838,  0.4147,  0.0595])

y_toe_Scan =      np.array([-0.4950, -0.0016, -0.2381, -0.0667])
y_camber_Scan =   np.array([-0.3156,  0.1847, -0.0793,  0.0166])
y_misalign_Scan = np.array([ 0.5871,  0.1848,  0.2510,  0.0689])

errors_toe_CMM =      np.array([0.0395, 0.0136, 0.1460, 0.0016])
errors_camber_CMM =   np.array([0.0418, 0.0663, 0.0182, 0.0010])
errors_misalign_CMM = np.array([0.0557, 0.0318, 0.0601, 0.0007])

errors_toe_Scan =     np.array([0.0298, 0.0177, 0.0267, 0.0390])
errors_camber_Scan =  np.array([0.0039, 0.0078, 0.0089, 0.0094])
errors_misalign_Scan = np.array([0.0245, 0.0076, 0.0264, 0.0395])

# Plotting
fig, ax = plt.subplots()

# Plot bars for each measurement
for i, obj in enumerate(objects):
    x = x_base + i * 2  # Shift each group
    ax.errorbar(x[0], y_toe_CMM[i],       yerr=errors_toe_CMM[i],       fmt='o', color='blue', label='Toe CMM'       if i == 0 else "", capsize=2)
    ax.errorbar(x[2], y_camber_CMM[i],    yerr=errors_camber_CMM[i],    fmt='s', color='blue', label='Camber CMM'    if i == 0 else "", capsize=2)
    ax.errorbar(x[4], y_misalign_CMM[i],  yerr=errors_misalign_CMM[i],  fmt='x', color='blue', label='Misalign CMM'  if i == 0 else "", capsize=2)
    ax.errorbar(x[1], y_toe_Scan[i],      yerr=errors_toe_Scan[i],      fmt='o', color='red',  label='Toe Scan'      if i == 0 else "", capsize=2)
    ax.errorbar(x[3], y_camber_Scan[i],   yerr=errors_camber_Scan[i],   fmt='s', color='red',  label='Camber Scan'   if i == 0 else "", capsize=2)
    ax.errorbar(x[5], y_misalign_Scan[i], yerr=errors_misalign_Scan[i], fmt='x', color='red',  label='Misalign Scan' if i == 0 else "", capsize=2)

# Customize
ax.set_xticks([1.3 + i * 2 for i in range(len(objects))])
ax.set_xticklabels(objects, fontsize=12)
ax.set_xlabel('Arm IDs', fontsize=14)
ax.set_ylabel('Measured Angle (degrees)', fontsize=14)
ax.set_title('Toe and Camber Comparison by Method', fontsize=16)
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()