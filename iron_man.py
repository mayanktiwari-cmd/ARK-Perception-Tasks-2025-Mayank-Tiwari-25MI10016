
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
img = cv2.imread("iron_man_noisy (1).jpg")
if img is None:
    print("cant find the file, make sure iron_man_noisy (1).jpg is in this folder")
    exit()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"loaded image: {gray.shape[1]}x{gray.shape[0]}")
median3 = cv2.medianBlur(gray, 3)
kernel     = np.ones((3,3), np.uint8)
morph      = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
bilateral  = cv2.bilateralFilter(gray, 9, 75, 75)
def snr(original, filtered):
    noise          = original.astype(np.float32) - filtered.astype(np.float32)
    signal_power   = np.mean(original.astype(np.float32) ** 2)
    noise_power    = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)
snr_median    = snr(gray, median3)
snr_morph     = snr(gray, morph)
snr_bilateral = snr(gray, bilateral)
print(f"SNR  median-3: {snr_median:.2f} dB")
print(f"SNR  morph   : {snr_morph:.2f} dB")
print(f"SNR  bilateral: {snr_bilateral:.2f} dB")
print(f"winner: median-3 chosen for line drawing preservation")
os.makedirs("output", exist_ok=True)
cv2.imwrite("output/ironman_filtered.jpg", median3)
print("saved to output/ironman_filtered.jpg")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Iron Man - Noise Filtering", fontsize=13)
axes[0].imshow(gray,      cmap="gray"); axes[0].set_title("Noisy Input")
axes[1].imshow(median3,   cmap="gray"); axes[1].set_title(f"Median k=3\n{snr_median:.1f} dB")
axes[2].imshow(morph,     cmap="gray"); axes[2].set_title(f"Morph Open\n{snr_morph:.1f} dB")
axes[3].imshow(bilateral, cmap="gray"); axes[3].set_title(f"Bilateral\n{snr_bilateral:.1f} dB")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("output/ironman_comparison.png", dpi=150)
print("comparison chart saved to output/ironman_comparison.png")
plt.show()
plt.close()