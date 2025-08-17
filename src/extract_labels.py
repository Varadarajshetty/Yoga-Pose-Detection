# Load processed image paths
with open("processed_images.txt", "r") as f:
    processed_images = [line.strip() for line in f.readlines()]

# Assuming you have a dictionary mapping image paths to numeric labels
# Example (you might want to create this from your folder structure or another source):
image_to_label = {}
for img_path in processed_images:
    # Extract class label from path (folder name)
    label = os.path.basename(os.path.dirname(img_path))
    image_to_label[img_path] = label  # or convert label to numeric if needed

# Now create labels array corresponding to processed images order
labels = [image_to_label[img] for img in processed_images]

import numpy as np
y = np.array(labels)

# Save label array
np.save("y_labels.npy", y)
print("✅ Labels saved to y_labels.npy")
