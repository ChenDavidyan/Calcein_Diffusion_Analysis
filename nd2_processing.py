import os
import numpy as np
from nd2reader import ND2Reader
import tifffile

# Path to ND2
nd2_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\Calcin_Dextran_CAAX_RFP_20250414.nd2"

# Open ND2
nd2 = ND2Reader(nd2_path)
nd2.iter_axes = ['z']
nd2.bundle_axes = 'yx'

# Get dimensions
z_levels = nd2.sizes["z"]
frame_height = nd2.sizes["y"]
frame_width = nd2.sizes["x"]
num_channels = nd2.sizes["c"]
num_fovs = nd2.sizes.get("v", 1)
frames_per_hour = 60 // 20
frames_to_load = 6 * frames_per_hour

print(f"Found {num_fovs} field(s) of view.")
print(f"Will process channels 1 and 2 (skipping channel 0)")

# Process each FOV
for v in range(num_fovs):
    # Output shape: (T, C, Y, X)
    mip_stack = np.zeros((frames_to_load, 2, frame_height, frame_width), dtype=np.uint16)

    for t in range(frames_to_load):
        nd2.default_coords["t"] = t
        nd2.default_coords["v"] = v
        for i, c in enumerate([1, 2]):
            nd2.default_coords["c"] = c
            z_stack = np.zeros((z_levels, frame_height, frame_width), dtype=np.uint16)
            for z in range(z_levels):
                nd2.default_coords["z"] = z
                z_stack[z] = nd2.get_frame_2D(t=t, c=c, z=z, v=v)
            mip = np.max(z_stack, axis=0)
            mip_stack[t, i] = mip
            print(f"✔ FOV={v}, T={t}, C={c} MIP done.")

    # Save TIFF with axes TCYX
    output_filename = f"fov{v+1:02d}_MIP_6h_channels1and2_TCYX.tif"
    output_path = os.path.join(os.path.dirname(nd2_path), output_filename)
    tifffile.imwrite(
        output_path,
        mip_stack,
        imagej=True,
        metadata={"axes": "TCYX"}
    )
    print(f"✅ Saved: {output_path}")
