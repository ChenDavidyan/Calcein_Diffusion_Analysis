# import os
# import numpy as np
# from nd2reader import ND2Reader
# import tifffile

# # This script processes ND2 files to extract maximum intensity projections (MIPs) from specified channels and saves them as TIFF files.
# # It uses the ND2Reader library to read ND2 files and tifffile to save TIFF files.
# # Ensure you have the required libraries installed:
# # pip install nd2reader tifffile numpy
# # This script is designed to work with ND2 files from Nikon microscopes.

# class ND2Processing:
#     def __init__(self, nd2_path):
#         self.nd2_path = nd2_path
#         self.nd2 = ND2Reader(nd2_path)
#         self.nd2.iter_axes = ['z']
#         self.nd2.bundle_axes = 'yx'
#         self.z_levels = self.nd2.sizes["z"]
#         self.frame_height = self.nd2.sizes["y"]
#         self.frame_width = self.nd2.sizes["x"]
#         self.num_channels = self.nd2.sizes["c"]
#         self.num_fovs = self.nd2.sizes.get("v", 1)
#         self.frames_per_hour = 60 // 20
#         self.frames_to_load = 6 * self.frames_per_hour
#     def process_fovs(self):
#         print(f"Found {self.num_fovs} field(s) of view.")
#         print(f"Will process channels 1 and 2 (skipping channel 0)")

#         # Process each FOV
#         for v in range(self.num_fovs):
#             # Output shape: (T, C, Y, X)
#             mip_stack = np.zeros((self.frames_to_load, 2, self.frame_height, self.frame_width), dtype=np.uint16)

#             for t in range(self.frames_to_load):
#                 self.nd2.default_coords["t"] = t
#                 self.nd2.default_coords["v"] = v
#                 for i, c in enumerate([1, 2]):
#                     self.nd2.default_coords["c"] = c
#                     z_stack = np.zeros((self.z_levels, self.frame_height, self.frame_width), dtype=np.uint16)
#                     for z in range(self.z_levels):
#                         self.nd2.default_coords["z"] = z
#                         z_stack[z] = self.nd2.get_frame_2D(t=t, c=c, z=z, v=v)
#                     mip = np.max(z_stack, axis=0)
#                     mip_stack[t, i] = mip
#                     print(f"✔ FOV={v}, T={t}, C={c} MIP done.")

#             # Save TIFF with axes TCYX
#             output_filename = f"fov{v+1:02d}_MIP_6h_channels1and2_TCYX.tif"
#             output_path = os.path.join(os.path.dirname(self.nd2_path), output_filename)
#             tifffile.imwrite(
#                 output_path,
#                 mip_stack,
#                 imagej=True,
#                 metadata={"axes": "TCYX"}
#             )
#             print(f"✅ Saved: {output_path}")
# # Example usage of the ND2Processing class
# if __name__ == "__main__":
#     # Path to ND2
#     nd2_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\Calcin_Dextran_CAAX_RFP_20250414.nd2"

#     # Create an instance of the ND2Processing class
#     processor = ND2Processing(nd2_path)
    
#     # Process the FOVs
#     processor.process_fovs()



# # Path to ND2
# nd2_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\Calcin_Dextran_CAAX_RFP_20250414.nd2"

# # Open ND2
# nd2 = ND2Reader(nd2_path)
# nd2.iter_axes = ['z']
# nd2.bundle_axes = 'yx'

# # Get dimensions
# z_levels = nd2.sizes["z"]
# frame_height = nd2.sizes["y"]
# frame_width = nd2.sizes["x"]
# num_channels = nd2.sizes["c"]
# num_fovs = nd2.sizes.get("v", 1)
# frames_per_hour = 60 // 20
# frames_to_load = 6 * frames_per_hour

# print(f"Found {num_fovs} field(s) of view.")
# print(f"Will process channels 1 and 2 (skipping channel 0)")

# # Process each FOV
# for v in range(num_fovs):
#     # Output shape: (T, C, Y, X)
#     mip_stack = np.zeros((frames_to_load, 2, frame_height, frame_width), dtype=np.uint16)

#     for t in range(frames_to_load):
#         nd2.default_coords["t"] = t
#         nd2.default_coords["v"] = v
#         for i, c in enumerate([1, 2]):
#             nd2.default_coords["c"] = c
#             z_stack = np.zeros((z_levels, frame_height, frame_width), dtype=np.uint16)
#             for z in range(z_levels):
#                 nd2.default_coords["z"] = z
#                 z_stack[z] = nd2.get_frame_2D(t=t, c=c, z=z, v=v)
#             mip = np.max(z_stack, axis=0)
#             mip_stack[t, i] = mip
#             print(f"✔ FOV={v}, T={t}, C={c} MIP done.")

#     # Save TIFF with axes TCYX
#     output_filename = f"fov{v+1:02d}_MIP_6h_channels1and2_TCYX.tif"
#     output_path = os.path.join(os.path.dirname(nd2_path), output_filename)
#     tifffile.imwrite(
#         output_path,
#         mip_stack,
#         imagej=True,
#         metadata={"axes": "TCYX"}
#     )
#     print(f"✅ Saved: {output_path}")



import os
import numpy as np
import pandas as pd
from datetime import datetime
from nd2reader import ND2Reader
import tifffile

# class ND2Processing:
#     def __init__(self, nd2_path, channels=None, frames_to_load=None):
#         """
#         nd2_path: path to the ND2 file
#         channels: list of channels to process (e.g., [1, 2]). If None, process all channels.
#         frames_to_load: number of timepoints to load. If None, load all available.
#         """
#         self.nd2_path = nd2_path
#         self.nd2 = ND2Reader(nd2_path)
#         self.nd2.iter_axes = ['z']
#         self.nd2.bundle_axes = 'yx'
#         self.z_levels = self.nd2.sizes["z"]
#         self.frame_height = self.nd2.sizes["y"]
#         self.frame_width = self.nd2.sizes["x"]
#         self.num_channels = self.nd2.sizes["c"]
#         self.num_fovs = self.nd2.sizes.get("v", 1)
#         self.total_timepoints = self.nd2.sizes["t"]

#         if channels is None:
#             self.channels = list(range(self.num_channels))
#         else:
#             self.channels = channels

#         if frames_to_load is None:
#             self.frames_to_load = self.total_timepoints
#         else:
#             if frames_to_load > self.total_timepoints:
#                 raise ValueError(f"Requested {frames_to_load} frames, but ND2 only has {self.total_timepoints} frames.")
#             self.frames_to_load = frames_to_load

#         # Try to extract pixel size (in µm)
#         self.pixel_size_um = self.get_pixel_size_um()

#         # Try to extract z-step size (in µm)
#         self.z_step_um = self.get_z_step_um()

#         # For CSV summary
#         self.summary_data = []

#     def get_z_step_um(self):
#         """Extract z-step size (µm) from ND2 metadata."""
#         try:
#             z_step = self.nd2.metadata.get("z_step_microns")
#             if z_step is None:
#                 # Older versions may store calibration differently
#                 z_step = self.nd2.metadata.get("calibration", {}).get("z", None)
#             if isinstance(z_step, (int, float)):
#                 return z_step
#         except Exception as e:
#             print(f"⚠️ Warning: Couldn't extract Z-step size: {e}")
#         return None  # if unavailable


#     def get_pixel_size_um(self):
#         """Extract pixel size (µm/pixel) from ND2 metadata."""
#         try:
#             # Newer versions
#             px_size = self.nd2.metadata.get("pixel_microns")
#             if px_size is None:
#                 # Older versions may store calibration separately
#                 px_size = self.nd2.metadata.get("calibration", {}).get("pixel_microns")
#             if isinstance(px_size, (int, float)):
#                 return px_size
#             elif isinstance(px_size, dict):
#                 return px_size.get("x")  # assume square pixels, so x = y
#         except Exception as e:
#             print(f"⚠️ Warning: Couldn't extract pixel size automatically: {e}")
#         return None  # if unavailable

#     def process_fovs(self):
#         print(f"Found {self.num_fovs} field(s) of view.")
#         print(f"Processing channels: {self.channels}")
#         print(f"Processing {self.frames_to_load} timepoints out of {self.total_timepoints} available.")
#         if self.pixel_size_um:
#             print(f"Pixel size: {self.pixel_size_um:.4f} µm/pixel")
#         else:
#             print("Pixel size not available.")

#         # Save original dimension info
#         original_dim = "TCZYX"

#         # Process each FOV
#         for v in range(self.num_fovs):
#             mip_stack = np.zeros((self.frames_to_load, len(self.channels), self.frame_height, self.frame_width), dtype=np.uint16)

#             for t in range(self.frames_to_load):
#                 self.nd2.default_coords["t"] = t
#                 self.nd2.default_coords["v"] = v
#                 for i, c in enumerate(self.channels):
#                     self.nd2.default_coords["c"] = c
#                     z_stack = np.zeros((self.z_levels, self.frame_height, self.frame_width), dtype=np.uint16)
#                     for z in range(self.z_levels):
#                         self.nd2.default_coords["z"] = z
#                         z_stack[z] = self.nd2.get_frame_2D(t=t, c=c, z=z, v=v)
#                     mip = np.max(z_stack, axis=0)
#                     mip_stack[t, i] = mip
#                     print(f"✔ FOV={v}, T={t}, C={c} MIP done.")

#             # Save TIFF with axes TCYX
#             channels_str = "_".join(str(c) for c in self.channels)
#             output_filename = f"fov{v+1:02d}_MIP_{self.frames_to_load}T_channels_{channels_str}_TCYX.tif"
#             output_path = os.path.join(os.path.dirname(self.nd2_path), output_filename)
#             tifffile.imwrite(
#                 output_path,
#                 mip_stack,
#                 imagej=True,
#                 metadata={"axes": "TCYX"}
#             )
#             print(f"✅ Saved: {output_path}")

#             # Record summary info
#             self.summary_data.append({
#                 "fov_num": v + 1,
#                 "path": output_path,
#                 "dim": "TCYX",
#                 "date_time_of_processing": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "original_data_path": self.nd2_path,
#                 "original_dim": original_dim,
#                 "pixel_size_um": self.pixel_size_um,
#                 "z_step_um": self.z_step_um

#             })

#         # After processing all FOVs, save the summary CSV
#         summary_df = pd.DataFrame(self.summary_data)
#         summary_csv_path = os.path.join(os.path.dirname(self.nd2_path), "processing_summary.csv")
#         summary_df.to_csv(summary_csv_path, index=False)
#         print(f"📄 Summary CSV saved at: {summary_csv_path}")

# # Example usage
# if __name__ == "__main__":
#     nd2_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\Calcin_Dextran_CAAX_RFP_20250414.nd2"

#     processor = ND2Processing(
#         nd2_path,
#         channels=[1, 2],        # or None for all channels
#         frames_to_load=18       # or None for all timepoints
#     )
    
#     processor.process_fovs()


# import os
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from nd2reader import ND2Reader
# import nd2
# import tifffile

# class ND2Processing:
#     def __init__(self, nd2_path, channels=None, frames_to_load=None):
#         """
#         nd2_path: path to the ND2 file
#         channels: list of channels to process (e.g., [1, 2]). If None, process all channels.
#         frames_to_load: number of timepoints to load. If None, load all available.
#         """
#         self.nd2_path = nd2_path
#         self.nd2 = ND2Reader(nd2_path)
#         print(self.nd2.metadata)
#         self.nd2.iter_axes = ['z']
#         self.nd2.bundle_axes = 'yx'
#         self.z_levels = self.nd2.sizes["z"]
#         self.frame_height = self.nd2.sizes["y"]
#         self.frame_width = self.nd2.sizes["x"]
#         self.num_channels = self.nd2.sizes["c"]
#         self.num_fovs = self.nd2.sizes.get("v", 1)
#         self.total_timepoints = self.nd2.sizes["t"]

#         if channels is None:
#             self.channels = list(range(self.num_channels))
#         else:
#             self.channels = channels

#         if frames_to_load is None:
#             self.frames_to_load = self.total_timepoints
#         else:
#             if frames_to_load > self.total_timepoints:
#                 raise ValueError(f"Requested {frames_to_load} frames, but ND2 only has {self.total_timepoints} frames.")
#             self.frames_to_load = frames_to_load

#         # Try to extract pixel size (in µm)
#         self.pixel_size_um = self.get_pixel_size_um()

#         # Try to extract z-step size (in µm)
#         self.z_step_um = self.get_z_step_um()

#         # For CSV summary
#         self.summary_data = []

#     def get_z_step_um(self):
#         """Extract z-step size (µm) from ND2 metadata."""
#         try:
#             z_step = self.nd2.metadata.get("z_step_microns")
#             if z_step is None:
#                 # Older versions may store calibration differently
#                 z_step = self.nd2.metadata.get("calibration", {}).get("z", None)
#             if isinstance(z_step, (int, float)):
#                 return z_step
#         except Exception as e:
#             print(f"⚠️ Warning: Couldn't extract Z-step size: {e}")
#         return None  # if unavailable

#     def get_pixel_size_um(self):
#         """Extract pixel size (µm/pixel) from ND2 metadata."""
#         try:
#             # Newer versions
#             px_size = self.nd2.metadata.get("pixel_microns")
#             if px_size is None:
#                 # Older versions may store calibration separately
#                 px_size = self.nd2.metadata.get("calibration", {}).get("pixel_microns")
#             if isinstance(px_size, (int, float)):
#                 return px_size
#             elif isinstance(px_size, dict):
#                 return px_size.get("x")  # assume square pixels, so x = y
#         except Exception as e:
#             print(f"⚠️ Warning: Couldn't extract pixel size automatically: {e}")
#         return None  # if unavailable

#     def process_fovs(self):
#         print(f"Found {self.num_fovs} field(s) of view.")
#         print(f"Processing channels: {self.channels}")
#         print(f"Processing {self.frames_to_load} timepoints out of {self.total_timepoints} available.")

#         if self.pixel_size_um:
#             print(f"Pixel size (XY): {self.pixel_size_um:.4f} µm/pixel")
#         else:
#             print("Pixel size (XY) not available.")

#         if self.z_step_um:
#             print(f"Z-step size: {self.z_step_um:.4f} µm")
#         else:
#             print("Z-step size not available.")

#         # Save original dimension info
#         original_dim = "TCZYX"

#         # Process each FOV
#         for v in range(self.num_fovs):
#             mip_stack = np.zeros((self.frames_to_load, len(self.channels), self.frame_height, self.frame_width), dtype=np.uint16)

#             for t in range(self.frames_to_load):
#                 self.nd2.default_coords["t"] = t
#                 self.nd2.default_coords["v"] = v
#                 for i, c in enumerate(self.channels):
#                     self.nd2.default_coords["c"] = c
#                     z_stack = np.zeros((self.z_levels, self.frame_height, self.frame_width), dtype=np.uint16)
#                     for z in range(self.z_levels):
#                         self.nd2.default_coords["z"] = z
#                         z_stack[z] = self.nd2.get_frame_2D(t=t, c=c, z=z, v=v)
#                     mip = np.max(z_stack, axis=0)
#                     mip_stack[t, i] = mip
#                     print(f"✔ FOV={v}, T={t}, C={c} MIP done.")

#             # Save TIFF with axes TCYX
#             channels_str = "_".join(str(c) for c in self.channels)
#             output_filename = f"fov{v+1:02d}_MIP_{self.frames_to_load}T_channels_{channels_str}_TCYX.tif"
#             output_path = os.path.join(os.path.dirname(self.nd2_path), output_filename)
#             tifffile.imwrite(
#                 output_path,
#                 mip_stack,
#                 imagej=True,
#                 metadata={"axes": "TCYX"}
#             )
#             print(f"✅ Saved: {output_path}")

#             # Record summary info
#             self.summary_data.append({
#                 "fov_num": v + 1,
#                 "path": output_path,
#                 "dim": "TCYX",
#                 "date_time_of_processing": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "original_data_path": self.nd2_path,
#                 "original_dim": original_dim,
#                 "pixel_size_um": self.pixel_size_um,
#                 "z_step_um": self.z_step_um
#             })

#         # After processing all FOVs, save the summary CSV
#         summary_df = pd.DataFrame(self.summary_data)
#         summary_csv_path = os.path.join(os.path.dirname(self.nd2_path), "processing_summary.csv")
#         summary_df.to_csv(summary_csv_path, index=False)
#         print(f"📄 Summary CSV saved at: {summary_csv_path}")

# # Example usage
# if __name__ == "__main__":
#     nd2_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\Calcin_Dextran_CAAX_RFP_20250414.nd2"

#     processor = ND2Processing(
#         nd2_path,
#         channels=[1, 2],        # or None for all channels
#         frames_to_load=18       # or None for all timepoints
#     )
    
#     processor.process_fovs()

import os
import numpy as np
import pandas as pd
from datetime import datetime
from nd2reader import ND2Reader
import nd2
import tifffile

class ND2Processing:
    def __init__(self, nd2_path, channels=None, frames_to_load=None):
        """
        nd2_path: path to the ND2 file
        channels: list of channels to process (e.g., [1, 2]). If None, process all channels.
        frames_to_load: number of timepoints to load. If None, load all available.
        """
        self.nd2_path = nd2_path
        self.nd2 = ND2Reader(nd2_path)
        print(self.nd2.metadata)
        self.nd2.iter_axes = ['z']
        self.nd2.bundle_axes = 'yx'
        self.z_levels = self.nd2.sizes["z"]
        self.frame_height = self.nd2.sizes["y"]
        self.frame_width = self.nd2.sizes["x"]
        self.num_channels = self.nd2.sizes["c"]
        self.num_fovs = self.nd2.sizes.get("v", 1)
        self.total_timepoints = self.nd2.sizes["t"]

        if channels is None:
            self.channels = list(range(self.num_channels))
        else:
            self.channels = channels

        if frames_to_load is None:
            self.frames_to_load = self.total_timepoints
        else:
            if frames_to_load > self.total_timepoints:
                raise ValueError(f"Requested {frames_to_load} frames, but ND2 only has {self.total_timepoints} frames.")
            self.frames_to_load = frames_to_load

        # Try to extract pixel size (in µm)
        self.pixel_size_um = self.get_pixel_size_um()

        # Try to extract z-step size (in µm)
        self.z_step_um = self.get_z_step_um()

        # For CSV summary
        self.summary_data = []

    def get_z_step_um(self):
        """Extract z-step size (µm) from ND2 metadata using `nd2` (python-nd2) if possible."""
        try:
            with nd2.ND2File(self.nd2_path) as nd2file:
                    meta = nd2file.metadata
                    z_step = meta.channels[0].volume.axesCalibration[2]
                    print(f"🔵 Extracted z step using python-nd2: {z_step} µm")
                    return z_step
        except Exception as e:
            print(f"⚠️ Warning: Couldn't extract Z-step size using python-nd2: {e}")
        
    def get_pixel_size_um(self):
        """Extract pixel size (µm/pixel) from ND2 metadata using `nd2` (python-nd2) if possible."""
        try:
            with nd2.ND2File(self.nd2_path) as nd2file:
                px_size = nd2file.metadata.scale["X"]  # micrometers/pixel
                print(f"🔵 Extracted pixel size using python-nd2: {px_size} µm/pixel")
                return px_size
        except Exception as e:
            print(f"⚠️ Warning: Couldn't extract pixel size using python-nd2: {e}")

        # fallback to nd2reader
        try:
            px_size = self.nd2.metadata.get("pixel_microns")
            if px_size is None:
                px_size = self.nd2.metadata.get("calibration", {}).get("pixel_microns")
            if isinstance(px_size, (int, float)):
                print(f"🟡 Extracted pixel size using nd2reader: {px_size} µm/pixel")
                return px_size
            elif isinstance(px_size, dict):
                px_size_value = px_size.get("x")
                print(f"🟡 Extracted pixel size using nd2reader dict: {px_size_value} µm/pixel")
                return px_size_value
        except Exception as e:
            print(f"⚠️ Warning: Couldn't extract pixel size using nd2reader: {e}")
        return None  # if unavailable

    def process_fovs(self):
        print(f"Found {self.num_fovs} field(s) of view.")
        print(f"Processing channels: {self.channels}")
        print(f"Processing {self.frames_to_load} timepoints out of {self.total_timepoints} available.")

        if self.pixel_size_um:
            print(f"Pixel size (XY): {self.pixel_size_um:.4f} µm/pixel")
        else:
            print("Pixel size (XY) not available.")

        if self.z_step_um:
            print(f"Z-step size: {self.z_step_um:.4f} µm")
        else:
            print("Z-step size not available.")

        # Save original dimension info
        original_dim = "TCZYX"

        # Process each FOV
        for v in range(self.num_fovs):
            mip_stack = np.zeros((self.frames_to_load, len(self.channels), self.frame_height, self.frame_width), dtype=np.uint16)

            for t in range(self.frames_to_load):
                self.nd2.default_coords["t"] = t
                self.nd2.default_coords["v"] = v
                for i, c in enumerate(self.channels):
                    self.nd2.default_coords["c"] = c
                    z_stack = np.zeros((self.z_levels, self.frame_height, self.frame_width), dtype=np.uint16)
                    for z in range(self.z_levels):
                        self.nd2.default_coords["z"] = z
                        z_stack[z] = self.nd2.get_frame_2D(t=t, c=c, z=z, v=v)
                    mip = np.max(z_stack, axis=0)
                    mip_stack[t, i] = mip
                    print(f"✔ FOV={v}, T={t}, C={c} MIP done.")

            # Save TIFF with axes TCYX
            channels_str = "_".join(str(c) for c in self.channels)
            output_filename = f"fov{v+1:02d}_MIP_{self.frames_to_load}T_channels_{channels_str}_TCYX.tif"
            output_path = os.path.join(os.path.dirname(self.nd2_path), output_filename)
            tifffile.imwrite(
                output_path,
                mip_stack,
                imagej=True,
                metadata={"axes": "TCYX"}
            )
            print(f"✅ Saved: {output_path}")

            # Record summary info
            self.summary_data.append({
                "fov_num": v + 1,
                "path": output_path,
                "dim": "TCYX",
                "date_time_of_processing": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "original_data_path": self.nd2_path,
                "original_dim": original_dim,
                "pixel_size_um": self.pixel_size_um,
                "z_step_um": self.z_step_um
            })

        # After processing all FOVs, save the summary CSV
        summary_df = pd.DataFrame(self.summary_data)
        summary_csv_path = os.path.join(os.path.dirname(self.nd2_path), "processing_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"📄 Summary CSV saved at: {summary_csv_path}")

# Example usage
if __name__ == "__main__":
    nd2_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\Calcin_Dextran_CAAX_RFP_20250414.nd2"

    processor = ND2Processing(
        nd2_path,
        channels=[1, 2],        # or None for all channels
        frames_to_load=18       # or None for all timepoints
    )
    
    processor.process_fovs()
