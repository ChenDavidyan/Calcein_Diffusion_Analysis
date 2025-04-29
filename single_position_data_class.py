import tifffile
import numpy as np
from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
import napari
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.draw import disk
import os
from tqdm import tqdm

class Microhole:

    def __init__(self, metadata_row):
        """
        metadata_row: a pandas Series (or dict) with keys:
            - 'path': path to the FOV TIFF file
            - 'pixel_size_um': microns per pixel
        """
        self.path = metadata_row["path"]
        self.pixel_size_um = metadata_row["pixel_size_um"]
        self.image_stack = tifffile.imread(self.path)  # Shape: T, C, Y, X

    def _attach_circle_gui(self, image, title="Manual Circle Drawer"):
        app = QApplication.instance() or QApplication([])

        main_window = QWidget()
        layout = QVBoxLayout(main_window)

        viewer = napari.Viewer()
        viewer_window = viewer.window._qt_window
        layout.addWidget(viewer_window)

        viewer.add_image(image, name='Image', contrast_limits=[np.min(image), np.max(image)])

        shapes_layer = viewer.add_shapes(
            shape_type='ellipse',
            edge_color='red',
            face_color='transparent',
            name='Draw Circles'
        )

        result = {}

        done_button = QPushButton("Done")

        def done_clicked():
            shapes = shapes_layer.data
            n_shapes = len(shapes)

            if n_shapes == 0:
                print("⚠️ No shapes drawn. Please draw one circle.")
                return

            if n_shapes > 1:
                print(f"⚠️ {n_shapes} shapes detected. Please delete all but one before clicking Done.")
                return

            ellipse = shapes[0]
            center = np.mean(ellipse, axis=0)
            radius_px = np.mean(np.abs(ellipse - center))
            radius_um = radius_px * self.pixel_size_um

            print(f"✅ Final Circle: Center=({center[1]:.1f}, {center[0]:.1f}), Radius={radius_um:.2f} μm")

            result['center'] = (center[1], center[0])  # (X, Y)
            result['radius_um'] = radius_um

            main_window.close()

        done_button.clicked.connect(done_clicked)
        layout.addWidget(done_button)

        main_window.setWindowTitle(title)
        main_window.resize(1000, 800)
        main_window.show()

        app.exec_()

        if 'center' in result:
            return result['center'], result['radius_um']
        else:
            return None, None

    def attach_circle_manually(self, channel_index=0):
        image_16bit = self.image_stack[0, channel_index]
        return self._attach_circle_gui(image_16bit, title="Attach Circle - First Timepoint. Please define the source border.")

    def _safe_disk(self, center, radius, shape):
        rr, cc = disk(center, radius)
        rr = np.clip(rr, 0, shape[0] - 1)
        cc = np.clip(cc, 0, shape[1] - 1)
        return rr, cc

    def radial_intensity_profile(self, center_xy, channel_index=0, ring_width_um=1.0, save_prefix=None):
        """
        Measure radial intensity profile around center for each timepoint.

        Parameters:
        - center_xy: (x, y) center in pixel coordinates
        - channel_index: channel to analyze
        - ring_width_um: width of each ring in microns
        - save_prefix: prefix for saving CSV (default: uses image path)

        Returns:
        - DataFrame with columns: ['radius_um', 'intensity', 'time_index']
        """
        x0, y0 = center_xy
        image_shape = self.image_stack.shape[2:]
        max_radius_px = int(np.hypot(*image_shape))

        ring_width_px = int(np.round(ring_width_um / self.pixel_size_um))
        if ring_width_px < 1:
            raise ValueError("Ring width is smaller than one pixel.")

        all_profiles = []

        print("Computing radial intensity profiles...")
        for t in tqdm(range(self.image_stack.shape[0]), desc="Timepoints"):
            image = self.image_stack[t, channel_index]
            ring_means = []
            ring_radii_um = []

            for r_px in range(0, max_radius_px, ring_width_px):
                r_outer = r_px + ring_width_px

                mask_outer = np.zeros_like(image, dtype=bool)
                rr_outer, cc_outer = self._safe_disk((y0, x0), r_outer, image.shape)
                mask_outer[rr_outer, cc_outer] = True

                mask_inner = np.zeros_like(image, dtype=bool)
                rr_inner, cc_inner = self._safe_disk((y0, x0), r_px, image.shape)
                mask_inner[rr_inner, cc_inner] = True

                ring_mask = mask_outer ^ mask_inner
                if np.any(ring_mask):
                    mean_intensity = image[ring_mask].mean()
                    ring_means.append(mean_intensity)
                    ring_radii_um.append((r_px + r_outer) / 2 * self.pixel_size_um)

            for r_um, intensity in zip(ring_radii_um, ring_means):
                all_profiles.append({
                    "radius_um": r_um,
                    "intensity": intensity,
                    "time_index": t
                })

        df_all = pd.DataFrame(all_profiles)

        if save_prefix is not None:
            df_all.to_csv(f"{save_prefix}_radial_profile.csv", index=False)

        return df_all


def plot_radial_profile(df, save_path=None, palette_name="rocket_r"):
    """
    Plot radial intensity profile from DataFrame.
    """
    n_timepoints = df['time_index'].nunique()
    palette = sns.color_palette(palette_name, n_colors=n_timepoints)

    plt.figure(figsize=(8, 6))
    for t, group in df.groupby("time_index"):
        plt.plot(group["radius_um"], group["intensity"], marker='o', label=f'Time {t}', color=palette[t])

    plt.xlabel("Distance from Center (µm)")
    plt.ylabel("Mean Intensity")
    plt.title("Radial Intensity Profile Over Time")
    plt.legend(title="Timepoint", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# === Example usage ===
if __name__ == "__main__":
    summary_csv_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\processing_summary.csv"
    df = pd.read_csv(summary_csv_path)

    row = df[df["fov_num"] == 1].iloc[0]
    fov = Microhole(row)

    center, radius_um = fov.attach_circle_manually(channel_index=0)
    print(f"Circle center: {center}, radius: {radius_um:.2f} μm")
    if center:
        df_profile = fov.radial_intensity_profile(center_xy=center, channel_index=0, ring_width_um=50, save_prefix="fov01")
        plot_radial_profile(df_profile, save_path="fov01_radial_profile.png")


# class Microhole:

#     def __init__(self, metadata_row):
#         """
#         metadata_row: a pandas Series (or dict) with keys:
#             - 'path': path to the FOV TIFF file
#             - 'pixel_size_um': microns per pixel
#         """
#         self.path = metadata_row["path"]
#         self.pixel_size_um = metadata_row["pixel_size_um"]
#         self.image_stack = tifffile.imread(self.path)  # Shape: T, C, Y, X

#     def _attach_circle_gui(self, image, title="Manual Circle Drawer"):
#         """Helper function to open a Napari GUI, let user draw circles, and return center and radius of the last circle."""
#         app = QApplication.instance() or QApplication([])

#         main_window = QWidget()
#         layout = QVBoxLayout(main_window)

#         viewer = napari.Viewer()
#         viewer_window = viewer.window._qt_window
#         layout.addWidget(viewer_window)

#         viewer.add_image(image, name='Image', contrast_limits=[np.min(image), np.max(image)])

#         shapes_layer = viewer.add_shapes(
#             shape_type='ellipse',
#             edge_color='red',
#             face_color='transparent',
#             name='Draw Circles'
#         )

#         result = {}

#         done_button = QPushButton("Done")

#         def done_clicked():
#             shapes = shapes_layer.data
#             n_shapes = len(shapes)

#             if n_shapes == 0:
#                 print("⚠️ No shapes drawn. Please draw one circle.")
#                 return

#             if n_shapes > 1:
#                 print(f"⚠️ {n_shapes} shapes detected. Please delete all but one before clicking Done.")
#                 return

#             # Exactly one circle
#             ellipse = shapes[0]
#             center = np.mean(ellipse, axis=0)
#             radius_px = np.mean(np.abs(ellipse - center))
#             radius_um = radius_px * self.pixel_size_um

#             print(f"✅ Final Circle: Center=({center[1]:.1f}, {center[0]:.1f}), Radius={radius_um:.2f} μm")

#             result['center'] = (center[1], center[0])  # (X, Y)
#             result['radius_um'] = radius_um

#             main_window.close()

#         done_button.clicked.connect(done_clicked)
#         layout.addWidget(done_button)

#         main_window.setWindowTitle(title)
#         main_window.resize(1000, 800)
#         main_window.show()

#         app.exec_()

#         if 'center' in result:
#             return result['center'], result['radius_um']
#         else:
#             return None, None

#     def attach_circle_manually(self, channel_index=0):
#         """Open the first timepoint and let user attach circles manually. Return center and radius."""
#         image_16bit = self.image_stack[0, channel_index]
#         return self._attach_circle_gui(image_16bit, title="Attach Circle - First Timepoint. Pleade define the sorce border.")
    
#     def radial_intensity_profile(self, center_xy, channel_index=0, ring_width_um=1.0, save_prefix=None):
#         """
#         Measure radial intensity profile around center for each timepoint.
        
#         Parameters:
#         - center_xy: (x, y) center in pixel coordinates
#         - channel_index: channel to analyze
#         - ring_width_um: width of each ring in microns
#         - save_prefix: prefix for saving CSV (default: uses image path)
        
#         Returns:
#         - DataFrame with columns: ['radius_um', 'intensity', 'time_index']
#         """
#         x0, y0 = center_xy
#         image_shape = self.image_stack.shape[2:]
#         max_radius_px = int(np.hypot(*image_shape))  # go up to corners

#         ring_width_px = int(np.round(ring_width_um / self.pixel_size_um))
#         if ring_width_px < 1:
#             raise ValueError("Ring width is smaller than one pixel.")

#         all_profiles = []

#         print("Computing radial intensity profiles...")
#         for t in tqdm(range(self.image_stack.shape[0]), desc="Timepoints"):
#             image = self.image_stack[t, channel_index]
#             ring_means = []
#             ring_radii_um = []

#             for r_px in range(0, max_radius_px, ring_width_px):
#                 r_outer = r_px + ring_width_px
#                 mask_outer = np.zeros_like(image, dtype=bool)
#                 rr, cc = disk((y0, x0), r_outer)
#                 mask_outer[rr, cc] = True

#                 mask_inner = np.zeros_like(image, dtype=bool)
#                 rr, cc = disk((y0, x0), r_px)
#                 mask_inner[rr, cc] = True

#                 ring_mask = mask_outer ^ mask_inner
#                 if np.any(ring_mask):
#                     mean_intensity = image[ring_mask].mean()
#                     ring_means.append(mean_intensity)
#                     ring_radii_um.append((r_px + r_outer) / 2 * self.pixel_size_um)

#             for r_um, intensity in zip(ring_radii_um, ring_means):
#                 all_profiles.append({
#                     "radius_um": r_um,
#                     "intensity": intensity,
#                     "time_index": t
#                 })

#         df_all = pd.DataFrame(all_profiles)

#         if save_prefix is not None:
#             df_all.to_csv(f"{save_prefix}_radial_profile.csv", index=False)

#         return df_all

#     def plot_radial_profile(df, save_path=None, palette_name="rocket_r"):
#         """
#         Plot radial intensity profile from DataFrame (as returned by radial_intensity_profile).
#         """
#         import seaborn as sns
#         import matplotlib.pyplot as plt

#         n_timepoints = df['time_index'].nunique()
#         palette = sns.color_palette(palette_name, n_colors=n_timepoints)

#         plt.figure(figsize=(8, 6))
#         for t, group in df.groupby("time_index"):
#             plt.plot(group["radius_um"], group["intensity"], marker='o', label=f'Time {t}', color=palette[t])

#         plt.xlabel("Distance from Center (µm)")
#         plt.ylabel("Mean Intensity")
#         plt.title("Radial Intensity Profile Over Time")
#         plt.legend(title="Timepoint", bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.tight_layout()

#         if save_path:
#             plt.savefig(save_path, dpi=300)
#         plt.show()

# # Example usage
# if __name__ == "__main__":
#     summary_csv_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\processing_summary.csv"
#     df = pd.read_csv(summary_csv_path)

#     # Work on one FOV (e.g., FOV 1)
#     row = df[df["fov_num"] == 1].iloc[0]
#     fov = Microhole(row)

#     center, radius_um = fov.attach_circle_manually(channel_index=0)
#     print(f"Circle center: {center}, radius: {radius_um:.2f} μm")
#     if center:
#         df_profile = fov.radial_intensity_profile(center_xy=center, channel_index=0, ring_width_um=50, save_prefix="fov01")
#         plot_radial_profile(df_profile, save_path="fov01_radial_profile.png")
