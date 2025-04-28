import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class Microhole:
    def __init__(self, path, pixel_size_um):
        self.__path = path
        self.pixel_size_um = pixel_size_um  # microns per pixel
        self.image_stack = tifffile.imread(path)
        self.axes = 'TCYX'  # assuming this order
        self.image_16bit = None  # Placeholder for the original 16-bit image

    def attach_circle_automatically(self, c, r_um):
        r = int(r_um / self.pixel_size_um)  # Convert radius from microns to pixels
        print(f"🔍 Converted radius: {r_um} μm = {r} pixels")

        # Extract first timepoint, specified channel
        self.image_16bit = self.image_stack[0, c]

        # Prepare the display for the image
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)

        # Function to update the image based on sliders (min and max)
        def update(val):
            # Get current slider values (min and max)
            vmin = slider_min.val
            vmax = slider_max.val
            
            # Update the display of the image with new vmin and vmax
            img_display.set_clim(vmin, vmax)
            fig.canvas.draw_idle()

        # Display the original 16-bit image with initial vmin and vmax
        img_display = ax.imshow(self.image_16bit, cmap='gray', vmin=self.image_16bit.min(), vmax=self.image_16bit.max())
        ax.set_title(f"Channel {c}, Original Image")

        # Sliders for min and max intensity
        ax_min = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_max = plt.axes([0.25, 0.15, 0.65, 0.03])
        slider_min = Slider(ax_min, 'Min Intensity', self.image_16bit.min(), self.image_16bit.max(), valinit=self.image_16bit.min())
        slider_max = Slider(ax_max, 'Max Intensity', self.image_16bit.min(), self.image_16bit.max(), valinit=self.image_16bit.max())

        slider_min.on_changed(update)
        slider_max.on_changed(update)

        plt.show()

        # Now display the image with adjusted min and max
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)

        # Display the original image with initial window
        img_display = ax.imshow(self.image_16bit, cmap='gray', vmin=self.image_16bit.min(), vmax=self.image_16bit.max())
        ax.set_title(f"Channel {c}, Adjusted Image")

        slider_min.on_changed(update)
        slider_max.on_changed(update)

        plt.show()

# Example usage of the class
fov1_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\fov01_MIP_6h_channels1and2_TCYX.tif"
fov1 = Microhole(fov1_path, 0.646)  # Assuming 0.646 microns per pixel

fov1.attach_circle_automatically(0, 450)  # Channel 0, radius 450 microns
