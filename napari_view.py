import tifffile
import numpy as np
from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
import napari


class Microhole:
    def __init__(self, path, pixel_size_um):
        self.path = path
        self.pixel_size_um = pixel_size_um  # microns per pixel
        self.image_stack = tifffile.imread(path)  # Shape: T, C, Y, X

    def attach_circle_manually(self, r_um, channel_index=0):
        r_px = int(r_um / self.pixel_size_um)

        image_16bit = self.image_stack[0, channel_index]  # Get first timepoint, selected channel

        app = QApplication.instance() or QApplication([])

        main_window = QWidget()
        layout = QVBoxLayout(main_window)

        viewer = napari.Viewer()
        viewer_window = viewer.window._qt_window
        layout.addWidget(viewer_window)

        # Add the image to Napari
        viewer.add_image(image_16bit, name=f'Channel {channel_index}', contrast_limits=[np.min(image_16bit), np.max(image_16bit)])

        # Add a circle shapes layer
        shapes_layer = viewer.add_shapes(
            shape_type='ellipse',
            edge_color='red',
            face_color='transparent',
            name='Draw Circle'
        )

        # Create a 'Done' button
        done_button = QPushButton("Done")

        def done_clicked():
            shapes = shapes_layer.data
            if len(shapes) == 0:
                print("⚠️ No shape drawn.")
                main_window.close()
                return
            ellipse = shapes[0]  # [[x1,y1], [x2,y2]]
            center = np.mean(ellipse, axis=0)
            radius_px = np.mean(np.abs(ellipse - center))
            radius_um = radius_px * self.pixel_size_um
            print(f"✅ Center: ({center[1]:.1f}, {center[0]:.1f}), Radius: {radius_um:.2f} μm")
            main_window.close()

        done_button.clicked.connect(done_clicked)
        layout.addWidget(done_button)

        main_window.setWindowTitle("Manual Circle Drawer")
        main_window.resize(1000, 800)
        main_window.show()

        app.exec_()


# Example usage
if __name__ == "__main__":
    fov1_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\fov01_MIP_6h_channels1and2_TCYX.tif"
    fov1 = Microhole(fov1_path, 0.646)
    fov1.attach_circle_manually(r_um=450, channel_index=0)