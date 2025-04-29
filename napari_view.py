# import tifffile
# import numpy as np
# from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
# import napari


# class Microhole:

#     def __init__(self, path, pixel_size_um):
#         self.path = path
#         self.pixel_size_um = pixel_size_um  # microns per pixel
#         self.image_stack = tifffile.imread(path)  # Shape: T, C, Y, X

#     def attach_circle_manually(self, channel_index=0):

#         image_16bit = self.image_stack[0, channel_index]  # Get first timepoint, selected channel

#         app = QApplication.instance() or QApplication([])

#         main_window = QWidget()
#         layout = QVBoxLayout(main_window)

#         viewer = napari.Viewer()
#         viewer_window = viewer.window._qt_window
#         layout.addWidget(viewer_window)

#         # Add the image to Napari
#         viewer.add_image(image_16bit, name=f'Channel {channel_index}', contrast_limits=[np.min(image_16bit), np.max(image_16bit)])

#         # Add a circle shapes layer
#         shapes_layer = viewer.add_shapes(
#             shape_type='ellipse',
#             edge_color='red',
#             face_color='transparent',
#             name='Draw Circle'
#         )

#         # Create a 'Done' button
#         done_button = QPushButton("Done")

#         def done_clicked():
#             shapes = shapes_layer.data
#             if len(shapes) == 0:
#                 print("⚠️ No shape drawn.")
#                 main_window.close()
#                 return
#             ellipse = shapes[0]  # [[x1,y1], [x2,y2]]
#             center = np.mean(ellipse, axis=0)
#             radius_px = np.mean(np.abs(ellipse - center))
#             radius_um = radius_px * self.pixel_size_um
#             print(f"✅ Center: ({center[1]:.1f}, {center[0]:.1f}), Radius: {radius_um:.2f} μm")
#             main_window.close()

#         done_button.clicked.connect(done_clicked)
#         layout.addWidget(done_button)

#         main_window.setWindowTitle("Manual Circle Drawer")
#         main_window.resize(1000, 800)
#         main_window.show()

#         app.exec_()


# # Example usage
# if __name__ == "__main__":
#     fov1_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\fov01_MIP_6h_channels1and2_TCYX.tif"
#     fov1 = Microhole(fov1_path, 0.646)
#     fov1.attach_circle_manually(channel_index=0)


import tifffile
import numpy as np
from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
import napari


class Microhole:

    def __init__(self, path, pixel_size_um):
        self.path = path
        self.pixel_size_um = pixel_size_um  # microns per pixel
        self.image_stack = tifffile.imread(path)  # Shape: T, C, Y, X

    def _attach_circle_gui(self, image, title="Manual Circle Drawer"):
        """Helper function to open a Napari GUI, let user draw circles, and return center and radius of the last circle."""
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

            # Exactly one circle
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
        """Open the first timepoint and let user attach circles manually. Return center and radius."""
        image_16bit = self.image_stack[0, channel_index]
        return self._attach_circle_gui(image_16bit, title="Attach Circle - First Timepoint. Pleade define the sorce border.")

    def attach_border_manually(self, channel_index=0):
        """Open the last timepoint and let user attach circles manually. Return center and radius."""
        image_16bit = self.image_stack[-1, channel_index]
        return self._attach_circle_gui(image_16bit, title="Attach Border - Last Timepoint. Please define the sink border.")


# Example usage
if __name__ == "__main__":
    fov1_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\fov01_MIP_6h_channels1and2_TCYX.tif"
    fov1 = Microhole(fov1_path, 0.646)

    center, radius_um = fov1.attach_circle_manually(channel_index=0)
    print(f"Circle center: {center}, radius: {radius_um:.2f} μm")

    # border_center, border_radius_um = fov1.attach_border_manually(channel_index=0)
    # print(f"Border center: {border_center}, radius: {border_radius_um:.2f} μm")
