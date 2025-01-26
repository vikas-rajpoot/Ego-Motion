import cv2
import numpy as np
from flirpy.camera.lepton import Lepton
import subprocess


class ThermalCamera:
    def __init__(self, min_temp=-10.0, max_temp=280.0):
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.hottest_point = None
        self.max_temperature = None
        self.fire_status = None

        self.camera = Lepton()

    def raw_to_celsius(self, raw_data):
        SCALE_FACTOR = 0.01
        OFFSET = 273.15
        return (raw_data * SCALE_FACTOR) - OFFSET

    def capture_thermal_image(self):
        thermal_image = self.camera.grab().astype(np.float32)
        return thermal_image

    def find_hottest_region_center(self, thermal_image):
        thermal_image_celsius = self.raw_to_celsius(thermal_image)
        # thermal_image_celsius_clipped = np.clip(thermal_image_celsius, self.min_temp, self.max_temp)

        normalized_img = cv2.normalize(
            thermal_image_celsius, None, 0, 255, cv2.NORM_MINMAX
        )
        normalized_img = np.uint8(normalized_img)
        color_mapped_img = cv2.applyColorMap(normalized_img, cv2.COLORMAP_JET)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(thermal_image_celsius)

        center_x, center_y = (
            color_mapped_img.shape[1] // 2,
            color_mapped_img.shape[0] // 2,
        )

        relative_max_loc = (max_loc[0] - center_x, center_y - max_loc[1])  # (x, y)

        return color_mapped_img, relative_max_loc, max_val

    def display_live_feed(self):
        print("Thermal camera live feed started")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            "images/thermal_camera_output.avi", fourcc, 20.0, (1080, 720)
        )
        while True:
            thermal_image = self.capture_thermal_image()
            color_mapped_img, self.hottest_point, self.max_temperature = (
                self.find_hottest_region_center(thermal_image)
            )

            if self.max_temperature > 140:
                self.fire_status = True
            else:
                self.fire_status = False

            print(
                f"Hottest Point: {self.hottest_point}, Temperature: {self.max_temperature:.2f}°C"
            )
            color_mapped_img = cv2.resize(
                color_mapped_img, (1080, 720), interpolation=cv2.INTER_LINEAR
            )
            out.write(color_mapped_img)
            cv2.imshow("color_mapped_img", color_mapped_img)
            """
            # Stream the thermal image using GStreamer
            pipeline = (
                'appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc speed-preset=ultrafast tune=zerolatency ! rtph264pay ! udpsink host=YOUR_LAPTOP_IP port=5001'
            )
            out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, 20, (color_mapped_img.shape[1], color_mapped_img.shape[0]), True)
            out.write(color_mapped_img)
            """
            if cv2.waitKey(1) & 0xFF == 27:
                break
        self.camera.close()


th = ThermalCamera()
th.display_live_feed()
