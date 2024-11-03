import cv2
import numpy as np
import time


class LaserPatternDetector:
    def __init__(self, camera_index=1, delay=0.1):
        """
        Initializes the detector with a specified camera index and delay.

        Parameters:
            camera_index (int): Index of the camera (default 0).
            delay (float): Delay in seconds between each frame capture.
        """
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise Exception("Could not open camera.")
        self.delay = delay
        self.frames = []

    def capture_frames(self, n):
        """
        Captures a series of frames with a delay between each capture.

        Parameters:
            n (int): Number of frames to capture.
        """
        self.frames = []
        for i in range(n):
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Could not capture frame.")
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cv2.imshow("frame", self.frames[-1])
            cv2.waitKey(0)
            print(f"Captured frame {i + 1}/{n}")
            # time.sleep(self.delay)

    def detect_laser_pattern(self, pattern_function):
        """
        Detects a laser point that turns on and off in a specific pattern across frames.

        Parameters:
            pattern_function (function): Function that returns the expected on/off pattern as a list.
                                         For example, [1, 0, 1, 0] for on-off-on-off.

        Returns:
            List of (x, y) coordinates where the laser point was detected, or None if no match.
        """
        pattern = pattern_function()
        if len(pattern) > len(self.frames):
            raise ValueError("Pattern length exceeds the number of frames captured.")
        pattern = pattern[:len(self.frames)]  # Truncate pattern if longer than frames

        # use the cross_correlation function to find the pattern in the frames
        correlation_image = np.zeros_like(self.frames[0], dtype=np.float32)
        for pixel_i in range(len(self.frames[0])):
            for pixel_j in range(len(self.frames[0][0])):
                recorded_signal = [self.frames[t][pixel_i][pixel_j] for t in range(len(pattern))]
                correlation_image[pixel_i, pixel_j] = cross_correlation(np.array(recorded_signal), np.array(pattern))

        cv2.imshow("correlation_image", correlation_image)
        cv2.waitKey(0)

        # apply gaussian filter to the correlation image
        correlation_image = cv2.GaussianBlur(correlation_image, (7, 7), 0)
        cv2.imshow("correlation_image", correlation_image)

        # Find the xy coords of the max correlation
        max_corr = np.max(correlation_image)
        max_corr_idx = np.where(correlation_image == max_corr)
        return list(zip(max_corr_idx[0], max_corr_idx[1]))

    def release_camera(self):
        """
        Releases the camera resource.
        """
        self.camera.release()


def cross_correlation(x, y):
    eps = 0.0001
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    x_sq_diff = np.sum((x - x_mean) ** 2)
    y_sq_diff = np.sum((y - y_mean) ** 2)
    denominator = np.sqrt(x_sq_diff * y_sq_diff) + eps
    correlation = numerator / denominator
    return correlation


# Example usage:
if __name__ == "__main__":
    # Define a pattern function, e.g., laser is on at frames 1, 3, 5, off in between
    def pattern_function():
        return [0, 1, 0, 0, 1, 0, 0, 0]


    detector = LaserPatternDetector(delay=0.2)  # Adjust delay as needed
    try:
        detector.capture_frames(n=8)
        laser_spot = detector.detect_laser_pattern(pattern_function)
        if laser_spot:
            print(f"Laser detected at {laser_spot}")
            # circle the detected laser spot
            frame_with_spot = detector.frames[0].copy()
            # bgr it
            frame_with_spot = cv2.cvtColor(frame_with_spot, cv2.COLOR_GRAY2BGR)
            cv2.circle(frame_with_spot,( laser_spot[0][1], laser_spot[0][0] ), 10, (0, 0, 255), 4)
            cv2.imshow("Detected laser spot", frame_with_spot)
            cv2.waitKey(0)
        else:
            print("No laser pattern detected matching the specified pattern.")
    finally:
        detector.release_camera()
