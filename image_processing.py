import cv2
import numpy as np


class LaserPatternDetector:
    def __init__(self, camera_index=1, delay=0.1):
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise Exception("Could not open camera.")
        self.delay = delay
        self.frames = []

    def capture_frames(self, n):
        self.frames = []
        for i in range(n):
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Could not capture frame.")
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            print(f"Captured frame {i + 1}/{n}")
            # Optional: Display each captured frame (commented out for efficiency)
            cv2.imshow("frame", self.frames[-1])
            cv2.waitKey(0)

    def detect_laser_pattern(self, pattern_function):
        pattern = np.array(pattern_function(), dtype=np.float32)
        if len(pattern) > len(self.frames):
            raise ValueError("Pattern length exceeds the number of frames captured.")
        pattern = pattern[:len(self.frames)]

        # Stack frames for vectorized processing
        frames_stack = np.stack(self.frames, axis=0).astype(np.float32)  # Shape: (n_frames, height, width)
        frames_mean = np.mean(frames_stack, axis=0)
        pattern_mean = np.mean(pattern)

        # Compute numerator and denominator in one vectorized step
        recorded_signals = frames_stack - frames_mean  # Center frames at each pixel
        pattern_centered = pattern - pattern_mean
        numerator = np.tensordot(recorded_signals, pattern_centered, axes=(0, 0))  # Cross-correlation for each pixel

        # Compute the variance for denominator
        frames_var = np.sqrt(np.sum(recorded_signals ** 2, axis=0) + 1e-4)
        pattern_var = np.sqrt(np.sum(pattern_centered ** 2) + 1e-4)
        correlation_image = numerator / (frames_var * pattern_var)  # Element-wise division for correlation

        # Apply Gaussian blur for smoothing
        correlation_image = cv2.GaussianBlur(correlation_image, (7, 7), 0)

        # Find coordinates of the max correlation
        max_corr_idx = np.unravel_index(np.argmax(correlation_image), correlation_image.shape)
        return [max_corr_idx] if np.max(correlation_image) > 0 else None

    def release_camera(self):
        self.camera.release()


def cross_correlation(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2)) + 1e-4
    return numerator / denominator


# Example usage:
if __name__ == "__main__":
    def pattern_function():
        return [0, 1, 0, 0, 1, 0, 0, 1, 1]

    detector = LaserPatternDetector(delay=0.2)
    try:
        detector.capture_frames(n=9)
        laser_spot = detector.detect_laser_pattern(pattern_function)
        if laser_spot:
            print(f"Laser detected at {laser_spot}")
            # Visualize the detected laser spot
            frame_with_spot = cv2.cvtColor(detector.frames[0].copy(), cv2.COLOR_GRAY2BGR)
            cv2.circle(frame_with_spot, (laser_spot[0][1], laser_spot[0][0]), 10, (0, 0, 255), 4)
            cv2.imshow("Detected laser spot", frame_with_spot)
            cv2.waitKey(0)
        else:
            print("No laser pattern detected matching the specified pattern.")
    finally:
        detector.release_camera()
