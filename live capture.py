from pypylon import pylon
import cv2
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('model.pt')

# Connect to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Configure camera settings (resolution, etc.) if needed
camera.MaxNumBuffer = 2  # Example setting, adjust as per your camera specs

# Start grabbing frames continuously
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


# Function to retrieve frames from camera
def get_frame_from_camera():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data and convert it to OpenCV format (numpy array)
        image = grabResult.Array
        return image


# Main loop for real-time object detection
while camera.IsGrabbing():
    # Get frame from camera
    frame = get_frame_from_camera()

    if frame is not None:
        # Perform object detection
        results = model(frame)

        # Render bounding boxes on the frame
        annotated_frame = results.render()

        # Convert from RGB (ultralytics) to BGR (OpenCV) for displaying
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Display the frame with bounding boxes
        cv2.imshow('Object Detection', annotated_frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Clean up resources
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
