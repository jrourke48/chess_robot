from picamera2 import Picamera2

# Initialize camera
picam2 = Picamera2()

# Configure the camera (simple preview configuration)
config = picam2.create_preview_configuration()
picam2.configure(config)

# Start camera
picam2.start()

# Capture an image
picam2.capture_file("test.jpg")
print("Image captured successfully as test.jpg")

# Stop camera
picam2.stop()