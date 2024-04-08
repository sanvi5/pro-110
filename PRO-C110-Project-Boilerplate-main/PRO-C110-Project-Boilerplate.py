import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow model
model = tf.keras.models.load_model('PRO-C110-Project-Boilerplate-main/keras/keras_model.h5')

# Function to preprocess the frame before feeding it to the model
def preprocess_frame(frame):
    # Resize the frame to match the input shape of the model
    resized_frame = cv2.resize(frame, (224, 224))
    # Normalize the frame (assuming your model expects input in the range [0,1])
    normalized_frame = resized_frame / 255.0  # Assuming frame is in range [0, 255]
    # Expand the dimensions to match the input shape of the model
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:
    # Reading / Requesting a Frame from the Camera 
    status, frame = camera.read()

    # If we were successfully able to read the frame
    if status:
        # Flip the frame
        frame = cv2.flip(frame, 1)

        # Preprocess the frame
        input_frame = preprocess_frame(frame)

        # Get predictions from the model
        predictions = model.predict(input_frame)

        # Displaying the frames captured
        cv2.imshow('feed', frame)

        # Waiting for 1ms
        code = cv2.waitKey(1)
        
        # If space key is pressed, break the loop
        if code == 32:
            break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()
