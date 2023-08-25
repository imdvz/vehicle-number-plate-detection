import cv2
import numpy as np

# Set the frame width and height
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Set the minimum area required for a license plate to be detected
MIN_AREA = 500

# Load the classifier for detecting license plates
PLATE_CASCADE = cv2.CascadeClassifier(
    "C:/Users/anupa/Desktop/testing/haarcascade_russian_plate_number.xml")

# Open the default camera (id=0)
cap = cv2.VideoCapture(0)

# Set the frame width, height, and brightness
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)
cap.set(10, 150)

# Initialize a counter for the saved images
count = 0

# Loop until the user presses the 'q' key
while True:
    # Read a frame from the camera
    success, frame = cap.read()

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the grayscale image
    number_plates = PLATE_CASCADE.detectMultiScale(frame_gray, 1.1, 4)

    # Loop through each license plate found
    for (x, y, w, h) in number_plates:
        # Calculate the area of the license plate
        area = w * h
        # If the area is greater than the minimum area, draw a rectangle around the license plate and display a label
        if area > MIN_AREA:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "NumberPlate", (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            # Extract the region of interest (ROI) containing the license plate
            roi = frame[y:y + h, x:x + w]
            cv2.imshow("ROI", roi)

    # Display the result with license plates highlighted
    cv2.imshow("Result", frame)

    # If the 's' key is pressed, save the ROI as an image file
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("C:/Users/anupa/Desktop/testing/IMAGES" +
                    str(count) + ".jpg", roi)
        # Display a message indicating that the scan has been saved
        cv2.rectangle(frame, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, "Scan Saved", (15, 265),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", frame)
        cv2.waitKey(500)
        count += 1

    # If the 'q' key is pressed, exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
