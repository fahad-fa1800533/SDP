# Import required libraries
import cv2
import numpy as np
import cam

# Load YOLOv3 Tiny configuration and weights
#net = cv2.dnn.readNet("yolov3-tiny.cfg", "yolov3-tiny.weights")
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Load COCO class labels (for reference)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set minimum confidence threshold and NMS threshold
confidence_threshold = 0.125
nms_threshold = 0.55


#take an image by camera
cam.capture()

# Load image
image = cv2.imread("capture.jpg")

# Obtain image dimensions
height, width, _ = image.shape

# Create a blob from the image and set it as the input to the network
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Perform forward pass and get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)

# Initialize lists to store bounding box coordinates, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

# Iterate over each detection from each output layer
for out in outs:
    for detection in out:
        # Extract class probabilities and find the index of the class with the highest confidence
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filter out weak detections below the confidence threshold
        if confidence > confidence_threshold:
            # Scale the bounding box coordinates to match the original image size
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Add bounding box coordinates, confidences, and class IDs to their respective lists
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to remove redundant overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

# Count the number of vehicles detected
vehicle_count = 0

# Draw bounding boxes and count vehicles
for i in indices:
    box = boxes[i[0]]
    x, y, w, h = box

    # Draw bounding box rectangle and label on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"{classes[class_ids[i[0]]]}: {confidences[i[0]]:.2f}"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Increment vehicle count
    if classes[class_ids[i[0]]] == 'car' or classes[class_ids[i[0]]] == 'motorbike' or classes[class_ids[i[0]]] == 'bus' or classes[class_ids[i[0]]] == 'truck':
      vehicle_count += 1
#determine the congestion case
if vehicle_count > 10:
    print("Traffic_light_stat = GREEN")
else:
    print("Traffic_light_stat = RED")


#Display the resulting image with bounding boxes and vehicle count
cv2.putText(image, f"Vehicle Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow('Traffic',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
