# Usage
# python scan.py -i images/page.jpg
import argparse
import cv2
import imutils
from imutils.perspective import four_point_transform

# Get input image as argument
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to input image")

args = parser.parse_args()


# Load image, make a copy, resize image
# Keep aspect ratio
original_document = cv2.imread(args.image)
document = original_document.copy()
document = imutils.resize(document, height=500)
aspect_ratio = original_document.shape[0] / 500.0

# Convert image to gray scale
# Filter image
# Find edges
gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
blurry = cv2.GaussianBlur(gray, (5, 5), 0)  # 0 - sigma is calculated from kernel size
edge = cv2.Canny(blurry, 75, 200)

# Show image and edge detection
print("1 - Edge Detection")
cv2.imshow("Document", document)
cv2.imshow("Document Edge", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find image contours
# We assume the largest square in the image
# is a piece of paper to be scanned
contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# We approximate the contour to a
# shape with less points, it this
# case a square
document_contour = None
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    epsilon = perimeter * 0.02
    approximation = cv2.approxPolyDP(contour, epsilon, True)

    if len(approximation) == 4:
        document_contour = approximation
        break


cv2.drawContours(document, [document_contour], -1, (0, 0, 255), 2)

# Show image and edge detection
print("2 - Find Contours")
cv2.imshow("Document", document)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Construct bird eye view of the document
# by applying the four points transform
warped = four_point_transform(original_document, document_contour.reshape(4, 2) * aspect_ratio)

# Convert warped to gray scale and threshold it
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)

original_document = imutils.resize(original_document, height=800)
warped = imutils.resize(warped, height=800)

# Show image and warped
print("3 - Perspective Transform")
cv2.imshow("Document", original_document)
cv2.imshow("Scanned", warped)
cv2.waitKey(0)
