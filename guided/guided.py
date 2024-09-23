import cv2
import numpy as np
import matplotlib.pyplot as plt

def guided_filter(I, p, r, eps):
    return cv2.ximgproc.guidedFilter(I, p, r, eps)
def guasian_filter(image):
    kernel_size = (15, 15)  # Adjust the kernel size as needed
    sigma_x = 0  # Standard deviation in X direction (if zero, it's calculated from kernel size)
    sigma_y = 0  # Standard deviation in Y direction (if zero, it's the same as sigma_x)
    return cv2.GaussianBlur(image, kernel_size, sigma_x, sigma_y)

# Load image
image = cv2.imread("output_image.png")
image2 = cv2.imread("pic2.jpg")
# image2 = guasian_filter(image2)
# cv2.imshow("guasian", image2)

print(image.shape)
desired_size = (260,193)
image = cv2.resize(image, desired_size)
image2 = cv2.resize(image2, desired_size)
print(image2.shape)
image2 = image2[:,:,:1]
image = image[:,:,:1]

print(image2.shape)
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# print(image2.shape)

# Convert to float32
image = np.float32(image) / 255.0
image2 = np.float32(image2) / 255.0

# cv2.imshow('image1',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('image2',image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Apply guided filter
rlist = [1, 2, 4, 8]

for r in rlist:
    guided = guided_filter(image2, image, r, eps=0.005**2)
    output = guided - np.squeeze(image)
    print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBbb")
    # Show result
    # cv2.imshow("Guided Filter Result", guided)
    output = abs(output*100)
    cv2.imshow("Guided Filter Result", guided)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

