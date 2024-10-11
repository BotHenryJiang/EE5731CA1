#############################################################################
##                                                                         ##
##           TODO: CODE BLOCK FOR STEP 2 IS HERE                           ##
#############################################################################
from scipy import ndimage

def sobel_filters(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = ndimage.convolve(image, Kx)
    Iy = ndimage.convolve(image, Ky)
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return G, theta

def non_max_suppression(image, theta):
    M, N = image.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 255
            r = 255
            
            #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = image[i, j+1]
                r = image[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = image[i+1, j-1]
                r = image[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = image[i+1, j]
                r = image[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = image[i-1, j-1]
                r = image[i+1, j+1]

            if (image[i,j] >= q) and (image[i,j] >= r):
                Z[i,j] = image[i,j]
            else:
                Z[i,j] = 0
    
    return Z

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    # Step 1: Noise reduction
    blurred = ndimage.gaussian_filter(image, sigma=1.4)
    
    # Step 2: Gradient calculation
    gradient_magnitude, gradient_direction = sobel_filters(blurred)
    
    # Step 3: Non-maximum suppression
    suppressed = non_max_suppression(gradient_magnitude, gradient_direction)
    
    # Step 4: Double thresholding
    suppressed_uint8 = suppressed.astype(np.uint8)
    _, strong_edges = cv2.threshold(suppressed_uint8, high_threshold, 255, cv2.THRESH_BINARY)
    _, weak_edges = cv2.threshold(suppressed_uint8, low_threshold, 255, cv2.THRESH_BINARY)
    
    thresholded = cv2.bitwise_and(strong_edges, weak_edges)
    
    # Step 5: Edge tracking by hysteresis
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final = cv2.dilate(thresholded, kernel)
    final = cv2.bitwise_and(final, weak_edges)
    
    return blurred, gradient_magnitude, suppressed, thresholded, final

# Apply Canny edge detection to both images
dark_steps = canny_edge_detection(dark_1_gray)
bright_steps = canny_edge_detection(bright_1_gray)

# Plot results
fig, axes = plt.subplots(2, 6, figsize=(24, 8))  # Changed from 5 to 6 columns
titles = ['Original', 'Blurred', 'Gradient', 'Suppressed', 'Thresholded', 'Final']

for i, (dark_step, bright_step) in enumerate(zip([dark_1_gray] + list(dark_steps), [bright_1_gray] + list(bright_steps))):
    axes[0, i].imshow(dark_step, cmap='gray')
    axes[0, i].set_title(f'Dark: {titles[i]}')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(bright_step, cmap='gray')
    axes[1, i].set_title(f'Bright: {titles[i]}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()





#############################################################################
##                                                                         ##
##           TODO: CODE BLOCK FOR STEP 3 IS HERE                           ##
#############################################################################
def draw_epipolar_lines(img1, img2, F, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        F - fundamental matrix
        pts1, pts2 - corresponding points in both images '''
    r, c, _ = img1.shape
    for pt1, pt2 in zip(pts1, pts2):
        # Compute the epipolar line in the second image
        line1 = F @ pt1
        # Compute the epipolar line in the first image
        line2 = F.T @ pt2

        # Draw the epipolar line on the second image
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line1[2] / line1[1]])
        x1, y1 = map(int, [c, -(line1[2] + line1[0] * c) / line1[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, thickness=5)
        img2 = cv2.circle(img2, tuple(pt2[:2]), 5, color, -1)

        # Draw the epipolar line on the first image
        x0, y0 = map(int, [0, -line2[2] / line2[1]])
        x1, y1 = map(int, [c, -(line2[2] + line2[0] * c) / line2[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, thickness=5)
        img1 = cv2.circle(img1, tuple(pt1[:2]), 5, color, -1)

    return img1, img2

# Draw epipolar lines on both images
img1_with_lines, img2_with_lines = draw_epipolar_lines(bright_1_rgb.copy(), bright_2_rgb.copy(), F, keypoints_1, keypoints_2)

# Display the images with epipolar lines
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img1_with_lines)
plt.title('Epipolar Lines on Bright-1')
plt.axis('off')

plt.subplot(122)
plt.imshow(img2_with_lines)
plt.title('Epipolar Lines on Bright-2')
plt.axis('off')

plt.tight_layout()
plt.show()





import numpy as np

def construct_W(points1, points2):
    """
    Construct linear system W for the epipolar constraint
    """
    W = []
    for (x, y, _), (x_prime, y_prime, _) in zip(points1, points2):
        W.append([
            x_prime*x, x_prime*y, x_prime,
            y_prime*x, y_prime*y, y_prime,
            x, y, 1
        ])
    return np.array(W)

def compute_fundamental_matrix(points1, points2, T1, T2):
    # Ensure points are in homogeneous coordinates
    assert points1.shape[1] == 3 and points2.shape[1] == 3, "Points must be in homogeneous coordinates"

    # Construct matrix W for the equation Wf = 0
    W = construct_W(points1, points2)

    # Perform SVD on W
    _, _, Vt = np.linalg.svd(W)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank 2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    # Denormalize the fundamental matrix
    F = T2.T @ F @ T1

    # Normalize F to have unit norm
    F = F / np.max(np.abs(F))

    return F

# Compute the fundamental matrix using normalized keypoints
F = compute_fundamental_matrix(normalized_keypoints_1, normalized_keypoints_2, T1, T2)

# Print the fundamental matrix
print("Computed Fundamental Matrix F:")
print(F)



"""
keypoints_1 = np.array([
    [2983, 940,1],   # Point 1
    [362, 701,1],   # Point 2
    [2190,1140,1],
    [1245,2103,1],
    [1321, 166,1], # Point 5
    [2359,349,1],
    [667,1785,1],
    [3338, 1879,1]  # Point 8
])

keypoints_2 = np.array([
    [2844, 950,1],   # Point 1
    [773, 719,1],   # Point 2
    [1954,1149,1],
    [752,2010,1],
    [1816, 174,1], # Point 5
    [2519,384,1],
    [324,1727,1],
    [2080, 1946,1]  # Point 8
])
"""




import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_epipolar_lines(img1, img2, pts1, pts2, F, colors=None):
    """Draw epipolar lines on both images"""
    c = img1.shape[1]
    if colors is None:
        colors = np.random.randint(0, 255, (pts1.shape[0], 3))
    
    # Compute epipolar lines in the second image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    
    # Compute epipolar lines in the first image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    
    for r, pt1, pt2, color in zip(range(pts1.shape[0]), pts1, pts2, colors):
        # Draw epipolar line on the second image
        x0, y0 = map(int, [0, -lines2[r][2]/lines2[r][1]])
        x1, y1 = map(int, [c, -(lines2[r][2]+lines2[r][0]*c)/lines2[r][1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color.tolist(), 2)
        img2 = cv2.circle(img2, tuple(pt2), 5, color.tolist(), -1)
        
        # Draw epipolar line on the first image
        x0, y0 = map(int, [0, -lines1[r][2]/lines1[r][1]])
        x1, y1 = map(int, [c, -(lines1[r][2]+lines1[r][0]*c)/lines1[r][1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color.tolist(), 2)
        img1 = cv2.circle(img1, tuple(pt1), 5, color.tolist(), -1)
    
    return img1, img2

# Use the previously defined keypoints and fundamental matrix
pts1 = keypoints_1[:, :2].astype(int)
pts2 = keypoints_2[:, :2].astype(int)

# Draw epipolar lines
img1_with_lines, img2_with_lines = draw_epipolar_lines(bright_1_rgb.copy(), bright_2_rgb.copy(), pts1, pts2, F)

# Display the images with epipolar lines
plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Epipolar Lines on Bright-1')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Epipolar Lines on Bright-2')
plt.axis('off')

plt.tight_layout()
plt.show()



keypoints_1 = np.array([
    [937, 599, 1],
    [153, 381, 1],
    [2184, 1127, 1],
    [1321, 166, 1],
    [3342, 1888, 1],
    #[2983, 940, 1],
    [3147, 1788, 1],
    [2859, 1167, 1], 
    [2308, 1275, 1],
    [1422, 598, 1],
    [2312, 1152, 1],
    [3097, 2546, 1],
    [2906, 445, 1]
])

keypoints_2 = np.array([
    [883, 641, 1],
    [584, 440, 1],
    [1943, 1137, 1],
    [1816, 174, 1],
    [2078, 1951, 1],
    #[2844, 950, 1],
    [2010, 1838, 1],
    [2671, 1175, 1],
    [2035, 1285, 1],
    [1926, 594, 1],
    [2113, 1165, 1],
    [1996, 2591, 1],
    [3527, 386, 1]
])
"""
keypoints_1 = np.array([
   [937, 599, 1],
    [153, 381, 1],
    [2184, 1127, 1],
    [1245,2103, 1],
    [1321, 166, 1], # Point 5
    [2359,349, 1],
    [667,1785, 1],
    [3338, 1879, 1],  # Point 8
    [351, 366, 1],   # Point 1
    [2005, 1164, 1], # Point 3
    [2322, 1207, 1], # Point 4
    [2771, 1073, 1], # Point 5
    [3114, 1090, 1], # Point 6
    [3155, 1785, 1], # Point 7
    [1321, 166, 1],
    [3342, 1888, 1],
    [2983, 940, 1],
    [3147, 1788, 1],
    [2859, 1167, 1], 
])

keypoints_2 = np.array([
     [883, 641, 1],
    [584, 440, 1],
    [1943, 1137, 1],
    [752,2010, 1],
    [1816, 174, 1], # Point 5
    [2519,384, 1],
    [324,1727, 1],
    [2080, 1946, 1],  # Point 8
    [763, 418, 1],   # Point 1
    [1826, 1174, 1], # Point 3
    [2045, 1234, 1], # Point 4
    [2660, 1084, 1], # Point 5
    [2941, 1105, 1], # Point 6
    [2009, 1835, 1], # Point 7
    [1816, 174, 1],
    [2078, 1951, 1],
    [2844, 950, 1],
    [2010, 1838, 1],
    [2671, 1175, 1]
])"""




def find_corresponding_keypoint(img1, img2, keypoint1, F):
    # Compute the epipolar line in img2 for the keypoint in img1
    epipolar_line = F @ keypoint1

    # Extract line coefficients
    a, b, c = epipolar_line

    # Define a range for searching along the line
    height, width = img2.shape[:2]
    best_match = None
    best_score = float('inf')

    # Iterate over the width of the image to find the best match
    for x in range(width):
        # Calculate the corresponding y coordinate on the line
        y = int(-(a * x + c) / b)
        
        # Ensure y is within image bounds
        if 0 <= y < height:
            # Compute a similarity score (e.g., intensity difference)
            score = np.abs(int(img1[keypoint1[1], keypoint1[0]]) - int(img2[y, x]))
            
            # Update the best match if the current score is better
            if score < best_score:
                best_score = score
                best_match = (x, y)

    return best_match

# Example usage
new_keypoint_1 = np.array([1000, 1500, 1])  # Example keypoint in homogeneous coordinates
corresponding_keypoint = find_corresponding_keypoint(bright_1, bright_2, new_keypoint_1, F)

print("Corresponding keypoint in Bright-2:", corresponding_keypoint)
# New keypoint on Bright-1
new_keypoint_1 = np.array([1000, 1500, 1])  # Example coordinates, you can choose any point

# Compute the epipolar line in Bright-2 for the new keypoint
new_epipolar_line = F @ new_keypoint_1

# Draw the new epipolar line on Bright-2
r, c = bright_2.shape  # 只解包两个值：高度和宽度
x0, y0 = map(int, [0, -new_epipolar_line[2] / new_epipolar_line[1]])
x1, y1 = map(int, [c, -(new_epipolar_line[2] + new_epipolar_line[0] * c) / new_epipolar_line[1]])
bright_2_with_new_line = cv2.line(bright_2.copy(), (x0, y0), (x1, y1), (255, 0, 0), 2)  # Red line for new epipolar line
bright_2_with_new_line = cv2.circle(bright_2_with_new_line, tuple(new_keypoint_1[:2]), 5, (255, 0, 0), -1)

# Display the image with the new epipolar line
plt.figure(figsize=(6, 6))
plt.imshow(bright_2_with_new_line, cmap='gray')  # 确保使用灰度色图
plt.title('New Epipolar Line on Bright-2')
plt.axis('off')
plt.show()