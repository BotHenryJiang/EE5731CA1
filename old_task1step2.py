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
