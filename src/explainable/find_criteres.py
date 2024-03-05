import cv2
import numpy as np
from typing import Callable
from scipy.ndimage import center_of_mass as calculate_center_of_mass


class Critieres:
    def __init__(self) -> None:

        self.crits: dict[str, Callable] = {
            'ovality': calculate_ovality,
            'satellites': count_satellites,
            'irregularity': calculate_irregularity,
            'ratio': calculate_satellite_ratio,
            'homogeneity': calculate_homogeneity,
            'striation': count_internal_striations,
            'distrib': classify_distribution,
        }
    
    def get_critieres_name(self) -> list[str]:
        return self.crits.keys()
    
    def __len__(self) -> int:
        return len(self.crits)
    
    def get_critieres(self, mask: np.ndarray) -> list[float]:
        output = np.zeros(len(self))
        for i, function in enumerate(self.crits.values()):
            output[i] = function[mask]
        return output


def generate_random_mask(size: int,
                         num_ellipses: int,
                         diversity: float=1.0
                         ) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)

    for _ in range(num_ellipses):
        # Generate random center, size and angle for the ellipse
        center = tuple(np.random.normal(loc=size // 2, scale=(size // 4) * diversity, size=2).astype(int))
        axes = tuple(np.random.normal(loc=size // 6, scale=(size // 8) * diversity, size=2).astype(int))
        angle = np.random.uniform(0, 360)

        # Ensure the generated parameters are within the valid range
        center = tuple(max(min(c, size - 1), 0) for c in center)
        axes = tuple(max(min(a, size - 1), 0) for a in axes)

        # Draw the ellipse on the mask
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

    return mask


def calculate_ovality(mask):
    """
    Calculates the ovality of a given mask.

    Parameters:
    - mask: A binary image mask.

    Returns:
    - ovality: The ovality value, which is the ratio of the maximum and minimum ellipse diameters.
    """
    mask = mask.astype(np.uint8)
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_stain = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(main_stain)
        ovality = max(ellipse[1]) / min(ellipse[1])
    except:
        ovality = 0
    return ovality


def classify_distribution(segmentation_mask):
    # Convert the image to a numpy array
    img = np.array(segmentation_mask)

    # If the image is not binary, convert it to grayscale and threshold it
    if np.unique(img).size > 2:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, blood_pixels = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    else:
        blood_pixels = img

    # Convert the binary mask to boolean
    blood_pixels = blood_pixels.astype(bool)

    # Calculate the center of mass of the blood stain
    center_of_mass = calculate_center_of_mass(blood_pixels)

    # Calculate the distance of each blood pixel to the center of mass
    y_indices, x_indices = np.indices(blood_pixels.shape)
    distances = np.sqrt((x_indices - center_of_mass[0])**2 + (y_indices - center_of_mass[1])**2)
    
    # Calculate the mean and standard deviation of the distances
    mean_distance = distances[blood_pixels].mean()
    std_distance = distances[blood_pixels].std()

    # Classify the distribution based on the mean and standard deviation
    if std_distance < mean_distance * 0.5:
        return 0  # Central distribution
    elif std_distance < mean_distance:
        return 1  # Linear distribution
    elif std_distance < mean_distance * 1.5:
        return 2  # Curvilinear distribution
    else:
        return 3  # Burst distribution
    

def calculate_homogeneity(mask):
    """
    Calculates the homogeneity of the distribution of millimetric traces around the centimetric trace.

    Parameters:
    mask (numpy.ndarray): Binary mask representing the stained regions.

    Returns:
    float: Homogeneity of the distribution of millimetric traces.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 3:  # If there are no satellites, return 0
        return 0

    # The first label is the background, so the main stain is the second label
    main_stain_centroid = centroids[1]

    # The rest of the labels are the satellites
    satellite_centroids = centroids[2:]

    # Calculate the distance from each satellite to the main stain
    distances = np.sqrt(np.sum((satellite_centroids - main_stain_centroid)**2, axis=1))

    # Calculate the variance of the distances
    variance = np.var(distances)

    # A lower variance means a more homogeneous distribution
    homogeneity = 1 / variance if variance != 0 else 0

    return homogeneity


def count_internal_striations(mask):
    """
    Counts the number of internal striations in the given mask.

    Parameters:
    mask (numpy.ndarray): Binary mask representing the stained regions.

    Returns:
    int: Number of internal striations.
    """
    try:
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # The hierarchy returned by cv2.findContours has the following form:
        # [Next, Previous, First_Child, Parent]
        # If the Parent is not -1, then the contour is internal
        internal_striations = sum(1 for contour, info in zip(contours, hierarchy[0]) if info[3] != -1)
        
        return internal_striations
    except:
        return 0


def count_satellites(mask):
    """
    Counts the number of satellite stains in the given mask.

    Parameters:
    mask (numpy.ndarray): Binary mask representing the stained regions.

    Returns:
    int: Number of satellite stains.
    """
    try:
        mask = mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        satellites = num_labels - 1  # Subtract 1 to exclude the main stain itself
    except:
        satellites = 0
    return satellites


def calculate_irregularity(mask):
    """
    Calculates the irregularity of a given mask.

    Parameters:
    - mask: A binary image mask.

    Returns:
    - irregularity: The irregularity value of the mask.
    """
    try:
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_stain = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(main_stain, True)
        area = cv2.contourArea(main_stain)
        perfect_circle_perimeter = 2 * np.pi * (area / np.pi)**0.5
        irregularity = perimeter - perfect_circle_perimeter
    except:
        irregularity = 0
    return irregularity


def calculate_satellite_ratio(mask):
    """
    Calculates the average ratio of the size of satellite stains to the size of the main stain.

    Parameters:
    mask (numpy.ndarray): Binary mask representing the stained regions.

    Returns:
    float: Average ratio of the size of satellite stains to the size of the main stain.
    """

    try:
        mask = mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels < 2:  # If there are no satellites, return 0
            return 0

        # The first label is the background, so the main stain is the second label
        main_stain_size = stats[1, cv2.CC_STAT_AREA]

        # The rest of the labels are the satellites
        satellite_sizes = stats[2:, cv2.CC_STAT_AREA]

        # Calculate the ratio for each satellite and return the average
        ratios = satellite_sizes / main_stain_size
    except:
        ratios = np.array([0])


    return ratios.mean()

if __name__=='__main__':
    mask = generate_random_mask(100, 10)
    print("shape:", mask.shape)
    print(f"Ovality: {calculate_ovality(mask)}")
    print(f"Classification: {classify_distribution(mask)}")
    print(f"Homogeneity: {calculate_homogeneity(mask)}")
    print(f"Internal striations: {count_internal_striations(mask)}")
    print(f"Satellites: {count_satellites(mask)}")
    print(f"Irregularity: {calculate_irregularity(mask)}")
    print(f"Satellite ratio: {calculate_satellite_ratio(mask)}")
