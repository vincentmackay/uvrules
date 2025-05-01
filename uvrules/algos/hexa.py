import numpy as np

def create_hexa(n, diameter=10.0):
    """
    Generate a hexagonal grid with n points, starting at [0,0] and spiraling outwards.
    
    Parameters:
    -----------
    n : int
        Number of points to generate
    diameter : float
        Diameter of hexagons (distance between opposite vertices)
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (n, 2) containing the x,y coordinates of the hexagonal grid points
    """
    # Calculate the distance between neighboring points
    # Distance between adjacent centers
    dx = diameter  # Horizontal distance
    dy = diameter * np.sqrt(3) / 2  # Vertical distance
    
    # Initialize the result array
    result = np.zeros((n, 2))
    
    if n <= 0:
        return result
    
    # First point is at origin
    if n == 1:
        return result
    
    # Directions for spiraling: right, up-right, up-left, left, down-left, down-right
    directions = np.array([
        [-dx/2, dy],       # Up-left
        [-dx, 0],          # Left
        [-dx/2, -dy],      # Down-left
        [dx/2, -dy],       # Down-right
        [dx, 0],           # Right
        [dx/2, dy],        # Up-right
    ])
    
    current_pos = np.array([0.0, 0.0])
    current_idx = 1  # We already placed the origin
    # Spiral outward layer by layer
    layer = 1
    while current_idx < n:
        # Start each layer by going right
        current_pos += [dx,0]
        result[current_idx] = current_pos
        current_idx += 1
        if current_idx >= n:
            break
            
        # Then follow each side of the hexagon
        if layer == 1:
            n_sides = 5
        else:
            n_sides = 6
        for side in range(n_sides):
            # Each side length increases with the layer number
            
            for _ in range(layer - 1*(side==5)):
                current_pos += directions[side]
                result[current_idx] = current_pos
                current_idx += 1
                if current_idx >= n:
                    break
            if current_idx >= n:
                break
            
        current_pos += directions[-1]
        
        layer += 1
    
    return result
