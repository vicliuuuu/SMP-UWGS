import numpy as np
from shapely.geometry import Polygon, box
from scipy.spatial import ConvexHull

def run_graham_scan(points, W, H, is_edge_partition=False):
    points = np.array(points)
    if len(points) < 3:
        min_x, min_y = np.min(points, axis=0) if len(points) > 0 else (0, 0)
        max_x, max_y = np.max(points, axis=0) if len(points) > 0 else (W, H)
        hull_points = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
        convex_hull_polygon = Polygon(hull_points)
    else:
        convex_hull_polygon = None
        try:
            if is_edge_partition:
                hull = ConvexHull(points, qhull_options='QJ')
            else:
                hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            convex_hull_polygon = Polygon(hull_points)
        except Exception as e:
            print(f'[GrahamScan] ConvexHull failed: {e}, using bounding box approximation')
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)
            hull_points = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
            convex_hull_polygon = Polygon(hull_points)
    if convex_hull_polygon is None or convex_hull_polygon.is_empty:
        print(f'[GrahamScan] Using image bounds as fallback')
        convex_hull_polygon = box(0, 0, W, H)
    image_bounds = box(0, 0, W, H)
    if is_edge_partition:
        expanded_bounds = box(-W * 0.1, -H * 0.1, W * 1.1, H * 1.1)
        expanded_intersection = convex_hull_polygon.intersection(expanded_bounds)
        if expanded_intersection.area > 0:
            actual_intersection = convex_hull_polygon.intersection(image_bounds)
            intersection_area = actual_intersection.area
            if intersection_area == 0 and expanded_intersection.area > 0:
                intersection_area = expanded_intersection.area * 0.3
        else:
            intersection_area = 0
    else:
        intersection = convex_hull_polygon.intersection(image_bounds)
        intersection_area = intersection.area
    image_area = W * H
    intersection_rate = intersection_area / image_area
    return {'intersection_area': intersection_area, 'image_area': image_area, 'intersection_rate': intersection_rate}
