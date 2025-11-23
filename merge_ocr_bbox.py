from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def array_to_list(array):
    if isinstance(array, np.ndarray):
        return [array_to_list(i) for i in array.tolist()]

    if isinstance(array, (list, tuple)):
        return [array_to_list(i) for i in array]

    return array

def get_polygons_projection(
    polygon: np.ndarray, 
    normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    #dots: np.ndarray = np.dot(polygon, normal)
    #return np.min(dots), np.max(dots)
    dots = [np.dot(normal, p) for p in polygon]
    return min(dots), max(dots)

def polygons_overlap(
    poly_1: np.ndarray, 
    poly_2: np.ndarray,
    threshold: float = 0
):
    gap_list: list[np.ndarray] = []

    for polygon in [poly_1, poly_2]:
        for index in range(len(polygon)):
            point_1 = polygon[index]
            point_2 = polygon[(index + 1) % len(polygon)]
            edge = point_2 - point_1
            axis = np.array([-edge[1], edge[0]])
            normal = axis / np.linalg.norm(axis)
        
            min_point_1, max_point_1 = get_polygons_projection(poly_1, normal)
            min_point_2, max_point_2 = get_polygons_projection(poly_2, normal)

            gap_list.extend([
                max_point_1 - min_point_2,
                max_point_2 - min_point_1
            ])
    
    min_gap = min(gap_list)

    return min_gap >= threshold, min_gap

def orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def on_segment(a, b, c):
    return (
        min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    )

def segments_intersect(A, B, C, D):
    o1 = orientation(A, B, C)
    o2 = orientation(A, B, D)
    o3 = orientation(C, D, A)
    o4 = orientation(C, D, B)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if o1 == 0 and on_segment(A, B, C): return True
    if o2 == 0 and on_segment(A, B, D): return True
    if o3 == 0 and on_segment(C, D, A): return True
    if o4 == 0 and on_segment(C, D, B): return True

    return False

def is_point_inside(
    polygon: np.ndarray, 
    point: np.ndarray,
    threshold: float = 0
):
    cross_list = []

    for index in range(len(polygon)):
        point_1 = polygon[index]
        point_2 = polygon[(index + 1) % len(polygon)]

        point_1_to_3 = point_1 - point
        point_2_to_3 = point_2 - point

        cross = point_1_to_3[0] * point_2_to_3[1] - point_1_to_3[1] * point_2_to_3[0]
        cross_list.append(cross)

    return all(cross >= threshold for cross in cross_list) or all(cross <= threshold for cross in cross_list)

def point_to_segment_distance(point_a, point_b, point_o):
    line_ab = point_b - point_a
    line_ao = point_o - point_a

    t = np.dot(line_ao, line_ab) / np.dot(line_ab, line_ab)

    if t < 0:
        return np.linalg.norm(point_o - point_a)

    if t > 1:
        return np.linalg.norm(point_o - point_b)

    min_distance = point_a + t * line_ab

    return np.linalg.norm(point_o - min_distance)

def line_intersection(point_a, point_b, point_c, point_d):
    x1, y1 = point_a
    x2, y2 = point_b
    x3, y3 = point_c
    x4, y4 = point_d

    vector_x_1 = x2 - x1
    vector_y_1 = y2 - y1
    vector_x_2 = x4 - x3
    vector_y_2 = y4 - y3

    denominator = vector_x_1 * vector_y_2 - vector_y_1 * vector_x_2

    t = ((x3 - x1) * vector_y_2 - (y3 - y1) * vector_x_2) / denominator

    intersection_x = x1 + t * vector_x_1
    intersection_y = y1 + t * vector_y_1
    
    return (intersection_x, intersection_y)

def get_merge_bbox(*polygons, image_size: int):
    all_xy_point: np.ndarray = np.concatenate(polygons)

    distance_0_0 = [
        (abs(np.linalg.norm(xy_point - np.array([0, 0]))), xy_point) 
        for xy_point in all_xy_point
    ]
    distance_0_max = [
        (abs(np.linalg.norm(xy_point - np.array([0, image_size]))), xy_point) 
        for xy_point in all_xy_point
    ]
    distance_max_max = [
        (abs(np.linalg.norm(xy_point - np.array([image_size, image_size]))), xy_point) 
        for xy_point in all_xy_point
    ]
    distance_max_0 = [
        (abs(np.linalg.norm(xy_point - np.array([image_size, 0]))), xy_point) 
        for xy_point in all_xy_point
    ]

    _, closest_0_0 = min(distance_0_0)
    _, closest_0_max = min(distance_0_max)
    _, closest_max_max = min(distance_max_max)
    _, closest_max_0 = min(distance_max_0)

    new_bbox = np.concatenate((
        [closest_0_0],
        [closest_max_0],
        [closest_max_max],
        [closest_0_max] 
    ))

    all_xy_point_sort = []
    for point_index, point in enumerate(all_xy_point):
        distance = []

        for index in range(len(new_bbox)):
            to_segment_distance = point_to_segment_distance(
                new_bbox[index], 
                new_bbox[(index + 1) % len(new_bbox)], 
                point
            )
            distance.append(to_segment_distance)
        
        min_distance = min(distance)

        all_xy_point_sort.append(
            (min_distance, point_index, new_bbox[index], new_bbox[(index + 1) % len(new_bbox)])
        )

    all_xy_point_sort.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    for _, point, _, _ in all_xy_point_sort:
        if is_point_inside(new_bbox, point, threshold=0):
            continue

        distance_from_point = [
            (np.linalg.norm(box_point - point), index) 
            for index, box_point in enumerate(new_bbox)
        ]
        _, min_distance_point_index = min(distance_from_point)
        
        point_a = new_bbox[min_distance_point_index]
        point_b = new_bbox[min_distance_point_index - 1]
        point_c = new_bbox[(min_distance_point_index + 1) % len(new_bbox)]

        if segments_intersect(point_a, point_b, point, point_c):
            new_xy = line_intersection(point_a, point_c, point, point_b)
        else:#if segments_intersect(point_a, point_c, point, point_b):
            new_xy = line_intersection(point_a, point_b, point, point_c)

        new_bbox[min_distance_point_index] = np.array(new_xy)

    return new_bbox


if __name__ == "__main__":
    MANGA_OCR_DATA_PATH = "/media/ifw/GameFile/linux_cache/data_unprocessed/manga_image_ocr.parquet"

    MIN_THRESHOLD = 0


    dataset = pd.read_parquet(MANGA_OCR_DATA_PATH)

    for data in tqdm(dataset.itertuples(), total=len(dataset), desc="duplicated"):
        ocr_bboxes: np.ndarray = data.rec_polys

        overlap_dict = defaultdict(list)
        for index, ocr_bboxe in enumerate(ocr_bboxes):
            for remaining_index in range(index + 1, len(ocr_bboxes)):
                is_overlap, _ = polygons_overlap(
                    poly_1=ocr_bboxe,
                    poly_2=ocr_bboxes[remaining_index],
                    threshold=MIN_THRESHOLD
                )

                if is_overlap:
                    overlap_dict[index].append(remaining_index)

        overlap_grouping: list[set[int]] = []
        for key, values in overlap_dict.items():
            for group in overlap_grouping:
                if key in group:
                    group.update(values)
                    break
                
                values_in_group = [value for value in values if value in group]

                if values_in_group:
                    group.add(key)
                    group.update(values)
                    break
            else:
                overlap_grouping.append(set((key,)))
                overlap_grouping[-1].update(values)
        
        overlaps = list(range(len(ocr_bboxes)))
        overlap_not_in_grouping =[]
        for overlap in overlaps:
            in_grouping = False

            for group in overlap_grouping:
                if overlap in group:
                    in_grouping = True
                    break
            
            if not in_grouping:
                overlap_not_in_grouping.append(overlap)

        break
