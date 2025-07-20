# code to compute morphometric measures
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import nibabel as nib
import pickle 
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize,remove_small_objects
from skimage.measure import label
import networkx as nx
from scipy.spatial.distance import euclidean
import math
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import graycoprops, local_binary_pattern
from collections import Counter

def load_nii(nii_file) :
    'Loads a .nii.gz or .nii file into a np.array'
    img = nib.load(nii_file)
    vol = np.asarray(img.get_fdata())
    return vol

def calculate_angle_between_vectors(vec1, vec2):
    """
    Calculates the angle in radians between two vectors.
    """
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0 # Or raise an error, depending on desired behavior for zero-length vectors

    dot_product = np.dot(vec1, vec2)
    cosine_angle = np.clip(dot_product / (norm_vec1 * norm_vec2), -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return angle_rad

def compute_morphometrics(volume, print_outputs = False, voxel_spacing = [1,1,1], visual_show = False):
    # input: 3D array of vessel structure (assume positive values inside vessels); if print_outputs true then prints these measures; 
    # alter voxel_spacing/volume as appropriate; plot 3D vessel network if visual_show is True
    # output: measures in the form of a dictionary as follows:
    # - **Quantitative "Amount" of Vessels:**
    #     - **Total Vessel Volume (BV) / Vascular Volume per Tissue/Marrow Volume (Ves.V/TV or Ves.V/Mar.V)**
    #     - **Vessel Number per unit area/volume (Ves.N/T.Ar or Ves.N/Mar.Ar)**
    # - **Quantitative "Quality" of Vessels:**
    #     - **Mean Vessel Diameter / Width Distribution**
    #     - **Mean Vessel Area (MVA) (Luminal Distension)**
    #     - **Vessel Tortuosity (SOAM)**
    #     - **Vessel Curvature (CLR)**
    # - **Quantitative "Network Architecture":**
    #     - **Total Vessel Lengths**
    #     - **Number of Branches / Junctions (Total Junctions, Triple/Quadruple Junctions)**
    # - **Quantitative "Spatial Relationships":**
    #     - **Distance between Blood Vessels.**
    
    # Preprocessing
    vessel_mask = volume >= 0  # Boolean mask of vessels
    voxel_spacing_x = voxel_spacing[0]
    voxel_spacing_y = voxel_spacing[1]
    voxel_spacing_z = voxel_spacing[2]
    voxel_volume = voxel_spacing_x*voxel_spacing_y*voxel_spacing_z
    
    n_x = volume.shape[0]
    n_y = volume.shape[1]
    n_z = volume.shape[2]
    
    # set up dictionary
    morphometric_dict = {}
   
    # Total Vessel Volume / Vascular Volume fraction
    total_voxels = vessel_mask.size
    vessel_voxels = vessel_mask.sum()
    vessel_volume = vessel_voxels * voxel_volume
    tissue_volume = total_voxels * voxel_volume

    vascular_volume_fraction = vessel_volume / tissue_volume
    morphometric_dict["vessel_volume_fraction"] = vascular_volume_fraction
    
    # Construct vessel skeleton
    binary_vessels_cleaned = remove_small_objects(vessel_mask, min_size=2)
    skeleton = skeletonize(binary_vessels_cleaned)
    
    # Compute Euclidean distance inside vessel mask: radius to boundary
    distance_map = distance_transform_edt(binary_vessels_cleaned)
    diameters = 2 * distance_map[skeleton]
    mean_diameter = diameters.mean()
    morphometric_dict["mean_diameter"] = mean_diameter
    
    # Graph extraction from skeleton
    skel_coords = np.argwhere(skeleton)
    G_pixel = nx.Graph()
    for r, c, z in skel_coords:
        G_pixel.add_node((r, c, z), diameter=2 * distance_map[r, c, z])
    offsets = [(-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
            (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
            (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
            (0, -1, -1), (0, -1, 0), (0, -1, 1),
            (0, 0, -1),             (0, 0, 1),
            (0, 1, -1), (0, 1, 0), (0, 1, 1),
            (1, -1, -1), (1, -1, 0), (1, -1, 1),
            (1, 0, -1), (1, 0, 0), (1, 0, 1),
            (1, 1, -1), (1, 1, 0), (1, 1, 1)]
    for r, c, z in skel_coords:
        current_node = (r, c, z)
        for dr, dc, dz in offsets:
            nr, nc, nz = r + dr, c + dc, z + dz
            if 0 <= nr < volume.shape[0] and 0 <= nc < volume.shape[1] and 0 <= nz < volume.shape[2]:
                if skeleton[nr, nc, nz] == 1:
                    neighbor_node = (nr, nc, nz)
                    G_pixel.add_edge(current_node, neighbor_node)
                    
    # Simplify the graph 
    endpoints = [node for node, degree in G_pixel.degree() if degree == 1]
    branch_points = [node for node, degree in G_pixel.degree() if degree > 2]

    # Store paths originating from branch points for later angle calculation
    # This will map a branch point to a list of its outgoing segment paths
    branch_point_segment_paths = {bp: [] for bp in branch_points}

    G_topo = nx.Graph()
    for bp in branch_points:
        G_topo.add_node(bp, type='branch_point')
    for ep in endpoints:
        G_topo.add_node(ep, type='endpoint')

    visited_pixels = set()
    for bp in branch_points:
        visited_pixels.add(bp)
    for ep in endpoints:
        visited_pixels.add(ep)

    for start_node in list(branch_points) + list(endpoints):
        for neighbor in G_pixel.neighbors(start_node):
            if neighbor not in visited_pixels:
                path = [start_node, neighbor]
                current_node = neighbor
                visited_pixels.add(current_node)

                while G_pixel.degree(current_node) == 2:
                    next_node = None
                    for n_neighbor in G_pixel.neighbors(current_node):
                        if n_neighbor not in visited_pixels:
                            next_node = n_neighbor
                            break
                    if next_node:
                        path.append(next_node)
                        visited_pixels.add(next_node)
                        current_node = next_node
                    else:
                        break

                end_node = current_node
                if not G_topo.has_edge(start_node, end_node):
                    # Calculate properties for the segment (path)
                    segment_length_voxels = len(path) - 1
                    euclidean_distance = euclidean(start_node, end_node)
                    
                    # Avoid division by zero for curvature if start_node == end_node (rare for segments)
                    curvature = segment_length_voxels / euclidean_distance if euclidean_distance > 0 else 1.0

                    # Average diameter along the path
                    path_diameters = [G_pixel.nodes[node]['diameter'] for node in path]
                    mean_diameter = np.mean(path_diameters) if path_diameters else 0

                    # Store segment properties
                    G_topo.add_edge(start_node, end_node,
                                    path=path,
                                    length_voxels=segment_length_voxels,
                                    length_physical=segment_length_voxels * np.sqrt(voxel_spacing_x**2 + voxel_spacing_y**2 + voxel_spacing_z**2)/np.sqrt(3), # Approx. diagonal voxel length
                                    euclidean_distance=euclidean_distance,
                                    curvature=curvature,
                                    mean_diameter=mean_diameter)   
                    
                    # If this segment starts at a branch point, store it for SOAM calculation
                    if start_node in branch_points:
                        branch_point_segment_paths[start_node].append(path)
                    # If this segment ends at a branch point (and wasn't the start_node), store it
                    if end_node in branch_points and start_node != end_node:
                        # We need the path to be oriented away from the branch point for vector calculation
                        # So, if end_node is the branch point, reverse the path.
                        branch_point_segment_paths[end_node].append(list(reversed(path)))             
        
    # Quantify Branching, Loops, and Segment Count
    num_branch_points = len(branch_points)
    num_endpoints = len(endpoints)
    num_segments = G_topo.number_of_edges()
    total_vessel_length_voxels = sum(data['length_voxels'] for u, v, data in G_topo.edges(data=True))
    
    morphometric_dict["num_branch_pts"] = num_branch_points
    morphometric_dict["num_end_pts"] = num_endpoints
    morphometric_dict["num_vessels"] = num_segments
    morphometric_dict["tot_vessel_length"] = total_vessel_length_voxels

    try:
        cycles = nx.cycle_basis(G_topo)
        num_loops = len(cycles)
    except nx.NetworkXNoCycle:
        num_loops = 0
    
    morphometric_dict["num_cycles"] = num_loops
    
    total_xy_area_sq_microns = (n_x * voxel_spacing_x) * (n_y * voxel_spacing_y)
    if total_xy_area_sq_microns > 0:
        vessel_segments_per_projected_area = num_segments / total_xy_area_sq_microns
        morphometric_dict["vessels_per_area"] = vessel_segments_per_projected_area
    
    total_physical_volume_cubic_microns = (n_x * voxel_spacing_x) * \
                                      (n_y * voxel_spacing_y) * \
                                      (n_z * voxel_spacing_z)
    if total_physical_volume_cubic_microns > 0:
        vessel_segments_per_unit_volume = num_segments / total_physical_volume_cubic_microns
        morphometric_dict["vessels_per_volume"] = vessel_segments_per_unit_volume
    
    # Visualize 3D vessel network
    if visual_show:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        skel_r, skel_c, skel_z = skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]
        ax.scatter(skel_c, skel_r, skel_z, c='gray', s=1, alpha=0.1, label='Skeleton')

        bp_c = [p[1] for p in branch_points]
        bp_r = [p[0] for p in branch_points]
        bp_z = [p[2] for p in branch_points]
        ax.scatter(bp_c, bp_r, bp_z, c='red', s=50, label='Branch Points')

        ep_c = [p[1] for p in endpoints]
        ep_r = [p[0] for p in endpoints]
        ep_z = [p[2] for p in endpoints]
        ax.scatter(ep_c, ep_r, ep_z, c='blue', s=30, label='Endpoints')

        for u, v, data in G_topo.edges(data=True):
            path_coords = np.array(data['path'])
            ax.plot(path_coords[:, 1], path_coords[:, 0], path_coords[:, 2], color='green', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Vessel Network Skeleton and Topological Graph')
        ax.legend()
        plt.show()
        
    # --- Quantitative "Quality" of Vessels ---

    # Mean Vessel Diameter / Width Distribution
    all_diameters = [G_pixel.nodes[node]['diameter'] for node in G_pixel.nodes()]
    mean_vessel_diameter = np.mean(all_diameters) if all_diameters else 0
    morphometric_dict["mean_vessel_diameter"] = mean_vessel_diameter

    # Mean Vessel Area (MVA) (Luminal Distension)
    # Assuming circular cross-section, Area = pi * (diameter/2)^2
    all_mean_segment_areas = [math.pi * (data['mean_diameter']/2)**2 for u,v,data in G_topo.edges(data=True) if data['mean_diameter'] > 0]
    mean_vessel_area = np.mean(all_mean_segment_areas) if all_mean_segment_areas else 0
    morphometric_dict["mean_vessel_area"] = mean_vessel_area

    # Vessel Curvature
    all_curvatures = [data['curvature'] for u,v,data in G_topo.edges(data=True)]
    mean_curvature = np.mean(all_curvatures) if all_curvatures else 0
    morphometric_dict["mean_curvature"] = mean_curvature
        
    # Classify junctions by degree in the pixel graph (degree in G_topo is always 3+)
    triple_junctions = [bp for bp in branch_points if G_pixel.degree(bp) == 3]
    quadruple_junctions = [bp for bp in branch_points if G_pixel.degree(bp) == 4]
    morphometric_dict["num_triple_junctions"] = len(triple_junctions)
    morphometric_dict["num_quadruple_junctions"] = len(quadruple_junctions)
    
    # Vessel Directionality (Angle relative to Z-axis)
    segment_angles_z = []
    for u, v, data in G_topo.edges(data=True):
        start_coords = np.array(u)
        end_coords = np.array(v)
        # Vector from start to end node
        segment_vector = end_coords - start_coords
        
        # Scale vector by physical spacing for accurate angles
        physical_segment_vector = segment_vector * np.array([voxel_spacing_x, voxel_spacing_y, voxel_spacing_z])
        
        # Z-axis vector is (0, 0, 1)
        z_axis_vector = np.array([0, 0, 1])

        dot_product = np.dot(physical_segment_vector, z_axis_vector)
        magnitude_segment = np.linalg.norm(physical_segment_vector)
        magnitude_z = np.linalg.norm(z_axis_vector)

        if magnitude_segment > 0:
            angle_rad = math.acos(np.clip(dot_product / (magnitude_segment * magnitude_z), -1.0, 1.0))
            angle_deg = math.degrees(angle_rad)
            segment_angles_z.append(angle_deg)

    mean_segment_angle_z = np.mean(segment_angles_z) if segment_angles_z else 0
    morphometric_dict["mean_angle_z"] = mean_segment_angle_z
    
    # Tortuosity: pass through network and calculate SOAM at branch points
    total_soam = 0
    for bp, paths_from_bp in branch_point_segment_paths.items():
        if len(paths_from_bp) < 2:
            # A branch point must have at least two segments to form an angle
            G_topo.nodes[bp]['angle'] = 0.0
            continue

        sum_of_angles_at_bp = 0.0
        
        # Iterate through all unique pairs of segments originating from the branch point
        # This forms all possible angles at the branch point.
        for i in range(len(paths_from_bp)):
            for j in range(i + 1, len(paths_from_bp)):
                path1 = paths_from_bp[i]
                path2 = paths_from_bp[j]

                # Vector for the first segment (from branch point to its next pixel)
                # Ensure the path has at least two points to form a vector
                if len(path1) >= 2:
                    vec1 = np.array(path1[-1]) - np.array(path1[0])
                else:
                    continue # Skip if path is too short

                # Vector for the second segment (from branch point to its next pixel)
                if len(path2) >= 2:
                    vec2 = np.array(path2[-1]) - np.array(path2[0])
                else:
                    continue # Skip if path is too short
                
                angle = calculate_angle_between_vectors(vec1, vec2)
                sum_of_angles_at_bp += angle
        
        G_topo.nodes[bp]['soam_at_branch_point'] = sum_of_angles_at_bp
        total_soam += sum_of_angles_at_bp

    morphometric_dict["total_SOAM"] = total_soam
    
    # Distance between Blood Vessels (Global Average)
    # Compute distance transform of the *background* of the vessels
    distance_to_vessel_boundary = distance_transform_edt(np.logical_not(binary_vessels_cleaned))
    # Average distance for non-vessel voxels
    background_distances = distance_to_vessel_boundary[np.logical_not(binary_vessels_cleaned)]
    mean_distance_between_vessels = np.mean(background_distances) if background_distances.size > 0 else 0
    morphometric_dict["mean_distance_between_vessels"] = mean_distance_between_vessels
    
            
        
    if print_outputs:
        print("Total Vessel Volume / Vascular Volume fraction =", vascular_volume_fraction)
        print("Mean Vessel Diameter:", mean_diameter)
        print(f"Number of branch points: {num_branch_points}")
        print(f"Number of endpoints: {num_endpoints}")
        print(f"Number of vessel segments: {num_segments}")
        print(f"Total vessel length (in voxels): {total_vessel_length_voxels}")
        print(f"Number of fundamental cycles (loops): {num_loops}")
        print(f"Vessel segments per projected XY area (sq microns): {vessel_segments_per_projected_area:.4f}")
        print(f"Vessel segments per unit volume (cubic microns): {vessel_segments_per_unit_volume:.4f}")
        print(f"Mean Vessel Diameter (voxels): {mean_vessel_diameter:.2f}")
        print(f"Mean Vessel Cross-sectional Area (voxels^2): {mean_vessel_area:.2f}")
        print(f"Mean Vessel Curvature (Path Length / Euclidean Distance): {mean_curvature:.2f}")
        print(f"Total Vessel Length (voxels): {total_vessel_length_voxels:.2f}")
        print(f"Number of Triple Junctions: {len(triple_junctions)}")
        print(f"Number of Quadruple Junctions: {len(quadruple_junctions)}") 
        print(f"Mean Vessel Segment Angle to Z-axis (degrees): {mean_segment_angle_z:.2f}")
        print(f"Total SOAM: {total_soam:.2f}")
        print(f"Mean Distance between Blood Vessels (voxels): {mean_distance_between_vessels:.2f}")
    
    return morphometric_dict    
    
    
# --- Custom 3D GLCM Function ---
def glcm_3d(input_array: np.ndarray,
                    distances: list[int] = [1],
                    directions: list[tuple[int, int, int]] = None,
                    levels: int = 2,
                    symmetric: bool = True,
                    normed: bool = True) -> np.ndarray:
    """
    Computes 3D Gray-Level Co-occurrence Matrices (GLCMs) for a given 3D integer input array.

    Args:
        input_array (np.ndarray): The 3D input array. Must be integer dtype (e.g., from quantization).
        distances (list[int], optional): A list of integer distances (d) to check for neighboring pixels. Defaults to [1].
        directions (list[tuple[int, int, int]], optional): A list of 3D direction vectors (dz, dy, dx).
                                                            If None, uses 13 common 3D directions at specified distances.
        levels (int, optional): The number of gray levels in the input_array. Defaults to 2.
                                This MUST match the actual number of unique values after quantization.
        symmetric (bool, optional): If True, GLCM is made symmetric (counts (i,j) and (j,i)). Defaults to True.
        normed (bool, optional): If True, GLCMs are normalized by their sum to become probabilities. Defaults to True.

    Returns:
        np.ndarray: A 3D array of GLCMs, with shape (levels, levels, num_directions * num_distances).

    Raises:
        ValueError: If input_array is not integer dtype or not 3D.
    """

    if not np.issubdtype(input_array.dtype, np.integer):
        raise ValueError("Input array should be of integer dtype. Please quantize your float array first.")
    if len(input_array.shape) != 3:
        raise ValueError("Input array should be 3-dimensional.")

    if levels < 2:
        levels = 2 

    if directions is None:
        unit_directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
            (0, 1, 1), (0, 1, -1), (1, 1, 1), (1, 1, -1),
            (1, -1, 1), (1, -1, -1)
        ]
    else:
        unit_directions = directions

    glcms_list = []

    for d in distances:
        for dz_unit, dy_unit, dx_unit in unit_directions:
            dz, dy, dx = dz_unit * d, dy_unit * d, dx_unit * d
            current_glcm = np.zeros((levels, levels), dtype=np.int32)
            count = 0

            for z in range(input_array.shape[0]):
                for y in range(input_array.shape[1]):
                    for x in range(input_array.shape[2]):
                        p1_val = input_array[z, y, x]

                        nz, ny, nx = z + dz, y + dy, x + dx

                        if (0 <= nz < input_array.shape[0] and
                            0 <= ny < input_array.shape[1] and
                            0 <= nx < input_array.shape[2]):

                            p2_val = input_array[nz, ny, nx]
                            current_glcm[p1_val, p2_val] += 1
                            count += 1
                            if symmetric:
                                current_glcm[p2_val, p1_val] += 1
                                count += 1
            
            if normed and count > 0:
                current_glcm = current_glcm / count
            
            glcms_list.append(current_glcm)

    if not glcms_list:
        return np.zeros((levels, levels, 0), dtype=np.float64 if normed else np.int32)

    return np.stack(glcms_list, axis=2)


# --- Quantization Step for GLCM ---

def glcm_summary(example, num_quantized_levels=128):
    # computes glcm summaries on example 3D image
    # Choose a desired number of gray levels (e.g., 8, 16, 32, 64, 128, 256)

    # Quantize data
    actual_min_val = example.min()
    actual_max_val = example.max()

    if actual_max_val == actual_min_val:
        # Handle the edge case where all values in the float array are identical
        quantized_data = np.zeros_like(example, dtype=np.uint8)
        print("Warning: Input array is flat (all values are identical). Quantized to a single level (0).")
        num_quantized_levels_actual = 1 # Only one unique level
    else:
        # Scale and shift the float values to fit into the integer range [0, num_quantized_levels - 1]
        # (value - min_val) / (max_val - min_val) -> scales to [0, 1]
        # * (num_quantized_levels - 1) -> scales to [0, num_quantized_levels - 1]
        scaled_data = (example - actual_min_val) / (actual_max_val - actual_min_val)
        quantized_data = (scaled_data * (num_quantized_levels - 1)).astype(np.uint8)
        num_quantized_levels_actual = num_quantized_levels

    # print(f"Quantized data range: [{quantized_data.min()}, {quantized_data.max()}] (Actual number of levels used: {num_quantized_levels_actual})")


    # --- Compute 3D GLCM and Texture Features on the QUANTIZED data ---

    glcm_distances_to_check = [1, 2, 3] # Example: check immediate, 2-voxel, and 3-voxel distances
    glcm_directions_all = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
        (0, 1, 1), (0, 1, -1), (1, 1, 1), (1, 1, -1),
        (1, -1, 1), (1, -1, -1)
    ]

    glcms_result = glcm_3d(quantized_data, # Pass the quantized integer data here
                                distances=glcm_distances_to_check,
                                directions=glcm_directions_all,
                                levels=num_quantized_levels_actual, # Pass the actual number of levels used
                                symmetric=True,
                                normed=True)    
    
    glcm_properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    mean_glcm_features_across_all_directions_and_distances = {}

    for prop in glcm_properties:
        prop_values = []
        for i in range(glcms_result.shape[2]):
            current_glcm = glcms_result[:, :, i]
            if current_glcm.sum() > 0:
                val = graycoprops(current_glcm[:, :, np.newaxis, np.newaxis], prop)[0,0] 
                prop_values.append(val)
            else:
                prop_values.append(np.nan)

        mean_glcm_features_across_all_directions_and_distances[prop] = \
            np.nanmean(prop_values) if prop_values and not np.all(np.isnan(prop_values)) else 0.0

    # print("\n--- Averaged 3D GLCM Texture Features (across all distances & directions) ---")
    # for prop, value in mean_glcm_features_across_all_directions_and_distances.items():
    #     print(f"{prop.capitalize()}: {value:.4f}")
    
    return mean_glcm_features_across_all_directions_and_distances


# --- 3D LBP Function (6-Connectivity, NON-ROTATION-INVARIANT) ---
def compute_3d_lbp_features(image_array: np.ndarray, mask_array: np.ndarray) -> dict:
    """
    Computes 3D Local Binary Pattern (LBP) features (histogram) for masked regions
    of a 3D image, using 6-connectivity (face neighbors, radius 1).
    This implementation DOES NOT apply rotation invariance mapping.

    Args:
        image_array (np.ndarray): The 3D input image array (integer dtype, e.g., quantized).
        mask_array (np.ndarray): The 3D binary mask array. LBP is computed only for mask=1 regions.

    Returns:
        dict: A dictionary representing the normalized histogram of LBP codes.
              Keys are LBP codes, values are their frequencies.
    """
    if not np.issubdtype(image_array.dtype, np.integer):
        print("Warning: LBP typically works best on quantized integer data. Input image_array is not integer dtype.")
        # If image_array is float, ensure it's quantized before calling this function.

    if image_array.shape != mask_array.shape:
        raise ValueError("Image array and mask array must have the same shape.")

    n, m, k = image_array.shape

    # Define the 6-connectivity unit directions (face neighbors)
    # The order matters for constructing the binary code (2**j)
    lbp_directions = [
        (0, 0, 1),  # +Z (bit 0)
        (0, 0, -1), # -Z (bit 1)
        (0, 1, 0),  # +Y (bit 2)
        (0, -1, 0), # -Y (bit 3)
        (1, 0, 0),  # +X (bit 4)
        (-1, 0, 0), # -X (bit 5)
    ]
    # For P=6 neighbors, LBP codes will range from 0 to 2^6-1 = 63.

    lbp_codes_masked = []

    # Iterate through each voxel
    for z_center in range(n):
        for y_center in range(m):
            for x_center in range(k):
                # Only calculate LBP for the mask region (where mask value is 1)
                if mask_array[z_center, y_center, x_center] == 1:
                    center_val = image_array[z_center, y_center, x_center]
                    decimal_lbp_code = 0

                    # Compare center voxel with its 6 neighbors
                    for j, (dz, dy, dx) in enumerate(lbp_directions):
                        nz, ny, nx = z_center + dz, y_center + dy, x_center + dx

                        # Check if neighbor is within image bounds
                        if (0 <= nz < n and 0 <= ny < m and 0 <= nx < k):
                            neighbor_val = image_array[nz, ny, nx]
                            
                            # Standard LBP comparison: 1 if neighbor > center, 0 otherwise
                            if neighbor_val > center_val:
                                bit = 1
                            else:
                                bit = 0 # Neighbor is less than or equal to center

                            decimal_lbp_code += bit * (2**j)
                        else:
                            # Handle boundary by implicitly assigning 0 for out-of-bounds bits
                            pass 

                    lbp_codes_masked.append(decimal_lbp_code)

    # Calculate frequency of each LBP pattern within the masked region
    frequency = Counter(lbp_codes_masked)

    # Normalize the frequencies to get the feature vector
    freq_total = sum(frequency.values())
    lbp_feature_dict = {}
    if freq_total > 0:
        for label, count in frequency.items():
            lbp_feature_dict[str(label)] = count / freq_total # Store as string label for consistency
    
    return lbp_feature_dict

# Apply LBP on vessel data
def lbp_summary(example, num_quantized_levels = 32, print_summary=False, plot_hist=False):

    vessel_mask = example >= 0
    # quantize
    actual_min_val = example.min()
    actual_max_val = example.max()

    if actual_max_val == actual_min_val:
        quantized_data = np.zeros_like(example, dtype=np.uint8)
        print("Warning: Input array is flat (all values are identical). Quantized to a single level (0).")
        num_quantized_levels_actual = 1
    else:
        # Scale and shift the float values to fit into the integer range [0, num_quantized_levels - 1]
        scaled_data = (example - actual_min_val) / (actual_max_val - actual_min_val)
        quantized_data = (scaled_data * (num_quantized_levels - 1)).astype(np.uint8)
        num_quantized_levels_actual = num_quantized_levels

    # --- Compute 3D LBP (6-Connectivity) on the QUANTIZED data within the MASK ---

    # Pass the quantized_data as the image_array and the boolean vessel_mask for masking.
    lbp_features_dict = compute_3d_lbp_features(quantized_data, vessel_mask) 

    if lbp_features_dict:
        # Sort by LBP pattern for consistent output
        sorted_lbp_features = dict(sorted(lbp_features_dict.items(), key=lambda item: int(item[0])))
        
        if print_summary:
            # LBP codes for 6-connectivity range from 0 to 63 (2^6 - 1)
            if len(sorted_lbp_features) <= 20: # Print all if few unique patterns
                for pattern, freq in sorted_lbp_features.items():
                     print(f"  Pattern {pattern}: {freq:.4f}")
            else: # Print summary if many unique patterns
                print(f"  Total {len(sorted_lbp_features)} unique patterns found.")
                print(f"  Top 10 most frequent patterns:")
                for pattern, freq in list(sorted_lbp_features.items())[:10]:
                    print(f"    Pattern {pattern}: {freq:.4f}")
                print(f"  ...")

    else:
        print("No 3D LBP features could be computed (mask might be empty or too sparse).")

    # --- Plot the 3D LBP Histogram if needed ---
    if plot_hist:
        if lbp_features_dict:
            lbp_codes = np.array([int(code) for code in lbp_features_dict.keys()])
            frequencies = np.array(list(lbp_features_dict.values()))

            # Sort by LBP code for correct bar order
            sort_indices = np.argsort(lbp_codes)
            sorted_lbp_codes = lbp_codes[sort_indices]
            sorted_frequencies = frequencies[sort_indices]

            plt.figure(figsize=(15, 7))

            # To ensure all 64 possible bins are represented (0-63), even if some have 0 frequency
            all_possible_codes = np.arange(0, 64)
            all_frequencies = np.zeros(64)
            for i, code in enumerate(sorted_lbp_codes):
                if 0 <= code < 64: # Safety check
                    all_frequencies[code] = sorted_frequencies[i]

            plt.bar(all_possible_codes, all_frequencies, width=1.0, edgecolor='black', alpha=0.7)

            plt.xlabel('3D LBP Code (6-Connectivity)')
            plt.ylabel('Normalized Frequency')
            plt.title('Histogram of 3D LBP Codes from Quantized Vessel Data')

            # Adjust x-axis ticks
            plt.xticks(np.arange(0, 64, 5)) # Show ticks every 5 codes
            plt.xlim([-1, 64]) # Ensure bars don't touch axes and cover all 64 bins
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

        else:
            print("LBP features dictionary is empty. Cannot plot histogram.")

    return lbp_features_dict
