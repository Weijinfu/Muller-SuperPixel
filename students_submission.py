import cv2
import numpy as np

############## mask #########################
def get_mask(image: np.ndarray, threshold: int) -> np.ndarray:
    mask = np.where(image < threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 可调整核大小
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    return mask
#############################################

############# muller features #########################
# from muller_features import M11, M12, M13, M14, M21, M22, M23, M24, M31, M32, M33, M34, M41, M42, M43, M44

# def my_feature_1(MM):
#     return (M22(MM) * M33(MM)) / 2.

# def my_feature_2(MM):
#     return M12(MM) + M23(MM) - 1.

# function_list = [my_feature_1, my_feature_2]
########################################################

############# pft #########################
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import MiniBatchKMeans
# import muller_utils
# def get_pft(cluster: MiniBatchKMeans, scaler: StandardScaler, data_list, mask_list, feature_type="pbp"):
#     feature_all = muller_utils.data_mask_to_feature(data_list, mask_list, type=feature_type)
#     feature_all = scaler.transform(feature_all)
#     labels = cluster.predict(feature_all)
#     cluster_counts = np.bincount(labels, minlength=cluster.n_clusters)
#     # choose PFTs based on top k cluster counts
#     top_k = 50  # 可以根据需要调整
#     top_k_indices = np.argsort(cluster_counts)[-top_k:][::-1]
#     pft = [i for i in top_k_indices if cluster_counts[i] > 0]  # 确保PFTs不为空
#     print(f"Selected PFTs: {pft}")
#     if len(pft) == 0:
#         raise ValueError("No valid PFTs found. Please check your data and clustering parameters.")
#     return pft
#######################################


