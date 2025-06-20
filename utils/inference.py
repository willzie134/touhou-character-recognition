import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import pywt
from imgutils.detect import detect_faces
from PIL import Image
import pickle
import json
import joblib

def flatten_feature_dict(feature_dict):
    """Convert dictionary to flat array and ordered keys"""
    keys = list(feature_dict.keys())
    values = np.array([feature_dict[k] for k in keys])
    return values

# ==== Parameters ====
GLCM_DISTANCES = [3]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# ==== ICCV ====
def extract_iccv(image, n_colors=27, tau=0.01):
    quantized = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    quantized = quantized[:, :, 0]
    bins = np.linspace(0, 180, n_colors + 1).astype(np.uint8)
    indices = np.digitize(quantized, bins) - 1

    iccv_features = {}
    min_region_size = int(tau * image.shape[0] * image.shape[1])

    for i in range(n_colors):
        mask = (indices == i).astype(np.uint8)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        coherent = 0
        incoherent = 0
        centroid = (0, 0)

        for j in range(1, n_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            if area >= min_region_size:
                coherent += area
                centroid = centroids[j]
            else:
                incoherent += area

        iccv_features[f"color_{i}_coherent"] = coherent
        iccv_features[f"color_{i}_incoherent"] = incoherent
        iccv_features[f"color_{i}_cx"] = centroid[0]
        iccv_features[f"color_{i}_cy"] = centroid[1]

    return iccv_features

# ==== GLCM ====
def extract_glcm(gray_img):
    glcm = graycomatrix(gray_img, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    feature_dict = {}
    for prop in props:
        stats = graycoprops(glcm, prop)
        for d_idx, d in enumerate(GLCM_DISTANCES):
            for a_idx, a in enumerate(GLCM_ANGLES):
                key = f"{prop}_d{d}_a{int(np.rad2deg(a))}"
                feature_dict[key] = stats[d_idx, a_idx]
    return feature_dict

# ==== DWT-MSLBP ====
def extract_dwt_mslbp(gray_img):
    coeffs = pywt.dwt2(gray_img, 'haar')
    LL, (LH, HL, HH) = coeffs
    LL_norm = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX)
    LL_uint8 = LL_norm.astype(np.uint8)

    lbp_LL = local_binary_pattern(LL_uint8, P=24, R=3, method='uniform')
    hist, _ = np.histogram(lbp_LL.ravel(), bins=np.arange(0, 24 + 3), density=True)
    return {f"dwt_lbp_{i}": val for i, val in enumerate(hist)}


# ==== Color Histogram (HSV) ====
def extract_color_histogram(img, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return {f"hsv_bin_{i}": val for i, val in enumerate(hist)}

# ==== ORB Keypoints ====
def extract_orb_keypoints(img):
    orb = cv2.ORB_create(nfeatures=100)
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        des = np.zeros((1, 32))
    avg_des = des.mean(axis=0)
    return {f"orb_{i}": val for i, val in enumerate(avg_des)}

def detect_and_crop_all_faces(image_cv, margin=0.4):
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    try:
        faces = detect_faces(image_pil)
    except:
        return [], []

    crops = []
    boxes = []

    for f in faces:
        print(f)
        if f[2] < 0.5:
            continue

        # Assume new format: f["bbox"] or similar → (x0, y0, x1, y1)
        x0, y0, x1, y1 = f[0]  # Replace with correct key if needed

        # Expand margin around center
        w, h = x1 - x0, y1 - y0
        cx, cy = x0 + w // 2, y0 + h // 2
        size = int(max(w, h) * (1 + margin))

        x1_exp = max(cx - size // 2, 0)
        y1_exp = max(cy - size // 2, 0)
        x2_exp = min(cx + size // 2, image_cv.shape[1])
        y2_exp = min(cy + size // 2, image_cv.shape[0])

        crop = image_cv[y1_exp:y2_exp, x1_exp:x2_exp]
        crops.append(crop)

        # Save the original bbox for annotation (not the expanded one)
        boxes.append((x0, y0, x1, y1))

    return crops, boxes


# ==== Feature Extraction Pipeline ====
def extract_faces_features(img, return_raw_faces=False, detect_face=True, margin=0.4, target_size=(100, 100)):
    if isinstance(img, str):
        img = cv2.imread(img)

    raw_faces = []

    if detect_face:
        faces, raw_faces = detect_and_crop_all_faces(img, margin=margin)
    else:
        faces = [img]

    all_features = []

    for face in faces:
        resized = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        features = {}
        features.update(extract_iccv(resized))
        features.update(extract_glcm(gray))
        features.update(extract_dwt_mslbp(gray))
        features.update(extract_color_histogram(resized))
        features.update(extract_orb_keypoints(gray))

        all_features.append(features)

    if return_raw_faces:
        return all_features, raw_faces
    else:
        return all_features



def predict_faces_in_image(img_cv, svm_model, lda_model, scaler, label_names, threshold=0.5, top_k=3):
    results = []

    faces_features, face_bboxes = extract_faces_features(img_cv, return_raw_faces=True)

    if len(faces_features) == 0:
        return results

    for i, features in enumerate(faces_features):

        features = flatten_feature_dict(features)
        features = features.reshape(1, -1)

        # Extract features
        features_lda = lda_model.transform(scaler.transform(features))

        # Predict probabilities
        probs = svm_model.predict_proba(features_lda)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_preds = [(label_names[idx], probs[idx]) for idx in top_indices]

        # Filter by confidence threshold
        best_conf = probs[top_indices[0]]
        is_confident = best_conf >= threshold

        # Get bounding box
        results.append({
            "face_id": i + 1,
            "box": face_bboxes[i],
            "predictions": top_preds,
            "is_confident": is_confident
        })

    return results

def annotate_image(img_cv, results):
    img_annotated = img_cv.copy()
    for res in results:
        if res["box"] is None:
            continue
        x0, y0, x1, y1 = res["box"]
        label_text = res['predictions'][0][0]

        # Box color
        color = (0, 255, 0) if res["is_confident"] else (0, 0, 255)

        # Draw face bounding box
        cv2.rectangle(img_annotated, (x0, y0), (x1, y1), color, 2)

        # --- Draw contrasting background for text ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label_text.capitalize(), font, font_scale, font_thickness)
        text_y = y1 + text_height + 5
        text_x = x0 + 10

        # Draw face id above the bounding box
        id_text = f"Face {res['face_id']}"
        (id_width, id_height), id_baseline = cv2.getTextSize(id_text, font, font_scale, font_thickness)
        id_x = x0 + 10
        id_y = y0 - 5

        # Background rectangle for id
        cv2.rectangle(
            img_annotated,
            (x0, y0 - id_height - id_baseline - 8),
            (id_x + id_width + 6, y0),
            (0, 255, 0),
            thickness=-1
        )

        # Draw id text
        cv2.putText(
            img_annotated,
            id_text,
            (id_x, y0 - 5),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA
        )

        # Background rectangle
        cv2.rectangle(
            img_annotated,
            (x0, y1),
            (text_x + text_width + 6, text_y + baseline + 6),
            (0, 255, 0),
            thickness=-1
        )

        # Draw text
        cv2.putText(
            img_annotated,
            label_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),  # White text
            font_thickness,
            lineType=cv2.LINE_AA
        )

    return img_annotated

def load_models():
    import os
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    with open(os.path.join(base_dir, 'label_names.json')) as f:
        label_names = json.load(f)
    with open(os.path.join(base_dir, 'svm_model.pkl'), 'rb') as f:
        svm = joblib.load(f)
    with open(os.path.join(base_dir, 'raw_lda.pkl'), 'rb') as f:
        lda = joblib.load(f)
    with open(os.path.join(base_dir, 'scaler.pkl'), 'rb') as f:
        scaler = joblib.load(f)
    return svm, lda, scaler, label_names
