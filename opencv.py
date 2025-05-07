import cv2
import numpy as np
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

os.makedirs("temp", exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.route("/")
def home():
    return "SIFT Flask API is running!"

def compute_lab_histogram_similarity(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return max(min(similarity * 100, 100), 0)

def check_if_images_are_identical(img1, img2):
    """Check if two images are identical by comparing pixel values."""
    return np.array_equal(img1, img2)

#FROM HERE

def remove_white_background(img):
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower_white, upper_white)
    return cv2.bitwise_and(img, img, mask=~mask)

def resize_with_padding(img, target_shape):
    """Resize image to fit within target_shape while maintaining aspect ratio and padding with white."""
    target_h, target_w = target_shape
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def compute_similarity(img1_path, img2_path, match_image_filename):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return {"error": "Could not load one or both images."}

    # Resize both images to same shape
    target_shape = (1000, 1000)
    img1 = resize_with_padding(img1, target_shape)
    img2 = resize_with_padding(img2, target_shape)

    # Remove white backgrounds
    img1_no_bg = remove_white_background(img1)
    img2_no_bg = remove_white_background(img2)

    # Grayscale conversion
    gray1 = cv2.cvtColor(img1_no_bg, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_no_bg, cv2.COLOR_BGR2GRAY)

    # SIFT + FLANN matcher
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return {"error": "Could not compute descriptors."} #para saan 'to?

    if des1 is None or des2 is None:
        return {"error": "Could not compute descriptors for one or both images."}

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    good_matches = [m for m in matches if m.distance < 0.85 * max(m.distance for m in matches)]
    matches = sorted(good_matches, key=lambda x: x.distance)

    total_keypoints = min(len(kp1), len(kp2))
    num_matches = len(matches)

    # Draw match image
    #match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    match_img = np.concatenate((img1, img2), axis=1)

    match_image_path = os.path.join(os.getcwd(), "static", match_image_filename)
    cv2.imwrite(match_image_path, match_img)

    # Final match ratio with 80% boost if good matches >= 500
    base_ratio = (num_matches / total_keypoints) * 100 if total_keypoints > 0 else 0
    final_match_ratio = base_ratio + 80 if num_matches >= 200 else base_ratio
    final_match_ratio = min(final_match_ratio, 100)

    # LAB color similarity using histogram method
    lab_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    lab_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    lab_similarity = compute_lab_histogram_similarity(lab_img1, lab_img2)

    # Final combined similarity (50/50 weighting)
    final_similarity_score = (final_match_ratio * 0.7) + (lab_similarity * 0.3)

    return {
        "sift_similarity": round(final_match_ratio, 2),
        "lab_similarity": round(lab_similarity, 2),
        "final_similarity_score": round(final_similarity_score, 2),
        "good_matches": num_matches,
        "keypoints_image1": len(kp1),
        "keypoints_image2": len(kp2),
        "match_image_url": request.host_url + f"static/{match_image_filename}"
    }

# TO HERE
@app.route("/compare", methods=["POST"])
def compare_images():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "Both image files are required."}), 400

    img1 = request.files['img1']
    img2 = request.files['img2']

    img1_name = os.path.splitext(img1.filename)[0]
    img2_name = os.path.splitext(img2.filename)[0]
    match_image_filename = f"match_{img1_name}_{img2_name}.jpg"

    # Unique temp paths
    import uuid
    img1_ext = os.path.splitext(img1.filename)[1]
    img2_ext = os.path.splitext(img2.filename)[1]

    img1_path = os.path.join("temp", f"temp1_{uuid.uuid4().hex}{img1_ext}")
    img2_path = os.path.join("temp", f"temp2_{uuid.uuid4().hex}{img2_ext}")

    img1.save(img1_path)
    img2.save(img2_path)

    result = compute_similarity(img1_path, img2_path, match_image_filename)

    os.remove(img1_path)
    os.remove(img2_path)

    return jsonify(result)


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5050)