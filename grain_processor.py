import cv2
import numpy as np
import os
import zipfile
import csv
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.color import label2rgb


# -------------------------------
# CREATE OUTPUT FOLDER IF NOT EXISTS
# -------------------------------
def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# -------------------------------
# CORE SEGMENTATION FUNCTION
# -------------------------------
def segment_grains(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Fast + effective blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu Threshold
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Distance Transform
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    # Find peaks
    coords = peak_local_max(dist, min_distance=10, labels=thresh)

    mask = np.zeros(dist.shape, dtype=bool)
    if len(coords) > 0:
        mask[tuple(coords.T)] = True

    markers, _ = ndi.label(mask)

    # Watershed
    labels = watershed(-dist, markers, mask=thresh)

    return labels


# -------------------------------
# ANALYSIS FUNCTION
# -------------------------------
def analyze_grains(labels):

    props = regionprops(labels)

    areas = []
    diameters = []

    for prop in props:
        areas.append(prop.area)
        diameters.append(prop.equivalent_diameter)

    total_count = len(areas)
    avg_area = np.mean(areas) if areas else 0

    return areas, diameters, total_count, avg_area


# -------------------------------
# SAVE CSV
# -------------------------------
def save_csv(csv_path, areas, diameters):

    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Grain ID", "Area", "Equivalent Diameter"])

        for i in range(len(areas)):
            writer.writerow([i + 1, areas[i], diameters[i]])


# -------------------------------
# SAVE HISTOGRAM
# -------------------------------
def save_histogram(hist_path, areas):

    plt.figure()
    plt.hist(areas, bins=20)
    plt.title("Grain Size Distribution")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()


# -------------------------------
# PROCESS SINGLE IMAGE
# -------------------------------
def process_single(image_path, output_folder):

    ensure_folder(output_folder)

    filename = os.path.basename(image_path)

    image = cv2.imread(image_path)

    if image is None:
        return None

    labels = segment_grains(image)

    areas, diameters, count, avg_area = analyze_grains(labels)

    # Color segmented image
    colored = label2rgb(labels, image=image, bg_label=0)
    colored = (colored * 255).astype(np.uint8)

    # File names
    name_no_ext = os.path.splitext(filename)[0]

    result_img = f"{name_no_ext}_result.png"
    csv_file = f"{name_no_ext}.csv"
    hist_file = f"{name_no_ext}_hist.png"

    # Save files
    cv2.imwrite(os.path.join(output_folder, result_img), colored)

    save_csv(os.path.join(output_folder, csv_file), areas, diameters)
    save_histogram(os.path.join(output_folder, hist_file), areas)

    return {
        "original": filename,
        "image": result_img,
        "csv": csv_file,
        "hist": hist_file,
        "count": count,
        "avg_area": round(avg_area, 2)
    }


# -------------------------------
# PROCESS ZIP FILE
# -------------------------------
def process_zip(zip_path, output_folder):

    ensure_folder(output_folder)

    results = []

    with zipfile.ZipFile(zip_path, 'r') as archive:

        for file in archive.namelist():

            if file.lower().endswith(('.png', '.jpg', '.jpeg')):

                try:
                    data = archive.read(file)
                    arr = np.frombuffer(data, np.uint8)
                    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if image is None:
                        continue

                    filename = os.path.basename(file)
                    temp_path = os.path.join(output_folder, filename)

                    # Save temp file
                    cv2.imwrite(temp_path, image)

                    result = process_single(temp_path, output_folder)

                    if result:
                        results.append(result)

                except Exception as e:
                    print(f"Error processing {file}: {e}")

    return results


# -------------------------------
# MAIN FUNCTION (AUTO DETECT)
# -------------------------------
def process_file(file_path, output_folder):

    if file_path.lower().endswith('.zip'):
        return process_zip(file_path, output_folder)
    else:
        return process_single(file_path, output_folder)