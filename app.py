#!/usr/bin/env python
"""
Interactive Streamlit application to load multiple annotation datasets in different labeling formats,
analyze each dataset separately, modify, merge, and delete labels, combine the datasets,
and generate a new annotations file in the desired format.
Includes functionality to convert segmentation annotations to bounding boxes.
Allows you to add new empty classes (especially useful for COCO).
"""

import json
import xml.etree.ElementTree as ET
import streamlit as st
from io import StringIO, BytesIO
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import copy

# --------------------------------------------------------------------------------
# Load and Save Functions, Correcting Duplicate IDs in COCO
# --------------------------------------------------------------------------------

def load_coco_annotations(file):
    """
    Loads a COCO JSON file and corrects duplicate annotation IDs if detected.
    """
    data = json.load(file)
    if "annotations" in data and isinstance(data["annotations"], list):
        ann_ids = [ann.get("id") for ann in data["annotations"] if "id" in ann]
        ann_ids = [x for x in ann_ids if x is not None]
        if len(ann_ids) != len(set(ann_ids)):
            st.warning("Duplicate annotation IDs detected in this COCO dataset; reassigning them...")
            new_id = 1
            for ann in data["annotations"]:
                ann["id"] = new_id
                new_id += 1
    return data

def save_coco_annotations(data):
    """
    Saves the COCO annotations as a JSON-formatted string with indentation for readability.
    """
    return json.dumps(data, indent=4)

def load_pascal_voc_annotations(file):
    """
    Loads a Pascal VOC XML file and returns the root of the XML tree.
    """
    tree = ET.parse(file)
    root = tree.getroot()
    return root

def save_pascal_voc_annotations(root):
    """
    Saves the Pascal VOC XML tree as a string.
    """
    tree = ET.ElementTree(root)
    with StringIO() as f:
        tree.write(f, encoding='unicode')
        return f.getvalue()

# --------------------------------------------------------------------------------
# Functions to Modify Labels (COCO and Pascal VOC)
# --------------------------------------------------------------------------------

def sanitize_class_name(name):
    """
    Sanitizes the class name to avoid invalid characters.
    """
    # Remove special characters and excessive spaces
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', name)).strip()

def modify_labels_coco(data, class_mapping, classes_to_delete, new_classes=None):
    """
    1) Removes classes specified in 'classes_to_delete'.
    2) Renames classes according to 'class_mapping'.
    3) Adds 'new_classes' without annotations (if they do not already exist).
    4) Sorts classes alphabetically by 'name'.
    5) Reassigns category IDs in alphabetical order and updates annotations.
    6) Groups annotations of renamed classes under the same name.
    """
    if new_classes is None:
        new_classes = []

    original_categories = data.get('categories', [])
    original_annotations = data.get('annotations', [])
    original_images = data.get('images', [])

    # 1) Remove categories that are in classes_to_delete
    #    And rename those in class_mapping
    filtered_categories = []
    for cat in original_categories:
        old_name = cat['name']
        if old_name in classes_to_delete:
            continue

        # Rename if in the mapping
        new_name = class_mapping.get(old_name, old_name)
        new_name = sanitize_class_name(new_name)
        filtered_categories.append({
            "id": cat["id"],  # Temporarily keep the old id
            "name": new_name,
            "supercategory": cat.get("supercategory", "")
        })

    # 2) Add 'new_classes' if they don't already exist
    existing_names = {c["name"] for c in filtered_categories}
    for cls in new_classes:
        cls_sanitized = sanitize_class_name(cls)
        if cls_sanitized and cls_sanitized not in existing_names:
            filtered_categories.append({
                "id": None,  # Temporary, will reassign later
                "name": cls_sanitized,
                "supercategory": ""
            })
            st.info(f"New class added: {cls_sanitized}")
        else:
            st.warning(f"Invalid or empty class name: '{cls}'")

    # 3) Build the list of annotations, discarding those pointing to deleted categories
    kept_old_ids = {cat["id"] for cat in filtered_categories if cat["id"] is not None}
    new_annotations = []
    for ann in original_annotations:
        old_cat_id = ann["category_id"]
        if old_cat_id in kept_old_ids:
            new_annotations.append(ann)

    # 4) Sort categories alphabetically by 'name'
    filtered_categories.sort(key=lambda c: c['name'])

    # 5) Reassign IDs from 1..N and map annotations
    category_name_to_new_id = {}
    new_categories = []
    new_id = 1
    for cat in filtered_categories:
        name = cat['name']
        if name not in category_name_to_new_id:
            category_name_to_new_id[name] = new_id
            new_categories.append({
                "id": new_id,
                "name": name,
                "supercategory": cat['supercategory']
            })
            new_id += 1
        else:
            # Class already exists, do not add again
            st.warning(f"Duplicate class name detected: '{name}'. All annotations will be grouped under a single class.")

    # 6) Update annotations with the new category IDs
    for ann in new_annotations:
        old_cat_id = ann["category_id"]
        # Find the original category name
        old_cat_name = next((c['name'] for c in original_categories if c['id'] == old_cat_id), None)
        if old_cat_name:
            new_cat_id = category_name_to_new_id.get(old_cat_name)
            if new_cat_id:
                ann["category_id"] = new_cat_id

    # 7) Remove images that have no annotations
    annotated_img_ids = {ann['image_id'] for ann in new_annotations}
    new_images = [img for img in original_images if img['id'] in annotated_img_ids]

    # 8) Verify bounding boxes
    final_annotations = []
    for ann in new_annotations:
        x, y, w, h = ann["bbox"]
        if w > 0 and h > 0:
            final_annotations.append(ann)
        else:
            st.warning(f"Annotation with ID {ann['id']} removed due to invalid bbox.")

    # 9) Save back to data
    data['categories'] = new_categories
    data['annotations'] = final_annotations
    data['images'] = new_images

    return data

def modify_labels_pascal_voc(root, class_mapping, classes_to_delete, new_classes=None):
    """
    Modifies labels in Pascal VOC XML files by:
    1) Removing objects of classes in 'classes_to_delete'.
    2) Renaming objects according to 'class_mapping'.
    Note: Adding new classes without annotations is not directly applicable in Pascal VOC.
    """
    if root is None:
        return None

    objects = root.findall('object')
    to_remove = []
    for obj in objects:
        original_name = obj.find('name').text
        if original_name in classes_to_delete:
            to_remove.append(obj)
            st.info(f"Object with class '{original_name}' removed.")
            continue
        new_name = class_mapping.get(original_name, original_name)
        new_name_sanitized = sanitize_class_name(new_name)
        if new_name_sanitized != original_name:
            st.info(f"Object renamed from '{original_name}' to '{new_name_sanitized}'.")
            obj.find('name').text = new_name_sanitized

    # Remove marked objects
    for obj in to_remove:
        root.remove(obj)

    # Pascal VOC does not allow adding <object> without a bounding box
    if not root.findall('object'):
        st.warning("No objects remaining in the image after modifications.")
        return None

    # Adding new classes without annotations is not directly applicable in Pascal VOC
    # New classes will be handled in annotations if added manually

    return root

# --------------------------------------------------------------------------------
# Functions to Merge Datasets
# --------------------------------------------------------------------------------

def merge_coco_datasets(datasets):
    """
    Merges multiple COCO datasets into a single dataset.
    Ensures that categories with the same name are merged and have unique IDs.
    """
    merged_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    category_name_set = set()
    all_categories = []
    image_id_offset = 0
    annotation_id = 1

    # 1. Collect all unique categories
    for data in datasets:
        if not data:
            continue
        for cat in data['categories']:
            cat_name = cat['name']
            supercat = cat.get('supercategory', '')
            if cat_name not in category_name_set:
                category_name_set.add(cat_name)
                all_categories.append({'name': cat_name, 'supercategory': supercat})

    # 2. Sort categories alphabetically
    all_categories = sorted(all_categories, key=lambda c: c['name'])

    # 3. Assign new IDs to categories
    category_name_to_id = {}
    new_categories = []
    new_id = 1
    for cat in all_categories:
        name = cat['name']
        if name not in category_name_to_id:
            category_name_to_id[name] = new_id
            new_categories.append({
                'id': new_id,
                'name': name,
                'supercategory': cat['supercategory']
            })
            new_id += 1
    merged_data['categories'] = new_categories

    # 4. Merge images and annotations
    for data in datasets:
        if not data:
            continue

        # Create a mapping from old image IDs to new image IDs
        image_id_mapping = {}
        for img in data['images']:
            new_image_id = img['id'] + image_id_offset
            image_id_mapping[img['id']] = new_image_id
            # Copy the image to avoid modifying the original
            img_copy = copy.deepcopy(img)
            img_copy['id'] = new_image_id
            merged_data['images'].append(img_copy)

        # Update the image ID offset
        if merged_data['images']:
            image_id_offset = max(img['id'] for img in merged_data['images']) + 1
        else:
            image_id_offset = 1

        # Process annotations
        for ann in data['annotations']:
            ann_copy = copy.deepcopy(ann)
            ann_copy['id'] = annotation_id
            # Update image_id
            ann_copy['image_id'] = image_id_mapping.get(ann['image_id'], ann['image_id'])
            # Map category_id to the new ID based on the name
            old_cat_id = ann['category_id']
            old_cat_name = next((c['name'] for c in data['categories'] if c['id'] == old_cat_id), None)
            if old_cat_name:
                new_cat_id = category_name_to_id.get(old_cat_name)
                if new_cat_id:
                    ann_copy['category_id'] = new_cat_id
                    merged_data['annotations'].append(ann_copy)
                    annotation_id += 1
                else:
                    st.warning(f"Category '{old_cat_name}' not found in the category mapping. Annotation ID {ann_copy['id']} omitted.")
            else:
                st.warning(f"Category ID {old_cat_id} not found in the dataset's categories. Annotation ID {ann_copy['id']} omitted.")

    return merged_data

def merge_pascal_voc_datasets(datasets):
    """
    Merges multiple Pascal VOC XML datasets into a single list of XML roots.
    """
    merged_roots = []
    for root in datasets:
        if root is not None:
            merged_roots.append(root)
    return merged_roots

# --------------------------------------------------------------------------------
# Analysis and Visualization Functions
# --------------------------------------------------------------------------------

def analyze_coco_dataset(data):
    """
    Analyzes a COCO dataset and returns insights including the number of images, annotations,
    instances per class, and images per class.
    """
    num_images = len(data['images'])
    num_annotations = len(data['annotations'])
    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    all_cat_names = list(category_id_to_name.values())

    annotations_df = pd.DataFrame(data['annotations'])
    instances_per_class = {cat_name: 0 for cat_name in all_cat_names}
    images_per_class = {cat_name: 0 for cat_name in all_cat_names}

    if not annotations_df.empty:
        annotations_df['category_id'] = annotations_df['category_id'].astype(int)
        annotations_df['category_name'] = annotations_df['category_id'].map(category_id_to_name)

        for cat_name, group in annotations_df.groupby('category_name'):
            instances_per_class[cat_name] = len(group)
            images_per_class[cat_name] = group['image_id'].nunique()

    return {
        'num_images': num_images,
        'num_annotations': num_annotations,
        'instances_per_class': instances_per_class,
        'images_per_class': images_per_class
    }

def analyze_pascal_voc_dataset(root):
    """
    Analyzes a Pascal VOC XML dataset and returns insights including the number of images, annotations,
    instances per class, and images per class.
    """
    if root is None:
        return {
            'num_images': 0,
            'num_annotations': 0,
            'instances_per_class': {},
            'images_per_class': {}
        }

    objects = root.findall('object')
    num_annotations = len(objects)
    num_images = 1  # 1 XML = 1 image

    class_names = [obj.find('name').text for obj in objects]
    if class_names:
        counts = pd.Series(class_names).value_counts()
        instances_per_class = counts.to_dict()
    else:
        instances_per_class = {}

    images_per_class = {c: 1 for c in instances_per_class}
    return {
        'num_images': num_images,
        'num_annotations': num_annotations,
        'instances_per_class': instances_per_class,
        'images_per_class': images_per_class
    }

def visualize_dataset_insights(insights, dataset_name):
    """
    Visualizes the analysis insights for a dataset using Streamlit components.
    """
    with st.expander(f"View analysis for {dataset_name}"):
        st.subheader(f"Analysis for {dataset_name}")
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Number of images:** {insights['num_images']}")
                st.write(f"**Number of annotations:** {insights['num_annotations']}")
                st.write("**Instances per category:**")
                if insights['instances_per_class']:
                    st.dataframe(pd.DataFrame.from_dict(
                        insights['instances_per_class'], orient='index', columns=['Count']))
                else:
                    st.write("No instances in this dataset.")

            with col2:
                st.write("**Images per category:**")
                if insights['images_per_class']:
                    st.dataframe(pd.DataFrame.from_dict(
                        insights['images_per_class'], orient='index', columns=['Count']))
                else:
                    st.write("No images associated to classes.")

        # Bar chart (instances)
        if insights['instances_per_class']:
            st.write("**Distribution of Instances per Category:**")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            keys = list(insights['instances_per_class'].keys())
            values = list(insights['instances_per_class'].values())
            bars = ax1.bar(range(len(keys)), values)
            ax1.set_xticks(range(len(keys)))
            ax1.set_xticklabels(keys, rotation=45, ha='right')
            ax1.set_ylabel('Number of Instances')
            ax1.set_xlabel('Classes')

            for bar in bars:
                yval = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width()/2.0,
                    yval + max(values)*0.01,
                    int(yval),
                    va='bottom',
                    ha='center'
                )

            st.pyplot(fig1)

        # Bar chart (images)
        if insights['images_per_class']:
            st.write("**Distribution of Images per Category:**")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            keys = list(insights['images_per_class'].keys())
            values = list(insights['images_per_class'].values())
            bars = ax2.bar(range(len(keys)), values, color='orange')
            ax2.set_xticks(range(len(keys)))
            ax2.set_xticklabels(keys, rotation=45, ha='right')
            ax2.set_ylabel('Number of Images')
            ax2.set_xlabel('Classes')
            for bar in bars:
                yval = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width()/2.0,
                    yval + max(values)*0.01,
                    int(yval),
                    va='bottom',
                    ha='center'
                )
            st.pyplot(fig2)

# --------------------------------------------------------------------------------
# Conversions (COCO <-> Pascal VOC)
# --------------------------------------------------------------------------------

def convert_segmentation_to_bboxes_coco(data):
    """
    Converts segmentation annotations to bounding boxes in a COCO dataset.
    """
    for ann in data['annotations']:
        segmentation = ann.get('segmentation', [])
        if segmentation:
            ann['bbox'] = polygon_to_bbox(segmentation)
    return data

def polygon_to_bbox(segmentation):
    """
    Converts polygon segmentation to a bounding box.
    """
    all_points = []
    for seg in segmentation:
        xs = seg[0::2]
        ys = seg[1::2]
        all_points.extend(zip(xs, ys))

    if not all_points:
        return [0, 0, 0, 0]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def force_forward_slash_join(prefix: str, filename: str) -> str:
    """
    Joins prefix and filename using forward slashes, ensuring no duplicate slashes.
    """
    prefix = prefix.replace('\\', '/')
    filename = filename.replace('\\', '/')
    prefix = prefix.rstrip('/')
    filename = filename.lstrip('/')
    return prefix + '/' + filename

def convert_pascal_voc_to_coco(roots):
    """
    Converts multiple Pascal VOC XML annotations to a single COCO dataset.
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    category_name_to_id = {}
    category_id = 1
    annotation_id = 1
    image_id = 1

    for root in roots:
        filename_node = root.find('filename')
        if filename_node is not None:
            filename = filename_node.text
        else:
            filename = f"image_{image_id}.jpg"

        size_node = root.find('size')
        if size_node is not None:
            width_node = size_node.find('width')
            height_node = size_node.find('height')
            width = int(float(width_node.text)) if width_node is not None else 0
            height = int(float(height_node.text)) if height_node is not None else 0
        else:
            width, height = 0, 0

        coco_data['images'].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in category_name_to_id:
                category_name_to_id[class_name] = category_id
                coco_data['categories'].append({
                    "id": category_id,
                    "name": class_name,
                    "supercategory": ""
                })
                category_id += 1

            cid = category_name_to_id[class_name]
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            else:
                bbox = [0, 0, 0, 0]

            coco_data['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cid,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    coco_data['categories'] = sorted(coco_data['categories'], key=lambda c: c['name'])
    return coco_data

def convert_coco_to_pascal_voc(data):
    """
    Converts a COCO dataset to multiple Pascal VOC XML annotations.
    Returns a list of tuples containing the XML root and the filename.
    """
    pascal_voc_files = []

    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}

    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    for img_id, img_info in images.items():
        filename = img_info.get('file_name', f"image_{img_id}.jpg")
        width = img_info.get('width', 0)
        height = img_info.get('height', 0)
        depth = 3  # Assuming RGB

        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'folder').text = 'images'
        ET.SubElement(annotation, 'filename').text = filename
        ET.SubElement(annotation, 'path').text = f'images/{filename}'

        source = ET.SubElement(annotation, 'source')
        ET.SubElement(source, 'database').text = 'Unknown'

        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = str(depth)

        ET.SubElement(annotation, 'segmented').text = '0'

        # <object> for each annotation
        for ann in annotations_by_image[img_id]:
            obj = ET.SubElement(annotation, 'object')
            cat_name = categories[ann['category_id']]
            ET.SubElement(obj, 'name').text = cat_name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'

            bndbox = ET.SubElement(obj, 'bndbox')
            bbox = ann['bbox']
            ET.SubElement(bndbox, 'xmin').text = str(int(bbox[0]))
            ET.SubElement(bndbox, 'ymin').text = str(int(bbox[1]))
            ET.SubElement(bndbox, 'xmax').text = str(int(bbox[0] + bbox[2]))
            ET.SubElement(bndbox, 'ymax').text = str(int(bbox[1] + bbox[3]))

        pascal_voc_files.append((annotation, filename))

    return pascal_voc_files

# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

def main():
    st.title("Interactive Tool for Modifying and Analyzing Annotation Datasets")

    # State control
    if "datasets" not in st.session_state:
        st.session_state.datasets = []
    if "folder_paths" not in st.session_state:
        st.session_state.folder_paths = []
    if "dataset_format" not in st.session_state:
        st.session_state.dataset_format = None
    if "merged_data" not in st.session_state:
        st.session_state.merged_data = None

    # Step 1: Input Format
    st.header("Step 1: Select Input Format")
    dataset_format = st.selectbox(
        "Select the format of your datasets:",
        ["COCO", "Pascal VOC", "COCO Segmentation"]
    )
    st.session_state.dataset_format = dataset_format

    # Step 2: Upload Files
    st.header("Step 2: Upload Annotation Files")
    uploaded_files = st.file_uploader(
        f"Upload one or more annotation files in {dataset_format} format",
        type=['json', 'xml'],
        accept_multiple_files=True
    )

    # If files are uploaded, show prefix inputs
    if uploaded_files:
        st.subheader("Step 2B: (Optional) Specify image folder prefix for each dataset")
        st.markdown(
            "If the images for each dataset are in different folders, you can specify a prefix or path here "
            "to adjust the file_name (for COCO) or filename/path (for Pascal VOC)."
        )

        # Generate a text_input for each uploaded file
        for i, f in enumerate(uploaded_files):
            prefix_key = f"prefix_input_{i}"
            # Get the previous value from session_state, or "" if it doesn't exist
            default_val = st.session_state.get(prefix_key, "")

            # Create the text_input with the default value. Streamlit will handle its value.
            prefix_val = st.text_input(
                label=f"Image folder/prefix for Dataset {i+1} (file '{f.name}'):",
                value=default_val,  # initial value
                key=prefix_key      # key to 'remember' this value
            )

    # Load button
    if st.button("Load/Refresh Files") and uploaded_files:
        st.session_state.datasets.clear()
        st.session_state.folder_paths.clear()
        st.session_state.merged_data = None

        for i, file in enumerate(uploaded_files):
            prefix_key = f"prefix_input_{i}"
            folder_path = st.session_state.get(prefix_key, "")
            st.session_state.folder_paths.append(folder_path)

            # Load based on dataset_format
            if dataset_format in ["COCO", "COCO Segmentation"]:
                data = load_coco_annotations(file)
                if dataset_format == "COCO Segmentation":
                    data = convert_segmentation_to_bboxes_coco(data)

                # Adjust paths
                if folder_path:
                    for img in data["images"]:
                        old_file_name = img["file_name"].replace("\\", "/")
                        base_name = old_file_name.split("/")[-1]
                        new_file_name = force_forward_slash_join(folder_path, base_name)
                        img["file_name"] = new_file_name

                st.session_state.datasets.append(data)

            else:
                # Pascal VOC
                root = load_pascal_voc_annotations(file)
                if folder_path:
                    filename_node = root.find('filename')
                    path_node = root.find('path')
                    if filename_node is not None:
                        old_filename = filename_node.text.replace("\\", "/")
                        base_name = old_filename.split("/")[-1]
                        new_filename = force_forward_slash_join(folder_path, base_name)
                        filename_node.text = new_filename
                    if path_node is not None:
                        old_path = path_node.text.replace("\\", "/")
                        base_name = old_path.split("/")[-1]
                        new_path = force_forward_slash_join(folder_path, base_name)
                        path_node.text = new_path

                st.session_state.datasets.append(root)

        st.success("Files have been loaded/refreshed successfully!")

    # If datasets are loaded, perform analysis
    if len(st.session_state.datasets) > 0:
        st.header("Current Datasets Analysis")
        label_sets = []

        for idx, data in enumerate(st.session_state.datasets):
            if dataset_format in ["COCO", "COCO Segmentation"]:
                insights = analyze_coco_dataset(data)
                visualize_dataset_insights(insights, f"Dataset {idx+1}")
                label_sets.extend([cat['name'] for cat in data['categories']])
            else:  # Pascal VOC
                if data is not None:
                    insights = analyze_pascal_voc_dataset(data)
                    visualize_dataset_insights(insights, f"Dataset {idx+1}")
                    for obj in data.findall('object'):
                        label_sets.append(obj.find('name').text)

        # Step 3: Modify / Delete / Add in one step
        st.header("Step 3: Modify, Merge, Delete or Add Labels (simultaneously)")
        st.markdown("Mark any combination of these options, then click 'Apply Label Changes' to do it all at once:")
        modify_labels_opt = st.checkbox("Rename (modify) labels?")
        delete_labels_opt = st.checkbox("Delete labels?")
        add_new_labels_opt = st.checkbox("Add new labels without instances? (COCO only)")

        if modify_labels_opt or delete_labels_opt or add_new_labels_opt:
            label_set_unique = sorted(set(label_sets))
            st.info("Define your rename mapping, labels to delete, and/or new labels. Then click 'Apply Label Changes'.")

            # Deletion
            classes_to_delete = set()
            if delete_labels_opt:
                classes_to_delete = set(st.multiselect("Select labels to delete:", label_set_unique))

            # Renaming
            class_mapping = {}
            if modify_labels_opt:
                st.subheader("Rename Labels")
                for label in label_set_unique:
                    if label in classes_to_delete:
                        continue
                    new_label = st.text_input(f"New name for label '{label}':",
                                              value=label,
                                              key=f"rename_{label}")
                    new_label = sanitize_class_name(new_label)
                    if new_label:
                        class_mapping[label] = new_label
                    else:
                        st.warning(f"The new name for class '{label}' is empty or invalid. The original name will be retained.")
                        class_mapping[label] = label

            # Adding new classes (only COCO)
            new_labels = []
            if add_new_labels_opt and dataset_format in ["COCO", "COCO Segmentation"]:
                st.subheader("Add New Labels")
                st.markdown("Enter new classes to add (comma-separated):")
                input_new_labels = st.text_input("New classes:", value="", key="new_labels_input")
                if input_new_labels.strip():
                    proposed_new_labels = [lbl.strip() for lbl in input_new_labels.split(",") if lbl.strip()]
                    # Validate new classes
                    valid_new_labels = []
                    for lbl in proposed_new_labels:
                        lbl_sanitized = sanitize_class_name(lbl)
                        if not lbl_sanitized:
                            st.warning(f"Invalid or empty class name: '{lbl}'. It will be ignored.")
                            continue
                        if lbl_sanitized in label_set_unique:
                            continue
                        valid_new_labels.append(lbl_sanitized)
                    new_labels = valid_new_labels

            # Button to apply changes
            if st.button("Apply Label Changes"):
                newly_added_cats_overall = set()
                st.session_state.datasets_backup = copy.deepcopy(st.session_state.datasets)

                try:
                    if dataset_format in ["COCO", "COCO Segmentation"]:
                        for i, data in enumerate(st.session_state.datasets):
                            data = modify_labels_coco(
                                data,
                                class_mapping=class_mapping,
                                classes_to_delete=classes_to_delete,
                                new_classes=new_labels
                            )
                            st.session_state.datasets[i] = data
                            new_cat_names = {cat['name'] for cat in data.get('categories', [])}
                            added = new_cat_names - set(label_set_unique)
                            newly_added_cats_overall.update(added)
                    else:
                        # Pascal VOC
                        for i, root in enumerate(st.session_state.datasets):
                            if root is None:
                                continue
                            root = modify_labels_pascal_voc(
                                root,
                                class_mapping=class_mapping,
                                classes_to_delete=classes_to_delete,
                                new_classes=new_labels
                            )
                            st.session_state.datasets[i] = root
                            if root is not None:
                                new_cat_names = {obj.find('name').text for obj in root.findall('object')}
                                added = new_cat_names - set(label_set_unique)
                                newly_added_cats_overall.update(added)

                    if newly_added_cats_overall:
                        st.success("**New categories successfully added**: " + ", ".join(sorted(newly_added_cats_overall)))
                    else:
                        st.info("No truly new categories were added (or they might already exist).")

                    # Show "Updated Insights"
                    st.header("Updated Insights After Label Changes")
                    for idx, data in enumerate(st.session_state.datasets):
                        if dataset_format in ["COCO", "COCO Segmentation"] and data is not None:
                            insights = analyze_coco_dataset(data)
                            visualize_dataset_insights(insights, f"Dataset {idx+1} (Updated)")
                        elif dataset_format == "Pascal VOC" and data is not None:
                            insights = analyze_pascal_voc_dataset(data)
                            visualize_dataset_insights(insights, f"Dataset {idx+1} (Updated)")
                except Exception as e:
                    st.error(f"Error applying label changes: {e}")
                    st.session_state.datasets = st.session_state.datasets_backup  # Revert changes

        # 3B: Merge
        combine_datasets = False
        if len(st.session_state.datasets) > 1:
            st.header("Step 3B: (Optional) Combine Datasets")
            combine_datasets = st.checkbox("Do you want to combine the loaded datasets into a single one?", value=False)

        if combine_datasets:
            if dataset_format in ["COCO", "COCO Segmentation"]:
                merged_data = merge_coco_datasets(st.session_state.datasets)
                st.session_state.merged_data = merged_data
                combined_insights = analyze_coco_dataset(merged_data)
                st.header("Analysis of the Combined Dataset")
                visualize_dataset_insights(combined_insights, "Combined Dataset")
            else:
                merged_data = merge_pascal_voc_datasets(st.session_state.datasets)
                st.session_state.merged_data = merged_data
                total_images = len(merged_data)
                total_annotations = sum(len(r.findall('object')) for r in merged_data if r is not None)
                combined_insights = {
                    'num_images': total_images,
                    'num_annotations': total_annotations,
                    'instances_per_class': {},
                    'images_per_class': {}
                }
                class_counts = {}
                image_counts = {}
                for root in merged_data:
                    ins = analyze_pascal_voc_dataset(root)
                    for class_name, count in ins['instances_per_class'].items():
                        class_counts[class_name] = class_counts.get(class_name, 0) + count
                        image_counts[class_name] = image_counts.get(class_name, 0) + 1
                combined_insights['instances_per_class'] = class_counts
                combined_insights['images_per_class'] = image_counts

                st.header("Analysis of the Combined Dataset")
                visualize_dataset_insights(combined_insights, "Combined Dataset")
        else:
            st.session_state.merged_data = st.session_state.datasets

        # Step 4: Output Format
        st.header("Step 4: Select Output Format")
        output_format = st.selectbox("Select the output annotation format:", ["COCO", "Pascal VOC"])

        # Step 5: Download
        st.header("Step 5: Generate and Download Annotation File(s)")
        if st.button("Generate and Download Annotation File(s)"):
            final_datasets = st.session_state.merged_data
            if not final_datasets:
                st.warning("No datasets available to download.")
                return

            if combine_datasets:
                # Single combined dataset
                if output_format == "COCO":
                    if dataset_format in ["COCO", "COCO Segmentation"]:
                        if isinstance(final_datasets, dict):
                            out_data = save_coco_annotations(final_datasets)
                            st.download_button(
                                label="Download combined JSON",
                                data=out_data,
                                file_name="annotations_combined.json",
                                mime="application/json"
                            )
                            cat_names = [cat['name'] for cat in final_datasets['categories']]
                            st.write("**Categories in the combined annotations**:", cat_names)
                    else:  # Pascal VOC
                        if isinstance(final_datasets, list):
                            coco_data = convert_pascal_voc_to_coco(final_datasets)
                            out_data = save_coco_annotations(coco_data)
                            st.download_button(
                                label="Download combined JSON",
                                data=out_data,
                                file_name="annotations_combined.json",
                                mime="application/json"
                            )
                            cat_names = [cat['name'] for cat in coco_data['categories']]
                            st.write("**Categories in the combined annotations**:", cat_names)
                else:  # Pascal VOC
                    if dataset_format == "Pascal VOC":
                        if isinstance(final_datasets, list):
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                                for i, root in enumerate(final_datasets):
                                    if root is None:
                                        continue
                                    out_data = save_pascal_voc_annotations(root)
                                    file_name = f"annotation_{i+1}.xml"
                                    zip_file.writestr(file_name, out_data)
                            zip_buffer.seek(0)
                            st.download_button(
                                label="Download combined ZIP of XMLs",
                                data=zip_buffer,
                                file_name="annotations_combined.zip",
                                mime="application/zip"
                            )
                    else:  # dataset_format in ["COCO", "COCO Segmentation"]
                        if isinstance(final_datasets, dict):
                            pascal_voc_files = convert_coco_to_pascal_voc(final_datasets)
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                                for root, filename in pascal_voc_files:
                                    out_data = save_pascal_voc_annotations(root)
                                    xml_filename = filename.replace('.jpg', '.xml').replace('.png', '.xml')
                                    zip_file.writestr(xml_filename, out_data)
                            zip_buffer.seek(0)
                            st.download_button(
                                label="Download combined ZIP of XMLs",
                                data=zip_buffer,
                                file_name="annotations_combined.zip",
                                mime="application/zip"
                            )
            else:
                # Multiple datasets separately
                for idx, data in enumerate(final_datasets):
                    if data is None:
                        continue

                    if output_format == "COCO":
                        if dataset_format in ["COCO", "COCO Segmentation"]:
                            out_data = save_coco_annotations(data)
                            st.download_button(
                                label=f"Download dataset {idx+1} in COCO format",
                                data=out_data,
                                file_name=f"annotations_{idx+1}.json",
                                mime="application/json"
                            )
                            cat_names = [cat['name'] for cat in data['categories']]
                            st.write(f"**Categories in dataset {idx+1}**:", cat_names)

                        elif dataset_format == "Pascal VOC":
                            coco_data = convert_pascal_voc_to_coco([data])
                            out_data = save_coco_annotations(coco_data)
                            st.download_button(
                                label=f"Download dataset {idx+1} converted to COCO",
                                data=out_data,
                                file_name=f"annotations_{idx+1}.json",
                                mime="application/json"
                            )
                            cat_names = [cat['name'] for cat in coco_data['categories']]
                            st.write(f"**Categories in dataset {idx+1} (converted to COCO)**:", cat_names)
                    else:  # Pascal VOC
                        if dataset_format == "Pascal VOC":
                            out_data = save_pascal_voc_annotations(data)
                            st.download_button(
                                label=f"Download dataset {idx+1} in Pascal VOC format",
                                data=out_data,
                                file_name=f"annotation_{idx+1}.xml",
                                mime="application/xml"
                            )
                        else:  # dataset_format in ["COCO", "COCO Segmentation"]
                            pascal_voc_files = convert_coco_to_pascal_voc(data)
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                                for root, filename in pascal_voc_files:
                                    out_data = save_pascal_voc_annotations(root)
                                    xml_filename = filename.replace('.jpg', '.xml').replace('.png', '.xml')
                                    zip_file.writestr(xml_filename, out_data)
                            zip_buffer.seek(0)
                            st.download_button(
                                label=f"Download dataset {idx+1} converted to Pascal VOC",
                                data=zip_buffer,
                                file_name=f"annotations_{idx+1}.zip",
                                mime="application/zip"
                            )
    else:
        st.info("Please upload at least one annotation file and press 'Load/Refresh Files' to continue.")

if __name__ == "__main__":
    main()
