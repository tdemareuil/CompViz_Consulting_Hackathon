import os
import random
import re
import shutil
from typing import List, Tuple

import albumentations as A
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from tqdm.notebook import tqdm


IMAGES_PATH = "/kaggle/input/11-xhec/images"
LABELS_PATH = "/kaggle/working/chronsite_images/labels/train"

FORMATTED_IMAGES_PATH = "/kaggle/working/chronsite_images_formatted/images/train"
FORMATTED_LABELS_PATH = "/kaggle/working/chronsite_images_formatted/labels/train"


def make_arborescence(labels_path: str, fmt_images_path: str, fmt_labels_path: str, test_output_path: str) -> None:
    """Create folders for YOLO-like organisation"""
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
        os.makedirs("/kaggle/working/chronsite_images/images/train/")

    if not os.path.exists(fmt_images_path):
        os.makedirs(fmt_images_path)
        os.makedirs(fmt_labels_path)
        
    os.makedirs(test_output_path)

    print("Created folders organization\n")


def write_labels_to_files(csv_labels_path: str) -> None:
    """Converts the dataframe with annotations on each line
    to one text file per image (YOLO requirement)
    """
    labels_df = pd.read_csv(csv_labels_path)

    for image in labels_df["img_id"].unique():
        values_to_write = labels_df[labels_df["img_id"] == image].values

        with open(LABELS_PATH + "/" + image.split(".")[0] + ".txt", "a") as f:
            for object_annot in values_to_write:
                annotation = str(object_annot[2:])[1:-1]
                f.write(annotation + "\n")

    print("Labels written from csv to .txt files\n")


def write_yolo_train_config(config_file_path: str, fmt_image_path: str) -> None:
    """ Write the yolo train config file with the right directories for data
    """
    config_yaml_str = f"""
    # train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
    train: {fmt_image_path}
    val: {"/" + os.path.join(*fmt_image_path.split("/")[:-1] + ["val"])}

    # number of classes
    nc: 2

    # class names
    names: ['People', 'Mixer_truck']
    """

    with open(config_file_path, "w") as f:
        f.write(config_yaml_str)

    print(f"Config written: \n {config_yaml_str} \n")


def clone_yolo_repo(yolo_git_url="https://github.com/ultralytics/yolov5") -> None:
    """Clone the yolov5 repo from Github"""
    os.system(f"git clone -q {yolo_git_url}")
    print("Yolo repo cloned\n")


def format_images(resized_height: int, max_img_dimension: int) -> A.core.composition.Compose:
    """Resizes and pad images so that all images are squared and have the same dimension"""
    return A.Compose(
        [
            A.Resize(height=resized_height,
                     width=max_img_dimension,
                     p=1.0),
            A.PadIfNeeded(min_height=max_img_dimension,
                          min_width=max_img_dimension,
                          border_mode=cv2.BORDER_CONSTANT,
                          p=1.0)
        ],
        p=1.0,
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])
    )


def load_and_process_bboxes(test_lbl: str, lbl_path=LABELS_PATH) -> Tuple[List, List]:
    """Read the .txt file with annotations and convert it to
    a Python list of annotations and list of classes"""
    with open(os.path.join(lbl_path, test_lbl), "r") as f:
        bboxes_raw = f.read().split('\n')[:-1]

    bboxes = [list(map(float, b.split()[1:])) for b in bboxes_raw]
    category_ids = [int(b.split()[0]) for b in bboxes_raw]

    return bboxes, category_ids


def get_corresponding_label_path(image_path: str, labels_path=LABELS_PATH):
    """Retrieves the label path from the image path"""
    return [p for p in os.listdir(labels_path) if image_path.split('.')[0] == p.split('.')[0]]


def get_desired_height(image: np.ndarray, max_img_dimension: int) -> int:
    """Computes the size of the image to make it square
    without distorting it"""
    return int(image.shape[0] * (max_img_dimension / image.shape[1]))


def write_annotation(fmt_label_path: str, label_path: str, is_any_annotation: bool, formatted: dict) -> None:
    """"""
    with open(os.path.join(fmt_label_path, label_path.split('.')[0] + '.txt'), "w") as text_file:
        if is_any_annotation:
            for bbox, categ in zip(formatted["bboxes"], formatted["category_ids"]):
                bbox_rounded = [str(round(c, 8)) for c in bbox]

                formatted_annotation = " ".join(
                    [str(categ), *bbox_rounded]) + "\n"
                text_file.write(formatted_annotation)
        else:
            text_file.write("")  # no object on the picture


def format_images_and_labels(n_images: int, max_img_dimension: int, images_path: str, fmt_label_path: str, fmt_image_path: str) -> None:
    """Processes the images to make them uniform and updates the corresponding annotations"""

    for k, image_path in tqdm(enumerate(os.listdir(images_path)), leave=False):
        img = plt.imread(os.path.join(images_path, image_path))
        label_path = get_corresponding_label_path(image_path)
        escaped_images = 0

        try:
            is_any_annotation = bool(len(label_path) > 0)
            if is_any_annotation:
                label_path = label_path[0]
                bboxes, category_ids = load_and_process_bboxes(label_path)
            else:
                label_path = image_path.split(".")[0] + ".txt"
                bboxes = []
                category_ids = []

            desired_height = get_desired_height(img, max_img_dimension)
            formatted = format_images(resized_height=desired_height, max_img_dimension=max_img_dimension)(
                image=img, bboxes=bboxes, category_ids=category_ids)

            # Write image to disk in the right folder
            imageio.imwrite(os.path.join(
                fmt_image_path, image_path), formatted["image"])

            # Write annotation to disk
            write_annotation(fmt_label_path, label_path,
                             is_any_annotation, formatted)

        except ValueError:
            escaped_images += 1

        if n_images:
            if k > n_images:
                break

    print(f"{k} images and labels written to disk")
    print(f"{escaped_images} images escaped because of Albumentations' bug\n")


def make_val_folders(images_path: str, labels_path: str) -> Tuple[str, str]:
    """Create the relevant folders for YOLO-like organisation
    """
    val_images_path = "/" + \
        os.path.join(*images_path.split("/")[:-1] + ["val"])
    val_labels_path = "/" + \
        os.path.join(*labels_path.split("/")[:-1] + ["val"])

    if not os.path.exists(val_images_path):
        os.makedirs(val_images_path)
        os.makedirs(val_labels_path)

    print("Created validation folders\n")

    return (val_images_path, val_labels_path)


def create_val_set(test_size: float, images_path: str, labels_path: str) -> None:
    """Divides the total set in train and val sets
    Equivalent to a train-test split
    """
    # Create validation folders
    val_images_path, val_labels_path = make_val_folders(
        images_path, labels_path)

    # Select and move files
    base_paths = os.listdir(images_path)

    test_images_path = random.sample(
        base_paths, int(len(base_paths) * test_size))
    test_images_path = [p.split(".")[0] for p in test_images_path]

    for path_to_move in test_images_path:
        # move images
        old_train_path = os.path.join(images_path, path_to_move + ".jpg")
        new_val_path = os.path.join(val_images_path, path_to_move + ".jpg")
        os.rename(old_train_path, new_val_path)

        # move labels
        old_train_path = os.path.join(labels_path, path_to_move + ".txt")
        new_val_path = os.path.join(val_labels_path, path_to_move + ".txt")
        os.rename(old_train_path, new_val_path)

    print(f"Validation set created with {test_size:.0%} of initial images!")


def augmentation_function(n_augment: int) -> A.core.composition.Compose:
    """Performs data augmentation on images with realistic parameters
    Returns a Albumentation composition function"""
    proba_transfo = 1 / n_augment

    def random_proba(proba_transfo: float) -> float:
        return max(0.4, min(1, proba_transfo + (2 * np.random.random() - 1) / 3))

    return A.Compose(
        [
            A.HorizontalFlip(p=1),
            A.Rotate(limit=25, p=random_proba(proba_transfo)),
            A.RandomBrightness(limit=0.1, p=random_proba(proba_transfo)),
            A.RandomSnow(brightness_coeff=0.95, snow_point_lower=0.1,
                         snow_point_upper=0.3, p=random_proba(proba_transfo)),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4,
                        alpha_coef=0.1, p=random_proba(proba_transfo)),

        ],
        p=1.0,
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])
    )


def augment_train_set(path_images_train: str, path_label_train: str, n_augment: int) -> None:
    escaped_aug_image = 0

    for i in range(n_augment):
        for k, image_path in tqdm(enumerate(os.listdir(path_images_train))):
            if "aug_" not in image_path:
                try:
                    img = plt.imread(os.path.join(
                        path_images_train, image_path))
                    label_path = get_corresponding_label_path(image_path)

                    is_any_annotation = bool(len(label_path) > 0)
                    if is_any_annotation:
                        label_path = label_path[0]
                        bboxes, category_ids = load_and_process_bboxes(
                            label_path, path_label_train)
                    else:
                        label_path = image_path.split(".")[0] + ".txt"
                        bboxes = []
                        category_ids = []

                    augmented = augmentation_function(n_augment=n_augment)(
                        image=img, bboxes=bboxes, category_ids=category_ids)

                    # Write image to disk in the right folder
                    imageio.imwrite(os.path.join(
                        path_images_train, f"aug_{i}_{image_path}"), augmented["image"])

                    # Write annotation to disk
                    write_annotation(path_label_train, f"aug_{i}_{label_path}",
                                     is_any_annotation, augmented)

                except ValueError:
                    escaped_aug_image += 1

    print(f"{escaped_aug_image} images not augmented because of bug")


def format_test_images(test_images_path: str, test_output_path: str, max_img_dimension: int) -> None:
    
    for k, image_path in tqdm(enumerate(os.listdir(test_images_path))):
        img = plt.imread(os.path.join(test_images_path, image_path))
        
        desired_height = get_desired_height(img, max_img_dimension)
        augmented = format_images(resized_height=desired_height, max_img_dimension=max_img_dimension)(
                        image=img, bboxes=[], category_ids=[])
        
        imageio.imwrite(os.path.join(test_output_path, f"{image_path}"), augmented["image"])


def gather_predictions(txt_output_path: str) -> pd.DataFrame:
    text_files_paths = [p for p in os.listdir(txt_output_path) if ".txt" in p]
    
    total_objects_detected = dict()
    for text_file_path in text_files_paths:
        with open(os.path.join(txt_output_path, text_file_path)) as f:
            for i, line in enumerate(f.read().split("\n")):
                if line:
                    total_objects_detected[f"{text_file_path}_{i}"] = line.split()
                    
    df_objects_detected = pd.DataFrame(total_objects_detected).T
    
    col_types = {"category": int, "x_center": float, "y_center": float, "width": float, "height": float}
    df_objects_detected.columns = list(col_types.keys())
    df_objects_detected = df_objects_detected.astype(col_types)

    return df_objects_detected



def gather_predictions(txt_output_path: str) -> pd.DataFrame:
    text_files_paths = [p for p in os.listdir(txt_output_path) if ".txt" in p]
    
    total_objects_detected = dict()
    for text_file_path in text_files_paths:
        with open(os.path.join(txt_output_path, text_file_path)) as f:
            for i, line in enumerate(f.read().split("\n")):
                if line:
                    total_objects_detected[f"{text_file_path}_{i}"] = line.split()
                    
    df_objects_detected = pd.DataFrame(total_objects_detected).T
    
    col_types = {"category": int, "x_center": float, "y_center": float, "width": float, "height": float}
    df_objects_detected.columns = list(col_types.keys())
    df_objects_detected = df_objects_detected.astype(col_types)

    return df_objects_detected


def from_yolo_to_cor(box, shape):
    img_h, img_w, _ = shape
    x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    return x1, y1, x2, y2


def draw_boxes(img, boxes, categories, mapping):
    draw = ImageDraw.Draw(img)
    
    for box, categ in zip(boxes, categories):
        x1, y1, x2, y2 = from_yolo_to_cor(box, np.asarray(img).shape)
        draw.rectangle((x1, y1, x2, y2), outline="green")
        draw.text((x1, y1), f"GT: {mapping[categ]}")

    display(img)
