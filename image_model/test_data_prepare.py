import os
import shutil
from pathlib import Path


def copy_images(source_dirs, dest_dir, max_images=10000):
    """
    Copies up to max_images from the source_dirs into the dest_dir.

    :param source_dirs: List of directories to collect images from.
    :param dest_dir: Directory to copy the images to.
    :param max_images: Maximum number of images to copy per source directory.
    """

    os.makedirs(dest_dir, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png"}

    for source_dir in source_dirs:
        images_copied = 0
        if not os.path.isdir(source_dir):
            print(f"Skipping non-existent directory: {source_dir}")
            continue
        print(f"Copying images from {source_dir} to {dest_dir}...")
        # Iterate through files in the directory
        for file_name in os.listdir(source_dir):
            # Only taking test imgaes from celebA (based on partition provided by dataset authors)
            if (
                source_dir == "datasets/dffd/img_align_celeba"
                and int(file_name.split(".")[0]) < 190000
            ):
                continue
            if images_copied >= max_images:
                print(f"Reached the maximum limit of {max_images} images.")
                break

            file_path = os.path.join(source_dir, file_name)
            if (
                os.path.isfile(file_path)
                and Path(file_path).suffix.lower() in img_extensions
            ):
                try:
                    dest_path = os.path.join(dest_dir, file_name)
                    shutil.copy(file_path, dest_path)
                    images_copied += 1
                except Exception as e:
                    print(f"Error copying {file_path}: {e}")

    print(f"Copied {images_copied} images to {dest_dir}.")


source_directories = ["datasets/dffd/pggan_v2/test", "datasets/dffd/img_align_celeba"]
destination_directory = "datasets/test"
max_images = 10000

copy_images(source_directories, destination_directory, max_images)
