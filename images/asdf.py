from PIL import Image, ImageOps
import os


def resize_images_in_folder(folder_path, new_width):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            full_path = os.path.join(folder_path, filename)
            with Image.open(full_path) as img:
                # Calculate the new height to maintain the aspect ratio
                width_percent = new_width / float(img.size[0])
                new_height = int((float(img.size[1]) * float(width_percent)))

                # Resize the image using LANCZOS resampling
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save the image, overwriting the existing file
                img.save(full_path)
                print(f"Resized {filename} to {new_width}x{new_height}")


# Example usage
folder_path = "../images"  # Change this to the path of your images
new_width = 1000  # Set the new width you want for all images
resize_images_in_folder(folder_path, new_width)
