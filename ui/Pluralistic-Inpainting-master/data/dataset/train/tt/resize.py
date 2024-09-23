from PIL import Image
import os


def resize_images(directory):
    if not directory.endswith('/'):
        directory += '/'

    files = os.listdir(directory)

    for file in files:
        if file.endswith('.png'):
            with Image.open(directory + file) as img:
                resized_img = img.resize((256, 256))

                resized_img.save(directory + file)


if __name__ == '__main__':
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    resize_images(current_directory)
