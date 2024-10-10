import os


def get_txt_filenames(txt_folder):
    txt_filenames = set()
    for filename in os.listdir(txt_folder):
        if filename.endswith('.txt'):
            txt_filenames.add(os.path.splitext(filename)[0])
    return txt_filenames


def delete_unmatched_images_and_txts(image_folder, txt_filenames, txt_folder):
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_name = os.path.splitext(filename)[0]
            if image_name not in txt_filenames:
                image_path = os.path.join(image_folder, filename)
                os.remove(image_path)
                print(f"Deleted unmatched image: {image_path}")

    for filename in os.listdir(txt_folder):
        if filename.endswith('.txt'):
            txt_name = os.path.splitext(filename)[0]
            image_exists = any(
                os.path.exists(os.path.join(image_folder, txt_name + ext)) for ext in ['.png', '.jpg', '.jpeg'])
            if not image_exists:
                txt_path = os.path.join(txt_folder, filename)
                os.remove(txt_path)
                print(f"Deleted unmatched txt: {txt_path}")


def delete_images_and_txts_for_empty_txts(txt_folder, image_folder):
    for filename in os.listdir(txt_folder):
        if filename.endswith('.txt'):
            txt_path = os.path.join(txt_folder, filename)
            if os.path.getsize(txt_path) == 0:  # Check if the .txt file is empty
                txt_name = os.path.splitext(filename)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    image_path = os.path.join(image_folder, txt_name + ext)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"Deleted image for empty txt: {image_path}")
                os.remove(txt_path)
                print(f"Deleted empty txt: {txt_path}")


def main(txt_folder, image_folder):
    txt_filenames = get_txt_filenames(txt_folder)
    delete_unmatched_images_and_txts(image_folder, txt_filenames, txt_folder)
    delete_images_and_txts_for_empty_txts(txt_folder, image_folder)


if __name__ == "__main__":
    txt_folder = 'archive (1)/test/labels'  # Đổi thành đường dẫn tới thư mục chứa file txt của bạn
    image_folder = 'archive (1)/test/images'  # Đổi thành đường dẫn tới thư mục chứa file hình ảnh của bạn
    main(txt_folder, image_folder)