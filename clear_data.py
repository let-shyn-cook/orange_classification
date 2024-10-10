import os

def get_txt_filenames(txt_folder):
    txt_filenames = set()
    for filename in os.listdir(txt_folder):
        if filename.endswith('.txt'):
            txt_filenames.add(os.path.splitext(filename)[0])
    return txt_filenames

def delete_unmatched_images(image_folder, txt_filenames):
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_name = os.path.splitext(filename)[0]
            if image_name not in txt_filenames:
                image_path = os.path.join(image_folder, filename)
                os.remove(image_path)
                print(f"Deleted: {image_path}")

def main(txt_folder, image_folder):
    txt_filenames = get_txt_filenames(txt_folder)
    delete_unmatched_images(image_folder, txt_filenames)

if __name__ == "__main__":
    txt_folder = 'zz.v2i.yolov8/train/labels'  # Đổi thành đường dẫn tới thư mục chứa file txt của bạn
    image_folder = 'zz.v2i.yolov8/train/images'  # Đổi thành đường dẫn tới thư mục chứa ảnh của bạn
    main(txt_folder, image_folder)
