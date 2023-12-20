import os
import lmdb
import cv2


def create_lmdb_dataset(root_dir, output_path):
    env = lmdb.open(output_path, map_size=int(1e12))

    with env.begin(write=True) as txn:
        dataset_folders = os.listdir(root_dir)
        num_datasets = len(dataset_folders)
        print(f"Total datasets: {num_datasets}")
        import ipdb;ipdb.set_trace()
        for i, dataset_folder in enumerate(dataset_folders):
            dataset_folder_path = os.path.join(root_dir, dataset_folder)
            if not os.path.isdir(dataset_folder_path):
                continue

            print(f"\nProcessing dataset {i+1}/{num_datasets}: {dataset_folder}")

            class_folders = os.listdir(dataset_folder_path)
            num_classes = len(class_folders)
            print(f"Total classes: {num_classes}")

            for j, class_folder in enumerate(class_folders):
                class_folder_path = os.path.join(dataset_folder_path, class_folder)
                if not os.path.isdir(class_folder_path):
                    continue
                print(f"\nProcessing class {j+1}/{num_classes}: {class_folder}")

                image_files = os.listdir(class_folder_path)
                num_images = len(image_files)
                print(f"Total images: {num_images}")

                for k, image_file in enumerate(image_files):
                    image_path = os.path.join(class_folder_path, image_file)
                    image = cv2.imread(image_path)
                    datum = cv2.imencode('.tiff', image)[1].tobytes()
                    key = f"{dataset_folder}_{class_folder}_{image_file}".encode('ascii')
                    label_key = f"{dataset_folder}_{class_folder}_{image_file}_label".encode('ascii')
                    txn.put(key, datum, overwrite=False)
                    txn.put(label_key, str(class_label).encode())
                    print(f"Processed image {k+1}/{num_images}", end='\r')

    env.close()

# 使用示例
root_dir = 'data'  # 指定根目录，包含5个数据集的文件夹
output_path = 'lmdb'  # 指定输出的LMDB数据库路径
create_lmdb_dataset(root_dir, output_path)
