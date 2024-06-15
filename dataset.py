import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def augment_and_save_images(image_path, num_samples=5, output_size=(224, 224)):
    # Đọc hình ảnh gốc
    image = Image.open(image_path).convert('RGB')
    
    # Định nghĩa các phép biến đổi
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),             # Lật ngang ngẫu nhiên với xác suất 0.5
        transforms.RandomVerticalFlip(p=0.5),               # Lật dọc ngẫu nhiên với xác suất 0.5
        transforms.RandomRotation(degrees=45),              # Xoay ngẫu nhiên trong khoảng +/- 45 độ
        transforms.RandomResizedCrop(size=output_size,      # Phóng to ngẫu nhiên và cắt kích thước đầu ra
                                     scale=(0.8, 1.0),      # Tỉ lệ phóng to trong khoảng 80% đến 100%
                                     ratio=(0.75, 1.33)),   # Tỉ lệ khung hình trong khoảng 3/4 đến 4/3
        transforms.ColorJitter(brightness=0.5,              # Thay đổi độ sáng ngẫu nhiên
                               contrast=0.5,                # Thay đổi độ tương phản ngẫu nhiên
                               saturation=0.5,              # Thay đổi độ bão hòa ngẫu nhiên
                               hue=0.1)                     # Thay đổi tông màu ngẫu nhiên
    ])
    
    augmented_images = []

    # Tạo nhiều mẫu từ một ảnh gốc
    for _ in range(num_samples):
        augmented_image = transform(image)
        augmented_images.append(augmented_image)

    return augmented_images

image_path = '/mnt/data/image.png'
augmented_images = augment_and_save_images(image_path, num_samples=10, output_size=(224, 224))

# Hiển thị các hình ảnh gốc và các hình ảnh đã được biến đổi
plt.figure(figsize=(15, 5))

plt.subplot(1, 6, 1)
plt.imshow(Image.open(image_path).convert('RGB'))
plt.title('Original Image')
plt.axis('off')

for i, augmented_image in enumerate(augmented_images):
    plt.subplot(1, 6, i + 2)
    plt.imshow(augmented_image)
    plt.title(f'Augmented Image {i+1}')
    plt.axis('off')

plt.show()