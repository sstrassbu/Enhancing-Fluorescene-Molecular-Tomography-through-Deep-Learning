import cv2
import numpy as np
import os

def combine_images_2folder(folder1_path, folder2_path, output_path):
    #images_folder1 = os.listdir(folder1_path)
    #images_folder2 = os.listdir(folder2_path)
    images_folder1 = [f for f in os.listdir(folder1_path) if f.endswith('.png')]
    images_folder2 = [f for f in os.listdir(folder2_path) if f.endswith('.png')]
    print(images_folder1)
    print(images_folder2)

    combined_image = None
    for i, (image1, image2) in enumerate(zip(images_folder1, images_folder2)):
        path_image1 = os.path.join(folder1_path, image1)
        path_image2 = os.path.join(folder2_path, image2)

        img1 = cv2.imread(path_image1)
        img2 = cv2.imread(path_image2)

        spacing = np.full((2, img1.shape[1], 3), 255, dtype=np.uint8)
        img1_with_spacing = np.vstack([spacing, img1, spacing])
        img2_with_spacing = np.vstack([spacing, img2, spacing])
        spacing_col = np.full((max(img1_with_spacing.shape[0], img2_with_spacing.shape[0]), 2, 3), 255, dtype=np.uint8)

        if combined_image is None:
            combined_image = np.hstack([img1_with_spacing, spacing_col, img2_with_spacing])
        else:
            combined_image = np.vstack([combined_image, np.hstack([img1_with_spacing, spacing_col, img2_with_spacing])])

    cv2.imwrite(output_path, combined_image)


def combine_images_4folder(folder1_path, folder2_path, folder3_path, folder4_path, output_path):
    images_folder1 = [f for f in os.listdir(folder1_path) if f.endswith('.png')]
    images_folder2 = [f for f in os.listdir(folder2_path) if f.endswith('.png')]
    images_folder3 = [f for f in os.listdir(folder3_path) if f.endswith('.png')]
    images_folder4 = [f for f in os.listdir(folder4_path) if f.endswith('.png')]
    print(images_folder1)
    print(images_folder2)

    combined_image = None
    for i, (image1, image2, image3, image4) in enumerate(zip(images_folder1, images_folder2, images_folder3, images_folder4)):
        path_image1 = os.path.join(folder1_path, image1)
        path_image2 = os.path.join(folder2_path, image2)
        path_image3 = os.path.join(folder3_path, image3)
        path_image4 = os.path.join(folder4_path, image4)

        img1 = cv2.imread(path_image1)
        img2 = cv2.imread(path_image2)
        img3 = cv2.imread(path_image3)
        img4 = cv2.imread(path_image4)

        spacing = np.full((2, img1.shape[1], 3), 255, dtype=np.uint8)
        img1_with_spacing = np.vstack([spacing, img1, spacing])
        img2_with_spacing = np.vstack([spacing, img2, spacing])
        img3_with_spacing = np.vstack([spacing, img3, spacing])
        img4_with_spacing = np.vstack([spacing, img4, spacing])
        spacing_col = np.full((max(img1_with_spacing.shape[0], img2_with_spacing.shape[0]), 2, 3), 255, dtype=np.uint8)

        if combined_image is None:
            combined_image = np.hstack([img1_with_spacing, spacing_col, img2_with_spacing, spacing_col, img3_with_spacing, spacing_col, img4_with_spacing])
        else:
            combined_image = np.vstack([combined_image, np.hstack([img1_with_spacing, spacing_col, img2_with_spacing, spacing_col, img3_with_spacing, spacing_col, img4_with_spacing])])

    cv2.imwrite(output_path, combined_image)

def combine_images_6folder(folder1_path, folder2_path, folder3_path, folder4_path, folder5_path, folder6_path, label1, label2, label3, label4, label5, label6, output_path):
    images_folder1 = [f for f in os.listdir(folder1_path) if f.endswith('.png')]
    images_folder2 = [f for f in os.listdir(folder2_path) if f.endswith('.png')]
    images_folder3 = [f for f in os.listdir(folder3_path) if f.endswith('.png')]
    images_folder4 = [f for f in os.listdir(folder4_path) if f.endswith('.png')]
    images_folder5 = [f for f in os.listdir(folder5_path) if f.endswith('.png')]
    images_folder6 = [f for f in os.listdir(folder6_path) if f.endswith('.png')]

    combined_image = None
    for i, (image1, image2, image3, image4, image5, image6) in enumerate(zip(images_folder1, images_folder2, images_folder3, images_folder4, images_folder5, images_folder6)):
        path_image1 = os.path.join(folder1_path, image1)
        path_image2 = os.path.join(folder2_path, image2)
        path_image3 = os.path.join(folder3_path, image3)
        path_image4 = os.path.join(folder4_path, image4)
        path_image5 = os.path.join(folder5_path, image5)
        path_image6 = os.path.join(folder6_path, image6)

        img1 = cv2.imread(path_image1)
        img2 = cv2.imread(path_image2)
        img3 = cv2.imread(path_image3)
        img4 = cv2.imread(path_image4)
        img5 = cv2.imread(path_image5)
        img6 = cv2.imread(path_image6)

        spacing = np.full((2, img1.shape[1], 3), 255, dtype=np.uint8)
        img1_with_spacing = np.vstack([spacing, img1, spacing])
        img2_with_spacing = np.vstack([spacing, img2, spacing])
        img3_with_spacing = np.vstack([spacing, img3, spacing])
        img4_with_spacing = np.vstack([spacing, img4, spacing])
        img5_with_spacing = np.vstack([spacing, img5, spacing])
        img6_with_spacing = np.vstack([spacing, img6, spacing])
        spacing_col = np.full((max(img1_with_spacing.shape[0], img2_with_spacing.shape[0]), 2, 3), 255, dtype=np.uint8)

        if combined_image is None:
            combined_image = np.hstack([img1_with_spacing, spacing_col, img2_with_spacing, spacing_col, img3_with_spacing, spacing_col, img4_with_spacing, spacing_col, img5_with_spacing, spacing_col, img6_with_spacing])
        else:
            combined_image = np.vstack([combined_image, np.hstack([img1_with_spacing, spacing_col, img2_with_spacing, spacing_col, img3_with_spacing, spacing_col, img4_with_spacing, spacing_col, img5_with_spacing, spacing_col, img6_with_spacing])])

    cv2.imwrite(output_path, combined_image)


def combine_images_4folder_with_labels(folder1_path, folder2_path, folder3_path, folder4_path, label1, label2, label3, label4, output_path):
    images_folder1 = [f for f in os.listdir(folder1_path) if f.endswith('.png')]
    images_folder2 = [f for f in os.listdir(folder2_path) if f.endswith('.png')]
    images_folder3 = [f for f in os.listdir(folder3_path) if f.endswith('.png')]
    images_folder4 = [f for f in os.listdir(folder4_path) if f.endswith('.png')]

    combined_image = None
    for i, (image1, image2, image3, image4) in enumerate(zip(images_folder1, images_folder2, images_folder3, images_folder4)):
        path_image1 = os.path.join(folder1_path, image1)
        path_image2 = os.path.join(folder2_path, image2)
        path_image3 = os.path.join(folder3_path, image3)
        path_image4 = os.path.join(folder4_path, image4)

        img1 = cv2.imread(path_image1)
        img2 = cv2.imread(path_image2)
        img3 = cv2.imread(path_image3)
        img4 = cv2.imread(path_image4)

        spacing = np.full((2, img1.shape[1], 3), 255, dtype=np.uint8)
        img1_with_spacing = np.vstack([spacing, img1, spacing])
        img2_with_spacing = np.vstack([spacing, img2, spacing])
        img3_with_spacing = np.vstack([spacing, img3, spacing])
        img4_with_spacing = np.vstack([spacing, img4, spacing])
        spacing_col = np.full((max(img1_with_spacing.shape[0], img2_with_spacing.shape[0]), 2, 3), 255, dtype=np.uint8)

        if combined_image is None:
            combined_image = np.hstack([img1_with_spacing, spacing_col, img2_with_spacing, spacing_col, img3_with_spacing, spacing_col, img4_with_spacing])
        else:
            combined_image = np.vstack([combined_image, np.hstack([img1_with_spacing, spacing_col, img2_with_spacing, spacing_col, img3_with_spacing, spacing_col, img4_with_spacing])])

    combined_image = np.vstack([combined_image, np.full((10, combined_image.shape[1], 3), 255, dtype=np.uint8)])

    # Add labels at the bottom
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_font_scale = 0.3
    label_thickness = 1
    label_color = (0, 0, 0)  # Black color for the labels

    # Calculate label position
    label_height = 20
    label_width = combined_image.shape[1] // 4  # Divide the width equally for each label
    label_pos1 = (0, combined_image.shape[0] - label_height + 15)
    label_pos2 = (label_width * 1, combined_image.shape[0] - label_height + 15)
    label_pos3 = (label_width * 2, combined_image.shape[0] - label_height + 15)
    label_pos4 = (label_width * 3, combined_image.shape[0] - label_height + 15)

    # Add labels
    cv2.putText(combined_image, label1, label_pos1, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label2, label_pos2, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label3, label_pos3, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label4, label_pos4, label_font, label_font_scale, label_color, label_thickness)
    

    cv2.imwrite(output_path, combined_image)

def combine_images_5folder_with_labels(folder1_path, folder2_path, folder3_path, folder4_path, folder5_path, label1, label2, label3, label4, label5, output_path):
    images_folder1 = [f for f in os.listdir(folder1_path) if f.endswith('.png')]
    images_folder2 = [f for f in os.listdir(folder2_path) if f.endswith('.png')]
    images_folder3 = [f for f in os.listdir(folder3_path) if f.endswith('.png')]
    images_folder4 = [f for f in os.listdir(folder4_path) if f.endswith('.png')]
    images_folder5 = [f for f in os.listdir(folder5_path) if f.endswith('.png')]

    combined_image = None
    for i, (image1, image2, image3, image4, image5) in enumerate(zip(images_folder1, images_folder2, images_folder3, images_folder4, images_folder5)):
        path_image1 = os.path.join(folder1_path, image1)
        path_image2 = os.path.join(folder2_path, image2)
        path_image3 = os.path.join(folder3_path, image3)
        path_image4 = os.path.join(folder4_path, image4)
        path_image5 = os.path.join(folder5_path, image5)

        img1 = cv2.imread(path_image1)
        img2 = cv2.imread(path_image2)
        img3 = cv2.imread(path_image3)
        img4 = cv2.imread(path_image4)
        img5 = cv2.imread(path_image5)

        spacing = np.full((2, img1.shape[1], 3), 255, dtype=np.uint8)
        img1_with_spacing = np.vstack([spacing, img1, spacing])
        img2_with_spacing = np.vstack([spacing, img2, spacing])
        img3_with_spacing = np.vstack([spacing, img3, spacing])
        img4_with_spacing = np.vstack([spacing, img4, spacing])
        img5_with_spacing = np.vstack([spacing, img5, spacing])
        spacing_col = np.full((max(img1_with_spacing.shape[0], img2_with_spacing.shape[0]), 2, 3), 255, dtype=np.uint8)

        if combined_image is None:
            combined_image = np.hstack([img1_with_spacing, spacing_col, img2_with_spacing, spacing_col, img3_with_spacing, spacing_col, img4_with_spacing, spacing_col, img5_with_spacing])
        else:
            combined_image = np.vstack([combined_image, np.hstack([img1_with_spacing, spacing_col, img2_with_spacing, spacing_col, img3_with_spacing, spacing_col, img4_with_spacing, spacing_col, img5_with_spacing])])

    combined_image = np.vstack([combined_image, np.full((10, combined_image.shape[1], 3), 255, dtype=np.uint8)])

    # Add labels at the bottom
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_font_scale = 0.3
    label_thickness = 1
    label_color = (0, 0, 0)  # Black color for the labels

    # Calculate label position
    label_height = 20
    label_width = combined_image.shape[1] // 5  # Divide the width equally for each label
    label_pos1 = (0, combined_image.shape[0] - label_height + 15)
    label_pos2 = (label_width * 1, combined_image.shape[0] - label_height + 15)
    label_pos3 = (label_width * 2, combined_image.shape[0] - label_height + 15)
    label_pos4 = (label_width * 3, combined_image.shape[0] - label_height + 15)
    label_pos5 = (label_width * 4, combined_image.shape[0] - label_height + 15)

    # Add labels
    cv2.putText(combined_image, label1, label_pos1, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label2, label_pos2, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label3, label_pos3, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label4, label_pos4, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label5, label_pos5, label_font, label_font_scale, label_color, label_thickness)

    cv2.imwrite(output_path, combined_image)

def combine_images_6folder_with_labels(folder1_path, folder2_path, folder3_path, folder4_path, folder5_path, folder6_path, label1, label2, label3, label4, label5, label6, output_path):
    images_folder1 = [f for f in os.listdir(folder1_path) if f.endswith('.png')]
    print(images_folder1)
    images_folder1.sort()
    images_folder2 = [f for f in os.listdir(folder2_path) if f.endswith('.png')]
    images_folder2.sort()
    images_folder3 = [f for f in os.listdir(folder3_path) if f.endswith('.png')]
    images_folder3.sort()
    images_folder4 = [f for f in os.listdir(folder4_path) if f.endswith('.png')]
    images_folder4.sort()
    images_folder5 = [f for f in os.listdir(folder5_path) if f.endswith('.png')]
    images_folder5.sort()
    images_folder6 = [f for f in os.listdir(folder6_path) if f.endswith('.png')]
    images_folder6.sort()
    print('sort', images_folder1)
    print(images_folder2)
    print(images_folder3)
    print(images_folder4)
    print(images_folder5)
    print(images_folder6)

    combined_image = None
    for i, (image1, image2, image3, image4, image5, image6) in enumerate(zip(images_folder1, images_folder2, images_folder3, images_folder4, images_folder5, images_folder6)):
        path_image1 = os.path.join(folder1_path, image1)
        path_image2 = os.path.join(folder2_path, image2)
        path_image3 = os.path.join(folder3_path, image3)
        path_image4 = os.path.join(folder4_path, image4)
        path_image5 = os.path.join(folder5_path, image5)
        path_image6 = os.path.join(folder6_path, image6)

        img1 = cv2.imread(path_image1)
        img2 = cv2.imread(path_image2)
        img3 = cv2.imread(path_image3)
        img4 = cv2.imread(path_image4)
        img5 = cv2.imread(path_image5)
        img6 = cv2.imread(path_image6)

        spacing = np.full((2, img1.shape[1], 3), 255, dtype=np.uint8)
        img1_with_spacing = np.vstack([spacing, img1, spacing])
        img2_with_spacing = np.vstack([spacing, img2, spacing])
        img3_with_spacing = np.vstack([spacing, img3, spacing])
        img4_with_spacing = np.vstack([spacing, img4, spacing])
        img5_with_spacing = np.vstack([spacing, img5, spacing])
        img6_with_spacing = np.vstack([spacing, img6, spacing])
        spacing_col = np.full((max(img1_with_spacing.shape[0], img2_with_spacing.shape[0]), 2, 3), 255, dtype=np.uint8)

        if combined_image is None:
            combined_image = np.hstack([img1_with_spacing, spacing_col, img2_with_spacing, spacing_col, img3_with_spacing, spacing_col, img4_with_spacing, spacing_col, img5_with_spacing, spacing_col, img6_with_spacing])
        else:
            combined_image = np.vstack([combined_image, np.hstack([img1_with_spacing, spacing_col, img2_with_spacing, spacing_col, img3_with_spacing, spacing_col, img4_with_spacing, spacing_col, img5_with_spacing, spacing_col, img6_with_spacing])])

    combined_image = np.vstack([combined_image, np.full((10, combined_image.shape[1], 3), 255, dtype=np.uint8)])

    # Add labels at the bottom
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_font_scale = 0.3
    label_thickness = 1
    label_color = (0, 0, 0)  # Black color for the labels

    # Calculate label position
    label_height = 20
    label_width = combined_image.shape[1] // 6  # Divide the width equally for each label
    label_pos1 = (0, combined_image.shape[0] - label_height + 15)
    label_pos2 = (label_width * 1, combined_image.shape[0] - label_height + 15)
    label_pos3 = (label_width * 2, combined_image.shape[0] - label_height + 15)
    label_pos4 = (label_width * 3, combined_image.shape[0] - label_height + 15)
    label_pos5 = (label_width * 4, combined_image.shape[0] - label_height + 15)
    label_pos6 = (label_width * 5, combined_image.shape[0] - label_height + 15)

    # Add labels
    cv2.putText(combined_image, label1, label_pos1, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label2, label_pos2, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label3, label_pos3, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label4, label_pos4, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label5, label_pos5, label_font, label_font_scale, label_color, label_thickness)
    cv2.putText(combined_image, label6, label_pos6, label_font, label_font_scale, label_color, label_thickness)

    cv2.imwrite(output_path, combined_image)

# Example usage
#combine_images_4folder("double/output_cleanBLI38", "double/output_cleanCT39", "double/output_noiseBLI-n41", "double/output_noiseCT-n40", "double/combined_image4.png")
#combine_images_4folder_with_labels("double/output_cleanBLI38", "double/output_cleanCT39", "double/output_noiseBLI-n41", "double/output_noiseCT-n40", "BLI", "CT", "noiseBLI", "noiseCT", "combined_image_with_labels4.png")
#combine_images_5folder_with_labels("double/imagesTs_clean","double/output_cleanBLI38", "double/output_cleanCT39", "double/output_noiseBLI-n41", "double/output_noiseCT-n40", "Label","BLI", "CT", "noiseBLI", "noiseCT", "combined_image_with_labels5.png")
#combine_images_6folder("double/imagesTs_clean", "double/test_label","double/output_cleanBLI38", "double/output_cleanCT39", "double/output_noiseBLI-n41", "double/output_noiseCT-n40", "Input", "Label","BLI", "CT", "noiseBLI", "noiseCT", "combined_image6.png")

print("single_allfolds")
#combine_images_6folder_with_labels("double/imagesTs_clean", "double/test_label","double/output_cleanBLI38", "double/output_cleanCT39", "double/output_noiseBLI-n41", "double/output_noiseCT-n40", "Input", "Label","BLI", "CT", "noiseBLI", "noiseCT", "addtwofold04_6.png")
print("single04")
combine_images_6folder_with_labels("single/imagesTs_clean", "single/labelsTs_rep","single/032_out_cl", "single/35output_clean", "single/37output_noise5_self", "single/30_noise_self", "Input", "Label","BLI", "CT", "noiseBLI", "noiseCT", "NEWsinglefold04_6.png")
print("double_allfolds") 
#combine_images_6folder_with_labels("double/imagesTs_clean", "double/test_label","all-folds/38/clean", "all-folds/39/clean", "all-folds/41/noise", "all-folds/40/noise", "Input", "Label","BLI", "CT", "noiseBLI", "noiseCT", "addtwofoldall_6.png")
print("double04")
#combine_images_6folder_with_labels("single/imagesTs_clean", "single/labelsTs_rep","all-folds/32/clean", "all-folds/35/clean", "all-folds/37/noise", "all-folds/30/noise", "Input", "Label","BLI", "CT", "noiseBLI", "noiseCT", "singleall_6.png")

