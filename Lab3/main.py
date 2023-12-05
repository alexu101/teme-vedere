import cv2 as cv
import numpy as np
import os

def skin_detection_rgb(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    skin_mask = cv.inRange(image, (0, 50, 100), (20, 255, 255))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r,g,b = image[i,j]
            if r > 95 and g > 40 and b > 20 and max(r,g,b) - min(r,g,b) > 15 and abs(r-g) > 15 and r > g and r > b:
                skin_mask[i,j] = 255
    return skin_mask

def skin_detection_hsv(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    min = np.array([0,0.23*255,0.35*255], dtype = "uint8")
    max = np.array([50, 0.68*255, 255], dtype = "uint8")
    skin_mask = cv.inRange(image, min, max)
    return skin_mask

def skin_detection_ycrcb(image):
  ycbcr_image = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
  min = np.array([80, 85, 135], dtype="uint8")
  max = np.array([255, 135, 180], dtype="uint8")
  skin_mask = cv.inRange(ycbcr_image, min, max)

  return skin_mask

def get_accuracy(folder1, folder2, detection):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for image in os.listdir(folder1):
        img = os.path.join(folder1,image)
        ground_truth_image = os.path.join(folder2,os.path.splitext(image)[0] + ".png")

        img = cv.imread(img)
        bw_img = detection(img)
        ground_truth_img = cv.imread(ground_truth_image)
        img_height, img_width = ground_truth_img.shape[:2]
        ground_truth_image_h, ground_truth_image_w = ground_truth_img.shape[:2]

        for y_original, y_gt in zip(range(img_height), range(ground_truth_image_w)):
            for x_original, x_gt in zip(range(img_width), range(ground_truth_image_h)):
                black_white_pixel = bw_img[y_original, x_original]
                ground_truth_pixel = ground_truth_img[y_gt, x_gt]
                if np.all(black_white_pixel == 255) and np.all(ground_truth_pixel == 255):
                    tp = tp + 1
                if np.all(black_white_pixel == 0) and np.all(ground_truth_pixel == 0):
                    tn = tn + 1
                if np.all(black_white_pixel == 255) and np.all(ground_truth_pixel == 0):
                    fp = fp + 1
                if np.all(black_white_pixel == 0) and np.all(ground_truth_pixel == 255):
                    fn = fn + 1
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    return (accuracy, tp, tn, fp, fn)

def create_confusion_matrix(tp, tn, fp, fn):
    confusion_matrix = np.array([[tp, fp], [fn, tn]])
    return confusion_matrix       

def task_a():
    #a)
    image = cv.imread('lena.tif')
    cv.imshow("Original", image)
    cv.imshow("Skin Detection 1", skin_detection_rgb(image))
    cv.imshow("Skin Detection 2", skin_detection_hsv(image))
    cv.imshow("Skin Detection 3", skin_detection_ycrcb(image))
    cv.waitKey(0)

def task_b():
    #b)
    images_folder_truth_folder = "./GroundT_FamilyPhoto"
    images_folder = "Pratheepan_Dataset/FamilyPhoto"
    accuracy, tp, tn, fp, fn = get_accuracy(images_folder, images_folder_truth_folder, skin_detection_rgb)
    print("RGB Accuracy: ", accuracy)
    print("RGB Confusion Matrix: ", create_confusion_matrix(tp, tn, fp, fn))

    #the same for other methods
    #display the statistics

def task_c():
    #c)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv.imread("pic.jpg")
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
    for (x, y, w, h) in faces:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    face_image = img.copy()
    cv.rectangle(face_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    cv.imshow("face",face_image)
    cv.waitKey(0)

def main():
    task_c()
    

if __name__ == '__main__':
    main()