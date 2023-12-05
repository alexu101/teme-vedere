import numpy as np
import cv2

def rotate_image(image, angle,clockwise):
    height, width = image.shape[:2]
    rotation_center = (width/2,height/2)
    if clockwise:
        angle = -angle
    
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center,angle,1)
    rotated_image = cv2.warpAffine(image,rotation_matrix,(width,height))
    return rotated_image

def crop_image(image,upper_left_pixel,width,length):
    cropped_image = image[upper_left_pixel[0]:upper_left_pixel[0]+length,upper_left_pixel[1]:upper_left_pixel[1]+width]
    return cropped_image

def create_emoji():
    #create a black image
    image = np.zeros((500,500,3),np.uint8)
    #draw a circle
    cv2.circle(image,(250,250),200,(255,255,255),-1)
    #draw eyes
    cv2.circle(image,(200,200),30,(0,0,0),-1)
    cv2.circle(image,(300,200),30,(0,0,0),-1)
    #draw mouth
    cv2.ellipse(image,(250,300),(100,50),0,0,180,(0,0,0),-1)
    cv2.imshow('emoji',image)
    cv2.waitKey(0)

def exercise_2(img):
    print(img.shape) #display image size
    cv2.imshow('image',img)
    cv2.waitKey(0)

def exercise_3(img):
    #blured and sharpened image
    blur_kernel = np.ones((10,10),np.float32)/100
    sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)
    blured_image = cv2.filter2D(src=img,ddepth=-1,kernel=blur_kernel)
    sharpened_image = cv2.filter2D(src=img,ddepth=-1,kernel=sharpen_kernel)
    cv2.imshow('image',img)
    cv2.imshow('blured_image',blured_image)
    cv2.imwrite('blured_image.png',blured_image)
    cv2.imshow('sharpened_image',sharpened_image)
    cv2.imwrite('sharpened_image.png',sharpened_image)
    cv2.waitKey(0)

def exercise_4(img):
    sepia_kernel = np.array([[0,-2,0],
                      [-2,8,-3],
                      [0,-2,0]])
    sepia_image = cv2.filter2D(src=img,ddepth=-1,kernel=sepia_kernel)
    cv2.imshow('image',sepia_image)
    cv2.imwrite('sepia_image.png',sepia_image)
    cv2.waitKey(0)

def exercise_5(img):
    #rotate image clockwise and counter clockwise
    clockwise_image_90deg = rotate_image(img,90,True)
    counter_clockwise_image_90deg = rotate_image(img,90,False)
    cv2.imshow('clockwise',clockwise_image_90deg)
    cv2.imshow('counterclockwise',counter_clockwise_image_90deg)
    cv2.waitKey(0)

def exercise_6(img):
    #crop image
    cropped_image = crop_image(img,(0,0),300,300)
    cv2.imshow('image',cropped_image)
    cv2.waitKey(0)  

def exercise_7(img):
    my_emoji = create_emoji()
    cv2.imshow('emoji',my_emoji)
    cv2.imwrite('emoji.jpg',my_emoji)
    cv2.waitKey(0)

def main():
    img = cv2.imread('lena.tif')

    exercise_7(img)
    
if __name__ == "__main__":
    main()

