import cv2
import numpy as np

#grayscale methods

def simple_averaging(img):
    blue, green, red = cv2.split(img)
    greyscale_img = (blue + green + red) / 3
    return greyscale_img

def weighted_averaging(img, w1, w2, w3):
    blue, green, red = cv2.split(img)
    greyscale_img = (w1*blue + w2*green + w3*red) / (w1+w2+w3)
    return greyscale_img

def desaturation(img):
    blue, green, red = cv2.split(img)
    greyscale_img = (np.minimum(blue,green,red)+np.maximum(blue,green,red))/2
    return greyscale_img

def decomposition(img):
    blue, green, red = cv2.split(img)
    max_value = cv2.max(red, cv2.max(green, blue))
    min_value = cv2.min(red, cv2.min(green, blue))
    grayscale_image = (max_value + min_value) / 2
    return grayscale_image

def single_color_channel(img, color):
    if color == "blue":
        blue, green, red = cv2.split(img)
        return blue
    elif color == "green":
        blue, green, red = cv2.split(img)
        return green
    elif color == "red":
        blue, green, red = cv2.split(img)
        return red
    else:
        print("Invalid color channel")
        return None

def custom_number_of_grey_shades(img, p):
    blue, green, red = cv2.split(img)
    greyscale_img = (blue + green + red) / 3
    greyscale_img = (greyscale_img * p) / 256
    greyscale_img = greyscale_img.astype(np.uint8)
    greyscale_img = (greyscale_img * 256) / p
    return greyscale_img

def floyd_steinberg_dithering(image, num_shades):
    height, width = image.shape[:2]
    output_image = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            old_pixel = image[y, x]
            new_pixel = np.round(old_pixel * (num_shades - 1) / 255)
            output_image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel * 255 // (num_shades - 1)
            
            if x < width - 1:
                image[y, x + 1] += quant_error * 7 // 16
            if y < height - 1:
                if x > 0:
                    image[y + 1, x - 1] += quant_error * 3 // 16
                image[y + 1, x] += quant_error * 5 // 16
                if x < width - 1:
                    image[y + 1, x + 1] += quant_error * 1 // 16
    return output_image

def burkes_dithering(image, num_shades):
    height, width = image.shape[:2]
    output_image = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            old_pixel = image[y, x]
            new_pixel = np.round(old_pixel * (num_shades - 1) / 255)
            output_image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel * 255 // (num_shades - 1)
            
            if x < width - 1:
                image[y, x + 1] += quant_error * 8 // 32
            if x < width - 2:
                image[y, x + 2] += quant_error * 4 // 32
            if y < height - 1:
                if x > 1:
                    image[y + 1, x - 2] += quant_error * 2 // 32
                if x > 0:
                    image[y + 1, x - 1] += quant_error * 4 // 32
                image[y + 1, x] += quant_error * 8 // 32
                if x < width - 1:
                    image[y + 1, x + 1] += quant_error * 4 // 32
                if x < width - 2:
                    image[y + 1, x + 2] += quant_error * 2 // 32
    return output_image

def main():
    img = cv2.imread("lena.tif")
    
    #1 simple averaging
    greyscale_img = simple_averaging(img)
    cv2.imwrite("simple_avg_greyscale_img.jpg", greyscale_img)

    #2 weighted averaging
    greyscale_img = weighted_averaging(img, 0.3, 0.59, 0.11)
    cv2.imwrite("weighted_avg_greyscale_img.jpg", greyscale_img)

    #3 desaturation    
    greyscale_img = desaturation(img)
    cv2.imwrite("desaturation_greyscale_img.jpg", greyscale_img)

    #4 decomposition
    greyscale_img = decomposition(img)
    cv2.imwrite("decomposition_greyscale_img.jpg", greyscale_img)

    #5 single color channel
    blue_img = single_color_channel(img, "blue")
    cv2.imwrite("blue_img.jpg", blue_img)

    #6 custom number of grey shades
    greyscale_img = custom_number_of_grey_shades(img, 128)
    cv2.imwrite("custom_number_of_grey_shades_greyscale_img.jpg", greyscale_img)

    #7 floyd steinberg dithering
    greyscale_img = floyd_steinberg_dithering(img, 128)
    cv2.imwrite("floyd_steinberg_dithering_greyscale_img.jpg", greyscale_img)

    #8 burkes dithering
    greyscale_img = burkes_dithering(img, 128)
    cv2.imwrite("burkes_dithering_greyscale_img.jpg", greyscale_img)

if __name__ == "__main__":
    main()