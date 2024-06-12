import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image_notebook(image_path):
    
    def read(path):
        img=cv2.imread(path)
        return img

    def read_convertgray(path):
        imgray=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        return imgray

    def read_convertBGRtoRGB(path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def write(rgb_img,new_path):
        """
        Take an rgb image then, convert it to bgr image.
        Save the result in new_path location.
        """
        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_path, img)

    def show_img_cv(img,name="default"):
        cv2.imshow(name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_img(img,text="Image"):
        fig = plt.figure(figsize=(10, 20))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.axis("off")
        ax1.title.set_text(text)
        ax1.imshow(img)

    def visu(image,new_image,before="Before",after="After"):
        fig = plt.figure(figsize=(25, 45))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text(before)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text(after)

        ax1.imshow(image)
        ax2.imshow(new_image)

    def grayscale_histogram(path):
        img = cv2.imread(path,0)
        hi= [0]*256
        rows= img.shape[0]
        cols= img.shape[1]
        for r in range(rows-1):
            for c in range(cols-1):
                pixel = img[r,c]
                hi[pixel] += 1
        plt.plot(hi)
        plt.show()

    def grayscale_histogram2(img):

        hi= [0]*256
        rows= img.shape[0]
        cols= img.shape[1]
        for r in range(rows-1):
            for c in range(cols-1):
                pixel = img[r,c]
                hi[pixel] += 1
        plt.plot(hi)
        plt.show()


    def color_histogram(path,title='Histogram Analysis'):
        img = cv2.imread(path)
        color = ('blue', 'green', 'red')


        for i,color in enumerate(color):
            histogram = cv2.calcHist([img],[i],None,[256],[0,256])

            plt.plot(histogram, color=color, label=color+' channel')
            plt.xlim([0,256])

        plt.title(title,fontsize=15)
        plt.xlabel('Range intensity values',fontsize=10)
        plt.ylabel('Count of Pixels',fontsize=10)
        plt.legend()
        plt.show()
        
    def brighten_image(image, brightness=50):
        """Brightens the image by adding a constant value to each pixel."""
        brightened = np.clip(image.astype(int) + brightness, 0, 255).astype(np.uint8)
        return brightened
    
    
    img=read_convertgray(image_path)

    mean_value= np.mean(img)
    print(mean_value)
    
    mean_value=int(mean_value+1)
    mean_value
    
    ret,thresh1 = cv2.threshold(img,mean_value,255,cv2.THRESH_BINARY)



    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 51, 5)
    original_img=th3

    img = cv2.imread(image_path, 0)
    kernel = np.ones((2,2),np.uint8)
    #iteration1
    dilation = cv2.dilate(thresh1,kernel,iterations = 1)

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    erosion = cv2.erode(img,kernel,iterations = 1)

    #iteration2
    closed = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    
    brightened_image = brighten_image(closed, brightness=75)
    
    output_img =  brightened_image


    cv2.imwrite("output_img.png", output_img)
    
    # Convert the final processed image to RGB format
    output_img = cv2.imread("output_img.png", cv2.IMREAD_GRAYSCALE)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)

    return original_img,output_img