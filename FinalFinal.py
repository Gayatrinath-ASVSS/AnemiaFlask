import joblib
import cv2
from PIL import Image
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import random
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgbtry():    # Open the image
   img = Image.open('static/images/eye.jpg') #C:\\Users\\PC\\Desktop\\anemia-detection-with-machine-learning-main\\

    # Get the RGB values from the image
   rgb_values = list(img.getdata())

    # Calculate the average RGB values
   total_pixels = len(rgb_values)
   total_red = total_green = total_blue = 0
   print('before')
   for r, g, b in rgb_values:
    total_red += r
    total_green += g
    total_blue += b
   avg_red = (total_red //total_pixels ) *(random.randint(35, 45)/100)
   avg_green = (total_green //total_pixels )*(random.randint(25, 35)/100)
   avg_blue = (total_blue //total_pixels )*(random.randint(30, 45)/100)
   print(avg_red,avg_green,avg_blue)
   return [avg_red,avg_green,avg_blue]

def sharp():
    img = Image.open('static/images/eye.jpg') #C:\\Users\\PC\\Desktop\\anemia-detection-with-machine-learning-main\\
    sharpened_img = img.filter(ImageFilter.SHARPEN) 
    sharpened_img.save("static/images/sharp.jpg")
    img.close()
    return
def laplacian():
    image_path = 'static/images/eye.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply the Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Convert the result to uint8 (8-bit) and normalize for display
    laplacian_display = cv2.convertScaleAbs(laplacian)

    # Display the original and filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(laplacian_display, cmap='gray')
    plt.title('Laplacian Filtered Image')
    plt.savefig('static/images/laplacian.jpg')
    # plt.show()


    # image = cv2.imread('static/images/eye.jpg', cv2.IMREAD_GRAYSCALE)

    #     # Apply Laplacian filter
    # laplacian_image = cv2.Laplacian(image, cv2.CV_64F)

    #     # Create a subplot
    # # plt.subplot(1, 2, 1)
    # # plt.imshow(image, cmap='gray')
    # # plt.title('Original Image')
    
    # plt.subplot(1, 2, 2)
    # # plt.imshow(laplacian_image, cmap='gray')
    # plt.title('Laplacian Filter')
    
    #     # Save the subplot
    # plt.savefig('static/images/laplacian.jpg')
    # plt.close()


    # img = Image.open('static/eye.jpg')
    # plt.savefig('static/lapplot1.jpg')
    # plt.subplot(1, 2, 1)
    # laplacian_kernel = ImageFilter.Kernel((3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1])
    # sharpened_image2 = img.filter(laplacian_kernel)
    # sharpened_image2.save('static/laplacian.jpg') 
    # plt.savefig('static/lapplot2.jpg')
    # plt.subplot(1, 2, 2) 
    return
def sobel(): 
# converting because opencv uses BGR as default
    img = cv2.imread('static/images/eye.jpg')

    img1 = Image.open('static/images/eye.jpg')
        # Convert the image to RGB mode
    rgb_image = img1.convert("RGB")
        # Save the RGB image
    rgb_image.save('static/images/RGB.jpg')
    gray = img1.convert("L")
    gray.save('static/images/GRAY.jpg')
    blurred_image = cv2.GaussianBlur(img,(5, 5),0)
    cv2.imwrite('static/images/Gauss.jpg', blurred_image)
    # convolute with sobel kernels
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    cv2.imwrite('static/images/sobelx.jpg', sobelx)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
    cv2.imwrite('static/images/sobely.jpg', sobely)
    #Plotting images
    # plt.savefig('static/images/sobelx.jpg')
    # plt.imshow(sobelx)
    # plt.savefig('static/images/sobely.jpg')
    # plt.imshow(sobely)
    plt.close()
    
    return

def histo():
    image = cv2.imread('static/images/eye.jpg', cv2.IMREAD_GRAYSCALE)
# Plot the histogram
    plt.hist(image.flatten(), bins=256, range=[0,256], color='gray', alpha=0.7)
    plt.title('Pixel Intensity Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig('static/images/histo.jpg')
    # plt.show()
    plt.close()
    return

def contour():

# Read the image
    image = cv2.imread('static/images/eye.jpg', cv2.IMREAD_GRAYSCALE)
    # Apply Canny edge detection
    edges = cv2.Canny(image, 30, 100)
    # Plot the original and edges side by side
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Edge Detection')
    plt.savefig('static/images/contour.jpg')
    # plt.show()
    plt.close()
    return

def cmv():
    image = cv2.imread('static/images/eye.jpg')

# Convert BGR to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot the image with different color maps
    plt.subplot(131), plt.imshow(image_rgb), plt.title('Original Image')
    plt.subplot(132), plt.imshow(image_rgb, cmap='viridis'), plt.title('Viridis Map')
    plt.subplot(133), plt.imshow(image_rgb, cmap='jet'), plt.title('Jet Map')
    plt.savefig('static/images/cmv.jpg')
    # plt.show()
    plt.close()
    return

def scatter():
    img = cv2.imread('static/images/eye.jpg')

# Create some sample data for the scatter plot
    x = np.random.rand(10) * img.shape[1]  # random x-coordinates within the image width
    y = np.random.rand(10) * img.shape[0]  # random y-coordinates within the image height

    # Create a scatter plot on the image
    # plt.imshow(img)
    plt.scatter(x, y, color='red', marker='o')  # You can customize the color and marker type
    plt.savefig('static/images/scatter.jpg')
    # Display the plot
    # plt.show()
    plt.close()

def line():
    img = mpimg.imread('static/images/eye.jpg')

# Create some sample data for the line plot
    x = np.linspace(0, img.shape[1], 100)  # Generate x-coordinates
    y = np.sin(x) * img.shape[0] / 4 + img.shape[0] / 2  # Generate corresponding y-coordinates

    # Create a line plot on the image
    # plt.imshow(img)
    plt.plot(x, y, color='blue', linewidth=2)  # You can customize the color and linewidth
    plt.savefig('static/images/line.jpg')
    # Display the plot
    # plt.show()
    plt.close()

def bar():
    image_path = 'static/images/eye.jpg'
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Calculate the average intensity along each channel (assuming RGB)
    average_intensity = np.mean(image_array, axis=(0, 1))

    # Bar plot
    channels = ['Red', 'Green', 'Blue']
    plt.bar(channels, average_intensity, color=['red', 'green', 'blue'])
    plt.title('Average Intensity per Channel')
    plt.xlabel('Channels')
    plt.ylabel('Average Intensity')
    plt.savefig('static/images/bar.jpg')
    plt.close()

   




#model.py
# excel_file_path = 'C:\\Users\\PC\\Desktop\\ADmodel.xlsx'
# df = pd.read_excel(excel_file_path)

# X = df.iloc[:, :-1]  # Assuming all columns except the last one are features
# y = df.iloc[:, -1]    # Selecting the last column as the target

# # Step 3: Handle inconsistent samples by filling missing values separately for training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# imputer = SimpleImputer(strategy='mean')  # You can use other strategies like 'median', 'most_frequent', etc.
# X_train_imputed = imputer.fit_transform(X_train)
# X_test_imputed = imputer.transform(X_test)

# # Step 4: Choose a machine learning model (Random Forest Regressor)
# model = RandomForestRegressor()

# # Step 5: Train the model
# model.fit(X_train_imputed, y_train)

# # Step 6: Make predictions on the test set
# y_pred = model.predict(X_test_imputed)

# # Step 7: Evaluate the model using Mean Squared Error (MSE)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

# model_filename = 'trained_model.joblib'
# joblib.dump(model, model_filename)
# print(f'Trained model saved to {model_filename}')

#rbganother program
# def calculate_percentage(image, target_color_lower, target_color_upper):
#     # Convert the image to HSV
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define the color range in HSV
#     lower_bound = np.array(target_color_lower, dtype=np.uint8)
#     upper_bound = np.array(target_color_upper, dtype=np.uint8)

#     # Create a mask to extract the pixels within the specified color range
#     mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

#     # Calculate the percentage of pixels that match the specified color
#     total_pixels = np.sum(mask > 0)
#     total_pixels_in_image = np.size(mask)
#     percentage = (total_pixels / total_pixels_in_image) * 100

#     return percentage

# def rbg_percentage():
#     image = cv2.imread('C:\\Users\\PC\\Desktop\\anemia-detection-with-machine-learning-main\\eye1.jpg')

# # Specify the color ranges in HSV
#     lower_red = [0, 50, 50]
#     upper_red = [10, 255, 255]

#     lower_green = [0, 50, 35]
#     upper_green = [18, 12, 13]

#     lower_blue = [60, 10, 139]
#     upper_blue = [10, 140, 40]

# # Calculate the percentage of each color in the image
#     red_percentage = calculate_percentage(image, lower_red, upper_red)
#     green_percentage = calculate_percentage(image, lower_green, upper_green)
    
#     blue_percentage = calculate_percentage(image, lower_blue, upper_blue)
#     return [red_percentage,blue_percentage,green_percentage]



def main():
    capture_and_save_eye_image()
    sharp()
    laplacian()
    sobel()
    histo()
    contour()
    cmv()
    scatter()
    line()
    bar()
    res=rgbtry()# res=rbg.rbg_percentage()
    loaded_model = joblib.load('trained_model.joblib')
    new_input_data = [res] 
    predicted_value = loaded_model.predict(new_input_data)
        # Step 11: Print the predicted value
    print(f'Predicted Value: {predicted_value[0]}')
    return predicted_value[0]
