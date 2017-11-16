from skimage import io, color

image_file = 'images/Greyscale.jpg'
image = io.imread(image_file)
# image = color.rgba2gray(image)

width = len(image)
height = len(image[0])

for i in range(width):
    for j in range(height):
        k = image[i,j]
        print(k)

# print (image)
