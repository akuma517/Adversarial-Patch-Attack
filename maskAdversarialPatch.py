import cv2
import numpy as np
from skimage.morphology import rectangle
import skimage.filters as filters

#Method to read the images from a folder
def readImagesFromFolder(root_dir_path):
  image_paths =[]
  for images in glob.iglob(f'{root_dir_path}/*'):
    image_paths.append(images)
  return image_paths

def maximumSum(list1):
    return(sum(max(list1, key = sum)))

# set the bounding box size
bbox_size = 4
alpha = 30 #Contrast control
beta = -127 #Brightness control
root_dir_path = '/content/drive/MyDrive/Adversarial Patch/Mini Dataset/Patched_Train/0'
masked_path_dir_path = '/content/drive/MyDrive/Adversarial Patch/Mini Dataset/MaskedPatchOutput/0'
image_path_list = readImagesFromFolder(root_dir_path)


for image_path in image_path_list:
  image = cv2.imread(image_path) # Read image
  adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

  # Variance = mean of square of image - square of mean of image

  # read the image
  # convert to 16-bits grayscale since mean filter below is limited 
  # to single channel 8 or 16-bits, not float
  # and variance will be larger than 8-bit range
  img = cv2.cvtColor(adjusted,cv2.COLOR_RGB2GRAY).astype(np.uint16)

  # compute square of image
  img_sq = cv2.multiply(img, img)

  # compute local mean in bbox_size x bbox_size rectangular region of each image
  # note: python will give warning about slower performance when processing 16-bit images
  region = rectangle(bbox_size, bbox_size)
  mean_img = filters.rank.mean(img, selem=region)
  mean_img_sq = filters.rank.mean(img_sq, selem=region)

  # compute square of local mean of img
  sq_mean_img = cv2.multiply(mean_img, mean_img)

  # compute variance using float versions of images
  var = cv2.add(mean_img_sq.astype(np.float32), -sq_mean_img.astype(np.float32))

  # compute standard deviation
  std = cv2.sqrt(var).clip(0,255).astype(np.uint16)

  binary_mask = std  < 1
  
  result = binary_mask[...,None] * image
  #change the color of masked part from black to gray
  new_res = result.copy()
  # d = result   # Difference of color channels
  # q = d < 2             # Threshold criterion
  # new_res[q] = 128      # Overwrite data based on threshold

  #Find the bbox around the portion where the ADVERSARIAL PATCH was.
  img_gray = cv2.cvtColor(new_res, cv2.COLOR_BGR2GRAY)
  ret, im = cv2.threshold(img_gray, 0, 1, cv2.THRESH_BINARY_INV)
  contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  for c in contours:
  # compute the bounding box of the contour and then draw the
  # bounding box on both input images to represent where the two images differ
    (x, y, w, h) = cv2.boundingRect(c)
    print(f'Height: {h} , Width: {w}')
    if h>6 and h<21 and w>6 and w<21:
      cv2.rectangle(new_res, (x, y), (x + w, y + h), (128, 128, 128), -1)
  # show the output images
  #save the results
  #cv2.imwrite('/content/output/output.jpg',result)
  # show results
  plt.figure()
  plt.subplot(1,5,1)
  plt.imshow( image)
  plt.title('Original Image') 
  plt.subplot(1,5,2)
  plt.imshow(std)
  plt.title('Standard Deviation') 
  plt.subplot(1,5,3)
  plt.imshow(binary_mask)  # grayscale image
  plt.title('Mask Image') 
  plt.subplot(1,5,4)
  plt.imshow(result)
  plt.title('Cropped Image')
  plt.subplot(1,5,5)
  plt.imshow(new_res)
  plt.show()
  
  out_file_name = image_path.split('/')[-1][:-4]
  cv2.imwrite(f'{masked_path_dir_path}/{out_file_name}.png',new_res)
  
