
import cv2
import numpy as np
from skimage.morphology import rectangle
import skimage.filters as filters
import glob as glob
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

#Method to read the images from a folder
def readImagesFromFolder(root_dir_path):
  image_paths =[]
  for images in glob.iglob(f'{root_dir_path}/*'):
    image_paths.append(images)
  return image_paths

def maximumSum(list1):
    return(sum(max(list1, key = sum)))

def combine_horizontally(images,padding=20):
    
    max_height = 0  # find the max height of all the images
    total_width = 0  # the total width of the images (horizontal stacking)

    image1 = images[0]
    image2 = images[1]

    image1_height = image1.shape[0]
    image1_width = image1.shape[1]

    image2_height = image2.shape[0]
    image2_width = image2.shape[1]

    if image1_height > image2_height:
      max_height = image1_height
    else:
      max_height = image2_height

    # add all the images widths
    total_width = image1_width + image2_width

    # create a new array with a size large enough to contain all the images
    # also add padding size for all the images except the last one
    final_image = np.zeros((max_height, (len(images)-1)*padding+ total_width, 3), dtype=np.uint8)

    current_x = 0  # keep track of where your current image was last placed in the x coordinate
    for image in images:
        # add an image to the final array and increment the x coordinate
        height = image.shape[0]
        width = image.shape[1]
        final_image[:height,current_x :width+current_x, :] = image
        #add the padding between the images
        current_x += width+padding

    return final_image

def patch_detection(image):

  # set the bounding box size
  bbox_size = 5
  alpha = 30 #Contrast control
  beta = -127 #Brightness control
  expected_patch_height_lowerlimit = 8
  expected_patch_height_upperlimit = 30
  expected_patch_width_lowerlimit = 8
  expected_patch_width_upperlimit = 30
 
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
  out_img = image.copy()
  #Find the bbox around the portion where the ADVERSARIAL PATCH was.
  #Threshold value based on the adjusted grayscale image
  (T, threshInv) = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  #print("[INFO] otsu's thresholding value: {}".format(T))


  img_gray = cv2.cvtColor(new_res, cv2.COLOR_BGR2GRAY)
  ret, im = cv2.threshold(img_gray, 0, T, cv2.THRESH_BINARY_INV)
  contours, _  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  x=0
  y=0
  w=0 
  h = 0
  for c in contours:
  # compute the bounding box of the contour and then draw the
  # bounding box on both input images to represent where the two images differ
    (x, y, w, h) = cv2.boundingRect(c)
    area_of_rectangle = w*h
    print(f'Height: {h} , Width: {w} , Area : {area_of_rectangle}')

    """ Range of area of rectangle is estimated according to the size of the image """
    if area_of_rectangle in list(range(200,360)):
          if w>h:
            w=h
            cv2.rectangle(out_img, (x, y), (x + w, y + h), (128, 128, 128), -1)
            print(f'From IF LOOP : Height: {h} , Width: {w}')
          else:
            h=w
            cv2.rectangle(out_img, (x, y), (x + w, y + h), (128, 128, 128), -1)
            print(f'From ELSE LOOP : Height: {h} , Width: {w}')
       
  return out_img,  (x, y, w, h)


if __name__ == '__main__':
    root_dir_path = '/content/drive/MyDrive/Adversarial Patch/Patched_Full_Dataset'
    masked_path_dir_path = '/content/output'
    image_path_list = readImagesFromFolder(root_dir_path)

    counter = 0

    for image_path in image_path_list:
        image = cv2.imread(image_path) # Read image
        print(image.shape)
        new_res,_ = patch_detection(image)

        images = []
        images.append(image)
        images.append(new_res)

        final_image=combine_horizontally(images,padding=0)

        out_file_name = image_path.split('/')[-1][:-4]
        cv2.imwrite(f'{masked_path_dir_path}/{out_file_name}.png',final_image)
        if counter >10:
          break
        else:
          counter = counter + 1
        
  



  
  
