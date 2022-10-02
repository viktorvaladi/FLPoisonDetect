from PIL import Image
import os
import glob
import random
from itertools import permutations

def poisonData(num_to_poison = 10, path = '/root/data/cinic-10/', amt_of_pixels = 100, acceptable_range = 0.05):
    fldr_list = os.listdir(path = path + 'train/')

    for fldrs in fldr_list:
        to_poison = random.sample(glob.glob(path + 'train/'  + fldrs + '/*.png'), num_to_poison)
        #Create the directories if needed.
        if not os.path.isdir(path + 'train_poison/' + fldrs):
            os.makedirs(path + '/train_poison/' + fldrs)
        for pics in to_poison:
            img = Image.open(pics)
            pixel_map = img.load()
            pixel_list = list(set(permutations(range(32), 2)))
            pixels_to_change = random.sample(pixel_list, k = amt_of_pixels)
            for pixel in pixels_to_change:
                #Select the pixel to change.
                x = pixel[0]
                y = pixel[1]
                #Change each selected pixel by altering each channel by the range.
                if not isinstance(pixel_map[x,y], int):
                    pixel_map[x,y] = (int(random.uniform(pixel_map[x,y][0]*(1-acceptable_range), pixel_map[x,y][0]*(1+acceptable_range))),
                                    int(random.uniform(pixel_map[x,y][1]*(1-acceptable_range), pixel_map[x,y][1]*(1+acceptable_range))),
                                    int(random.uniform(pixel_map[x,y][2]*(1-acceptable_range), pixel_map[x,y][2]*(1+acceptable_range))))
                #If the image is greyscale we instead change just have to modify the intensity.
                else:
                    pixel_map[x,y] = int(random.uniform(pixel_map[x,y]*(1-acceptable_range), pixel_map[x,y]*(1+acceptable_range)))
            #Generate the new filename.
            #new_name = img.filename.split('/')[-1].split('.')[0] + '_poisoned.png'
            new_name = os.path.basename(os.path.splitext(img.filename)[0] + '_poisoned.png')
            #Save the new image to the selected folder.
            img.save(path + 'train_poison/' + fldrs + '/' + new_name)

if __name__ == '__main__':
    poisonData()
