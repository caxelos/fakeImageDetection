from PIL import Image
import os
import numpy as np

path_to_files = "dataset1/"
train_images = []
val_images=[]
test_images=[]

i = 0

for _, file in enumerate(os.listdir(path_to_files)):

    #data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
    single_im = Image.open(path_to_files+file).convert("RGB")
    #array = np.array(single_im)
    #img = Image.fromarray(array, 'RGB')
    img = np.array(single_im)
    if i<8000:
        train_images.append(img)#single_array)
    elif i < 9000:
        val_images.append(img)
    elif i < 10000:
        test_images.append(img)
    else:
        break
    i=i+1

np.savez("train_images.npz",images=train_images) # save all in one file
np.savez("eval_images.npz",images=val_images) # save all in one file
np.savez("test_images.npz",images=val_images) # save all in one file
