import os
import multiprocessing as mp
from PIL import Image

def scale(image_path, scale=2.0):

    image = Image.open(image_path).convert('RGB')
    # (width, height) = (int(image.width // scale), int(image.height // scale))
    (width, height) = (768, 256)
    # (width, height) = (224, 224)
    im_resized = image.resize((width, height), Image.ANTIALIAS)

    return im_resized

def thread(img_path):
    # im1 = Image.open(img_path)
    
    # if not im1.mode == 'RGB':
    #     im1 = im1.convert('RGB')

    # folder = 'downsampled'
    # folder = 'downsampled_224'
    folder = 'downsampled_256x768'
    new_img_path = img_path.replace("images", folder)
    
    if not os.path.isdir(new_img_path[:-7]):
        os.makedirs(new_img_path[:-7])


    if not os.path.isfile(new_img_path):
        image = scale(img_path)
        image.save(new_img_path)
        print(new_img_path)
    # print(img_path)

def main():
    shpfiles = []
    # if not os.path.isdir("./downsampled_224"):
        # os.makedirs("./downsampled_224")
    if not os.path.isdir("./downsampled_256x768"):
        os.makedirs("./downsampled_256x768")
    for dirpath, subdirs, files in os.walk("./images"):
        for x in files:
            if x.endswith(".jpg"):
                shpfiles.append(os.path.join(dirpath, x))
    pool = mp.Pool(processes = 12)
    for img_path in shpfiles:
        if "images" in img_path:
            # print(img_path)
            pool.apply_async(thread, (img_path,))
    pool.close()
    pool.join()
    print("finish")


if __name__ == "__main__":
    main()
