import os
import multiprocessing as mp
from PIL import Image

def thread(img_path):
    im1 = Image.open(img_path)
    
    if not im1.mode == 'RGB':
        im1 = im1.convert('RGB')

    im1.save(img_path.replace("png", "jpg") )
    os.remove(img_path)

def main():
    shpfiles = []
    maps = ['ap_Town01','ap_Town02','ap_Town03','ap_Town04'
    ,'ap_Town05','ao_Town06','ap_Town07','ap_Town10HD',
    'interactive','non-interactive',
    'runner_Town03','ap_Town05','ap_Town10HD']
    for m in maps:
        for dirpath, subdirs, files in os.walk("./" + m):
            for x in files:
                if x.endswith(".png"):
                    shpfiles.append(os.path.join(dirpath, x))
        pool = mp.Pool(processes = 12)
        for img_path in shpfiles:
            if "rgb" in img_path:
                print(img_path)
                pool.apply_async(thread, (img_path,))
        pool.close()
        pool.join()
    print("finish")


if __name__ == "__main__":
    main()
