import os
import urllib.request
import zipfile
import shutil

DATA_DIR = "data"
TINY_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")
ZIP_PATH = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(TINY_DIR):
        if not os.path.exists(ZIP_PATH):
            print("Downloading Tiny-ImageNet (this may take a minute or two)...")
            urllib.request.urlretrieve("http://cs231n.stanford.edu/tiny-imagenet-200.zip", ZIP_PATH)
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        print("Formatting validation directory for PyTorch ImageFolder...")
        val_dir = os.path.join(TINY_DIR, 'val')
        val_annotations = os.path.join(val_dir, 'val_annotations.txt')
        
        if os.path.exists(val_annotations):
            with open(val_annotations, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    if len(parts) < 2: continue
                    img_file = parts[0]
                    img_class = parts[1]
                    
                    class_dir = os.path.join(val_dir, img_class)
                    if not os.path.exists(class_dir):
                        os.makedirs(class_dir)
                        
                    src = os.path.join(val_dir, 'images', img_file)
                    dst = os.path.join(class_dir, img_file)
                    if os.path.exists(src):
                        shutil.move(src, dst)
            # Cleanup: remove the empty images directory
            shutil.rmtree(os.path.join(val_dir, 'images'))
            print("Done formatting Tiny-ImageNet and cleaned up val/images.")
        else:
            print("Validation annotations not found, may already be formatted.")
    else:
        print("Tiny-ImageNet already exists.")

if __name__ == "__main__":
    download_and_extract()
