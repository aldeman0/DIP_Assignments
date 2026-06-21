import os
from PIL import Image

img_dir = '/root/autodl-tmp/data/lego/images'
backup_dir = os.path.join(img_dir + '_rgba_backup')
os.makedirs(backup_dir, exist_ok=True)

for f in sorted(os.listdir(img_dir)):
    if f.endswith('.png'):
        path = os.path.join(img_dir, f)
        img = Image.open(path)
        if img.mode == 'RGBA':
            img.save(os.path.join(backup_dir, f))  # backup
            img.convert('RGB').save(path)           # overwrite
            print(f'Converted: {f}')
print('Done')
