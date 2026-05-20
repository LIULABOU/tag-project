import json, random
from PIL import Image
import matplotlib.pyplot as plt

DATA_FILE = "data/photochat/train_image_photo_desc.jsonl"

lines = open(DATA_FILE).read().splitlines()
sample = json.loads(random.choice(lines))

print("Photo ID:", sample["photo_id"])
print("Photo Description:", sample["photo_description"])

img = Image.open(sample["image_path"])
plt.figure(figsize=(5,5))
plt.imshow(img)
plt.axis("off")
plt.show()
