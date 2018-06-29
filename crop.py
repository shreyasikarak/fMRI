from matplotlib import pyplot as plt
from PIL import Image


imageObject  = Image.open("image.png")
plt.imshow(imageObject)
plt.show()
cropped     = imageObject.crop((40,40,180,200))
cropped.show()
plt.savefig('NewImage.png')