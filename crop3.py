from matplotlib import pyplot as plt
from PIL import Image
import pdb

imageObject  = Image.open("cropa.png")
plt.imshow(imageObject)
plt.show()
cropped = imageObject.crop((76,76,76,255))
cropped.show()
pdb.set_trace()
plt.savefig('cropa1.png')
