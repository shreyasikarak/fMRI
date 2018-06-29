from matplotlib import pyplot as plt
from PIL import Image
import pdb

imageObject  = Image.open("imagea.png")

plt.imshow(imageObject)
plt.show()
# cropped = imageObject.crop((1590, 749,900,550))
cropped = imageObject.crop((1570, 723, 2194, 1394))
cropped.save('cropa.png',dpi=(1280, 1080))
pdb.set_trace()

cropped.show()
plt.savefig('cropa.png')
