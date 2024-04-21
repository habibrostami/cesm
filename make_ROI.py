
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms as T

# Read image

img = Image.open('./data/masks/P102_R_CM_CC.jpg.png') #mask
main_image = Image.open('./data/org/CM/1/P102_R_CM_CC.jpg')

convert_tensor = transforms.ToTensor()
conv = convert_tensor(img)


#remove heart
#conv =  torch.where(conv == 1, torch.tensor(0), conv)
conv = torch.where(conv != 0, torch.tensor(1), conv)


main_conv = convert_tensor(main_image)
prepr = main_conv * conv

transform_to_pil = T.ToPILImage()

proc_img = transform_to_pil(prepr)
# main_image.show()
proc_img.show()
# Output Images
img = transform_to_pil(conv)
# img.show()
print(type(img))

import numpy as np
np.savetxt('/stuff/saves_txt/my_file.txt', conv.numpy()[0])

# prints format of image
print(img.format)

# prints mode of image
print(img.mode)
