import os
import tifffile

filename = "MBP_S0010_F0007_Z00.tif"
I = tifffile.imread(filename)
print I.shape


startx = 700
starty = 800
sz = 500

newI = I[startx:startx+sz, starty:starty+sz]
tifffile.imsave("template.tif",newI)
