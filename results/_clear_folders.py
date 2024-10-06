import os

files = os.listdir("./results/pred_E/")
for f in files:
    os.remove("./results/pred_E/" + f)
    
files = os.listdir("./results/pred_v/")
for f in files:
    os.remove("./results/pred_v/" + f)

files = os.listdir("./results/pred_ux/")
for f in files:
    os.remove("./results/pred_ux/" + f)
    
files = os.listdir("./results/pred_uy/")
for f in files:
    os.remove("./results/pred_uy/" + f)