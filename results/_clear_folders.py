import os

files = os.listdir("./results/pred_E/")
for f in files:
    os.remove("./results/pred_E/" + f)
files = os.listdir("./results/pred_v/")
for f in files:
    os.remove("./results/pred_v/" + f)