#%%
import os

l_experiments = os.listdir("experiments")
l_experiments = [x.split(".")[0] for x in l_experiments if x.split(".")[-1] == "py"]

for experiment in l_experiments:
    os.system("python experiments/"+experiment+".py")

print("Json files for the experiments generated")