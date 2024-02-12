import os

n_nodes = 8

if os.path.exists("machine_name.txt"):
    os.system("rm machine_name.txt")

print(f"sbatch run_mobilenet.sh '0' '' '{n_nodes}'")
os.system(f"sbatch run_mobilenet.sh '0' '' '{n_nodes}'")

test_read = False
while not(test_read):
    try:
        machine_name = open("machine_name.txt", "r").read().rstrip()
        test_read = True
        print("Master address found:", machine_name+"!", flush = True)
    except:
        pass

for i in range(1,n_nodes):
    print(f"sbatch run_mobilenet.sh '{i}' '{machine_name}' '{n_nodes}'")
    os.system(f"sbatch run_mobilenet.sh '{i}' '{machine_name}' '{n_nodes}'")
