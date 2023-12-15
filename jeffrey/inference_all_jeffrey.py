import subprocess
import os
import threading
import re
files = os.listdir("checkpoints")
pattern = re.compile(r'ckpt_epoch_(\d+)\.pth')
processes=[]

semaphore = threading.Semaphore(8)

def run_process(cmd):
    with semaphore:
        subprocess.run(cmd)
try:
    os.mkdir("results_jeffrey")
except:
    print("Directory already exists")
for file in files:
    this_match = pattern.match(file)
    if this_match:
        number = this_match.group(1)
        number = int(number)
        if number%100!=0:
            continue
        cmd = ["python","inference.py", str(number)]
        t = threading.Thread(target=run_process, args=(cmd,))
        t.start()
        processes.append(t)

for p in processes:
    p.join()
