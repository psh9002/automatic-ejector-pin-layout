import subprocess
import sys


argv = sys.argv[1:]

while True:
    """However, you should be careful with the '.wait()'"""
    cmd = ["python"]
    cmd += argv
    cmd = " ".join(cmd)
    p = subprocess.Popen(cmd, shell=True).wait()

    """#if your there is an error from running  
    the while loop will be repeated, 
    otherwise the program will break from the loop"""
    if p != 0:
        continue
    else:
        break