import sys
import os 
import argparse
import subprocess

def set_args():
    parser = argparse.ArgumentParser("Script for deploy sbtach training")
    parser.add_argument("--root", type=str, default="/home/tianqinl/PCL/script")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--args", nargs='+', default="", help="args for the files")

    args = parser.parse_args()
    return args




def sbatch(args):
    """function that involke python script"""
    bashFileName = os.path.join(args.root, args.file)
    argument = " ".join(args.args)
    bashCommand = f"sbatch {bashFileName} {argument}"
    print(f"BASH COMMAND :{bashCommand}")
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode('ascii')
    print(f"OUTPUT: {output}")
    # get the submission id
    sid = output.split(" ")[-1][:-1]
    print(f"SID: {sid}")



def main():
    args = set_args()
    print(" ".join(args.args))
    sbatch(args)

if __name__ == "__main__":
    main()