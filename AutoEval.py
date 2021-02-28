"""Class that automatically deploy eval test"""
import sys 
import subprocess


def sbatch(bashFileName, argument):
    """function that involke python script"""
    bashCommand = f"sbatch {bashFileName} {argument}"
    print("######## Deploy SBTACH COMMAND #######")
    print(f"BASH COMMAND :{bashCommand}")
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode('ascii')
    print(f"OUTPUT: {output}")
    # get the submission id
    sid = output.split(" ")[-1][:-1]
    print(f"SID: {sid}")
    return sid


class AutoEval:
    """class that schedule automatical eval during training"""

    def __init__(self, model_folder, eval_script_filename, checkpoint_templete="checkpoint_{}.pth.tar"):
        self.model_folder = model_folder
        self.eval_script_filename = eval_script_filename # e.g. run_linear_eval_targert.sh
        self.checkpoint_templete = checkpoint_templete


    def eval(self, epoch):
        """on call, deploy and return sbatch id"""
        argument = "{} {}".format(self.model_folder, str(epoch).zfill(4))
        self.sid = sbatch(self.eval_script_filename, argument)
        return self.sid 



if __name__ == "__main__":
    autoE = AutoEval("folder_name", "script")
    autoE.eval(30)
