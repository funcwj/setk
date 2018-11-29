# wujian@2018

import subprocess


def run_command(command, wait=True):
    """ 
    Runs shell commands. These are usually a sequence of 
        commands connected by pipes, so we use shell=True
    """
    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if wait:
        [stdout, stderr] = p.communicate()
        if p.returncode is not 0:
            raise Exception(
                "There was an error while running the command \"{0}\":\n{1}\n".
                format(command, bytes.decode(stderr)))
        return stdout, stderr
    else:
        return p