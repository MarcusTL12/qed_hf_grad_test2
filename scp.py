#!/usr/bin/env python3
import sys
import subprocess
import os

action = sys.argv[1]
cluster_name = sys.argv[2]
filepath = sys.argv[3]

username = "marcusl"

remote_prefix = f"/home/{username}/qed_hf_grad_test2/"

command = "scp"

dirname = os.path.dirname(filepath)

if action == 'f':
    remote_path = f"{username}@{cluster_name}.nt.ntnu.no:{remote_prefix}{filepath}"
    command = ' '.join([command, remote_path, dirname])
elif action == 'p':
    remote_path = f"{username}@{cluster_name}.nt.ntnu.no:{remote_prefix}{dirname}"
    command = ' '.join([command, filepath, remote_path])

print(command)

subprocess.run(command, shell=True)
