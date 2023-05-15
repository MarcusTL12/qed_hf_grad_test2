#!/usr/bin/env python3
import sys
import subprocess
import os

action = sys.argv[1]
cluster_name = sys.argv[2]
filepath = sys.argv[3]

remote_prefix = f"~/qed_hf_grad_test2/"

command = "scp -r"

dirname = os.path.dirname(filepath)

if action == 'f': # fetch
    remote_path = f"{cluster_name}:{remote_prefix}{filepath}"
    command = ' '.join([command, remote_path, dirname])
elif action == 'p': # push
    remote_path = f"{cluster_name}:{remote_prefix}{dirname}/"
    command = ' '.join([command, filepath, remote_path])

print(command)

subprocess.run(command, shell=True)

