import subprocess
import os

path = "/home/ubuntu/datasets/datasets/preprocessed_data/"
file_name = ["my-gpt2_text_document_test_indexmap_640000ns_1024sl_1234s_doc_idx.npy", "my-gpt2_text_document_test_indexmap_640000ns_1024sl_1234s_sample_idx.npy", "my-gpt2_text_document_test_indexmap_640000ns_1024sl_1234s_shuffle_idx.npy", "my-gpt2_text_document_valid_indexmap_2560000ns_1024sl_1234s_sample_idx.npy", "my-gpt2_text_document_valid_indexmap_2560000ns_1024sl_1234s_shuffle_idx.npy"]

hosts = ["ec2-18-237-46-46.us-west-2.compute.amazonaws.com", "ec2-54-202-60-154.us-west-2.compute.amazonaws.com", "ec2-18-237-225-91.us-west-2.compute.amazonaws.com"]
for file in file_name:
    file_path = os.path.join(path, file)
    for host in hosts:
        subprocess.run(f"scp {file_path} ubuntu@{host}:{path}", shell=True)
