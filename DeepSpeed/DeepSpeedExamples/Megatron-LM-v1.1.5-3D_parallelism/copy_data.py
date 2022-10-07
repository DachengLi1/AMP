import subprocess
import os

path = "/home/ubuntu/datasets/datasets/preprocessed_data/"
#file_name = ["my-gpt2_text_document_test_indexmap_640000ns_1024sl_1234s_doc_idx.npy", "my-gpt2_text_document_test_indexmap_640000ns_1024sl_1234s_sample_idx.npy", "my-gpt2_text_document_test_indexmap_640000ns_1024sl_1234s_shuffle_idx.npy", "my-gpt2_text_document_valid_indexmap_2560000ns_1024sl_1234s_sample_idx.npy", "my-gpt2_text_document_valid_indexmap_2560000ns_1024sl_1234s_shuffle_idx.npy"]
file_name = ["/home/ubuntu/datasets/datasets/preprocessed_data/my-gpt2_text_document_test_indexmap_1280000ns_1024sl_1234s_shuffle_idx.npy"]
hosts = ["ec2-34-218-237-162.us-west-2.compute.amazonaws.com", "ec2-34-213-12-63.us-west-2.compute.amazonaws.com", "ec2-34-209-240-108.us-west-2.compute.amazonaws.com"]
for file in file_name:
    file_path = os.path.join(path, file)
    for host in hosts:
        subprocess.run(f"scp {file_path} ubuntu@{host}:{path}", shell=True)
