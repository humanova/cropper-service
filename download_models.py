import gdown
import os

bn_url = "https://github.com/OPHoperHPO/image-background-remove-tool/releases/download/3.2/u2net.pth"
bn_dir = os.path.join("models", "u2net")

bnp_url = "https://github.com/OPHoperHPO/image-background-remove-tool/releases/download/3.2/u2netp.pth"
bnp_dir = os.path.join("models", "u2netp")

if not os.path.exists(bn_dir):
    os.makedirs(bn_dir)

if not os.path.exists(bnp_dir):
    os.makedirs(bnp_dir)

gdown.download(bn_url, os.path.join(bn_dir, "u2net.pth"), quiet=False)
gdown.download(bnp_url, os.path.join(bnp_dir, "u2netp.pth"), quiet=False)