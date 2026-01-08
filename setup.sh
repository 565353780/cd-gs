cd ..
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git
git clone --depth 1 --recursive https://github.com/NVlabs/tiny-cuda-nn.git

pip install ninja

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

pip install plyfile tqdm matplotlib opencv-python joblib timm

cd pytorch3d
python setup.py install

cd ../tiny-cuda-nn/bindings/torch
python setup.py install

cd ../../../cd-gs/submodules/diff-gaussian-rasterization-depth-acc
python setup.py install

cd ../simple-knn
python setup.py install
