conda create -n lotus python=3.10 -y
conda activate lotus
pip install -r requirements.txt

pip install scipy easydict


# prepare the dataset
# cd $PATH_TO_RAW_HYPERSIM_DATA
cd /labworking/Users/jyang/data/lotus/hypersim

# Download the tone-mapped images
python ./download.py --contains scene_cam_ --contains final_preview --contains tonemap.jpg --silent

# Download the depth maps
python ./download.py --contains scene_cam_ --contains geometry_hdf5 --contains depth_meters --silent

# Download the normal maps
python ./download.py --contains scene_cam_ --contains geometry_hdf5 --contains normal --silent # running

# download the split file: metadata_images_split_scene_v1.csv and put to the $PATH_TO_RAW_HYPERSIM_DATA
# since I downloaded ealier, I therefore use cp ../../hypersim/metadata_images_split_scene_v1.csv ./

# Process the data with the command
bash utils/process_hypersim.sh