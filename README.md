cd path/to/code

# brdf fitting
bash train_scripts/run_fit_brdf.sh

# inverse rendering
bash train_scripts/train_rgb2x.sh

# forward rendering
bash train_scripts/train_x2rgb.gbuffer.sh
bash train_scripts/train_x2rgb.polarization.sh