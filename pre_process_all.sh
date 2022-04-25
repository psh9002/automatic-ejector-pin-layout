source ~/miniconda3/etc/profile.d/conda.sh 
python tools/rerun.py tools/convert_stp_to_stl.py ~/Dataset/processed_data/datasets/Mold_Cover_Rear
conda activate samsung && ./blender-2.90.0-linux64/blender "dataset_generator.blend" --background --python "tools/create_blender_data.py" -- ~/Dataset/processed_data/datasets/Mold_Cover_Rear
conda activate cadquery && python tools/rerun.py tools/create_line_dataset.py ~/Dataset/processed_data/datasets/Mold_Cover_Rear
conda activate samsung && python tools/preprocess_png.py ~/Dataset/processed_data/datasets/Mold_Cover_Rear

python tools/rerun.py tools/convert_stp_to_stl.py ~/Dataset/processed_data/datasets/Mold_Chassis_Rear
conda activate samsung && ./blender-2.90.0-linux64/blender "dataset_generator.blend" --background --python "tools/create_blender_data.py" -- ~/Dataset/processed_data/datasets/Mold_Chassis_Rear
conda activate cadquery && python tools/rerun.py tools/create_line_dataset.py ~/Dataset/processed_data/datasets/Mold_Chassis_Rear
conda activate samsung && python tools/preprocess_png.py ~/Dataset/processed_data/datasets/Mold_Chassis_Rear

python tools/rerun.py tools/convert_stp_to_stl.py ~/Dataset/processed_data/datasets/Press_Chassis_Rear
conda activate samsung && ./blender-2.90.0-linux64/blender "dataset_generator.blend" --background --python "tools/create_blender_data.py" -- ~/Dataset/processed_data/datasets/Press_Chassis_Rear
conda activate cadquery && python tools/rerun.py tools/create_line_dataset.py ~/Dataset/processed_data/datasets/Press_Chassis_Rear
conda activate samsung && python tools/preprocess_png.py ~/Dataset/processed_data/datasets/Press_Chassis_Rear

