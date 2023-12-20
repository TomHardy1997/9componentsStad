import os
from concurrent.futures import ThreadPoolExecutor
from histolab.slide import Slide
from histolab.tiler import GridTiler
import argparse
import concurrent.futures
from tqdm import tqdm  

parser = argparse.ArgumentParser(description='use the histolab package to handle the svs data.')
parser.add_argument('--data', type=str, help='data dir for input data')
parser.add_argument('--output', type=str, help='data dir for output data')
parser.add_argument('--pixel', type=int, help='pixel size for overlap')
args = parser.parse_args()

# Specify the input and output folder paths
base_path = args.data
output_folder = args.output
os.makedirs(output_folder, exist_ok=True)

# Get the SVS image file path for all inputs
svs_files = [os.path.join(base_path, filename) for filename in os.listdir(base_path) if filename.endswith(".svs")]


grid_tiles_extractor = GridTiler(
   tile_size=(224, 224),
   level=0,
   check_tissue=False,  # default
   pixel_overlap=args.pixel,  # default
   suffix=".png"  # default
)

def process_svs(svs_file):
    filename = os.path.basename(svs_file).replace('.svs', '')
    processed_path = os.path.join(output_folder, filename)
    if os.path.exists(processed_path):
        return
    
    os.makedirs(processed_path, exist_ok=True)
    slide = Slide(svs_file, processed_path)

    print(f"Slide name: {slide.name}")
    print(f"Dimensions at level 0: {slide.dimensions}")

    grid_tiles_extractor.extract(slide)

    print(f"Processing {slide.name} -> {output_folder}")

# Use thread pools to process SVS images in parallel
with ThreadPoolExecutor(max_workers=8) as executor:
    # Use tqdm as a context manager to coutput_folder
    with tqdm(total=len(svs_files), desc='Processing SVS Files') as pbar:
        def update_pbar(*_):
            pbar.update()

        futures = {executor.submit(process_svs, svs_file): svs_file for svs_file in svs_files}
        for future in concurrent.futures.as_completed(futures):
            future.add_done_callback(update_pbar)

print("Processing complete.")
