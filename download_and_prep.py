import os
import shutil
import urllib.request
import zipfile

def initialize_plantvillage_pipeline():
    """
    Downloads and structures the public PlantVillage benchmark database
    directly into organized, local category folders.
    """
    # 1. Path definitions
    root_dir = "plantvillage_data"
    zip_path = os.path.join(root_dir, "dataset.zip")
    extract_path = os.path.join(root_dir, "raw_extraction")
    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    # 2. Download from official public data repository
    # Storing a consolidated 5,000 image balanced slice for rapid edge iteration
    url = "github.com"
    
    if not os.path.exists(zip_path):
        st_status = print("🛰️ Connecting to PlantVillage Network... Downloading dataset zip layer.")
        urllib.request.urlretrieve(url, zip_path)
        print("✓ Download complete.")
        
    # 3. Extract the archive elements
    if not os.path.exists(extract_path):
        print("📦 Decompressing matrix components...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
    # 4. Relocate structural folders to clean up paths
    # Targets the color leaf subdirectory containing separated classes
    source_inner = os.path.join(extract_path, "PlantVillage-Dataset-master", "raw", "color")
    final_dataset_path = os.path.join(root_dir, "structured_dataset")
    
    if os.path.exists(source_inner) and not os.path.exists(final_dataset_path):
        shutil.copytree(source_inner, final_dataset_path)
        print(f"🌟 Success. Labeled plant classes isolated at: {final_dataset_path}")
        
    return final_dataset_path

if __name__ == "__main__":
    initialize_plantvillage_pipeline()
