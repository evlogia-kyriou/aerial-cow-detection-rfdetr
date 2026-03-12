from rfdetr import RFDETRBase
from roboflow import Roboflow 
import torch
import multiprocessing
from pathlib import Path
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def download_roboflow_dataset(api_key, workspace, project_name, version=1):
    """
    Download dataset directly from Roboflow
    
    Args:
        api_key: Your Roboflow API key
        workspace: Your workspace name
        project_name: Your project name in Roboflow
        version: Dataset version number
    """
    print(f"📥 Downloading from Roboflow: {workspace}/{project_name} (v{version})...")
    
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(version).download("coco")
        
        print(f"✓ Dataset downloaded to: {dataset.location}")
        return Path(dataset.location)
    
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise

def main():
    # Get project root directory
    project_root = Path(__file__).parent
    dataset_path = Path(r"D:\Demo\RF-DETR\aerial-cows-kt2wd-waby-1")
    output_path = project_root / "output" / "first"
    
    ROBOFLOW_API_KEY = "BaeTy8ILqUbULKH4kqgf"  # Replace with your API key
    ROBOFLOW_WORKSPACE = "cctv-d30uc"    # e.g., "john-doe-abc123"
    ROBOFLOW_PROJECT = "aerial-cows-kt2wd-waby-utbtn"        # e.g., "defect-detection"
    ROBOFLOW_VERSION = 1                     # Dataset version number
    
    # Download from Roboflow if dataset doesn't exist
    if not dataset_path.exists():
        print("📂 Dataset not found locally. Downloading from Roboflow...")
        roboflow_path = download_roboflow_dataset(
            api_key=ROBOFLOW_API_KEY,
            workspace=ROBOFLOW_WORKSPACE,
            project_name=ROBOFLOW_PROJECT,
            version=ROBOFLOW_VERSION
        )
        
        # Copy to expected location
        import shutil
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        shutil.move(str(roboflow_path), str(dataset_path))
    else:
        print(f"✓ Using existing dataset at: {dataset_path}")

    # GPU Detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Verify dataset exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    print(f"✓ Dataset found at: {dataset_path}")
    
    model = RFDETRBase(num_classes=1)
    
    model.train(
        dataset_dir=str(dataset_path),
        epochs=80,
        batch_size=4,
        resolution=672,
        multi_scale=True,
        grad_accum_steps=4,
        lr=5e-5,
        lr_scheduler="cosine",
        warmup_epochs=5,
        weight_decay=1e-4,
        use_ema=True,
        ema_decay=0.9998,
        num_workers=6,  # Windows stable setting
        output_dir=str(output_path),
        device=device,
        resume="output/first/checkpoint0059.pth"
    )

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()