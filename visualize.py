import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from PIL import Image

def visualize_dataset(dataset_dir: str, num_images: int = 5):
    ann_file = os.path.join(dataset_dir, "annotations.json")
    if not os.path.exists(ann_file):
        print(f"Error: Could not find annotations.json in {dataset_dir}")
        return

    # Load COCO dataset
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    
    # Create an output directory for the visual checks
    out_dir = os.path.join(dataset_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Drawing masks for {min(num_images, len(img_ids))} images...")

    for img_id in img_ids[:num_images]:
        img_data = coco.loadImgs(img_id)[0]
        img_path = os.path.join(dataset_dir, img_data['file_name'])

        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue

        # Setup Matplotlib figure
        img = Image.open(img_path)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        ax.axis('off')

        # Load annotations for this specific image
        ann_ids = coco.getAnnIds(imgIds=img_data['id'])
        anns = coco.loadAnns(ann_ids)

        # 1. Draw the Segmentation Masks (Built-in pycocotools feature)
        coco.showAnns(anns)

        # 2. Draw the Bounding Boxes and Labels
        for ann in anns:
            [x, y, w, h] = ann['bbox']
            
            # Draw Box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)

            # Add Label Name
            cat_info = coco.loadCats(ann['category_id'])[0]
            ax.text(x, y - 5, cat_info['name'], color='cyan', fontsize=12, weight='bold', 
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

        # Save the result
        safe_name = img_data['file_name'].replace("/", "_")
        save_path = os.path.join(out_dir, f"vis_{safe_name}")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Saved: {save_path}")

    print(f"\nDone! Check the '{out_dir}' folder to see the results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COCO Annotations")
    parser.add_argument("--dataset", required=True, help="Path to your generated dataset directory")
    parser.add_argument("--count", type=int, default=5, help="Number of images to visualize")
    args = parser.parse_args()

    visualize_dataset(args.dataset, args.count)