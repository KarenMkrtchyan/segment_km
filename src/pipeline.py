import yaml
from pathlib import Path
from Segmenter_Module import Segmenter
from extraction_module.Extraction_Module import Extractor
from extraction_module.Data_Handler import CustomImageDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

def main():
    with open(Path('./src/config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
        
    segmentor_model = Segmenter(
                      pretrained_model=config['segmentation_model'], 
                      device=config['device'], 
                       data_dir=config['data_dir'],
                      image_extension=config['image_extension'],
                      output_dir=config['output_dir'],
                      offset=config['offset'],
                      save_masks=config['save_masks']
                      )
    
    extraction_model = Extractor(
        model_path=config['extraction_model'],
        device=config['device']
    )
        
    print("\n📠 Segmenting frames in directory:")
    images = segmentor_model.load_images(Path(config['data_dir'])) # TODO: Run this on multiple cores

    print("\n📠 Combining 4 scans into 1 image ...")
    frames=segmentor_model.combine_images(images)

    print("\n📠 Computing masks ...")
    masks, _, _ = segmentor_model.segment_frames(frames)
    masks = np.array(masks)
    del frames
    print("\n📠 Cropping images ...")

    offset = 10 # for sample data 10, set to config['offset'] for actual run
    dapi = images[:offset] # these are all curently 1044*1362
    ck = images[offset:2*offset]
    cd45 = images[2*offset:3*offset]
    fitc = images[3*offset:4*offset]
    images = np.stack((dapi, ck, cd45, fitc), axis=1) # N 4 H W 
    
    image_crops, mask_crops, centers = segmentor_model.get_cell_crops(masks, images)
    del images

    print("\n📠 Doing all the data loader nonsense ...")
    # have
    # image_crops -> 3 channels: RGB                  (N, 75, 75, 3)
    # mask_crops  -> 0 channels: instance mask        (N, 75, 75)

    # need
    # image       -> 4 channels: dapi, ck, cd45, fitc (N, 4, 75, 75)
    # mask        -> 1 channels: binary mask          (N, 1, 75, 75)

    # plan 
    # load in all the slides and save them
    # get composites 
    # use compositres to segment and get mask
    # use mask to crop the orignial 4 channel slides
    # feed new crops to data loader
    # feed dataloader to extractor

    dataset = CustomImageDataset(image_crops, mask_crops, labels=np.zeros(image_crops.shape[0]), tran=False)
    dataloader = DataLoader(dataset, batch_size=config['inference_batch'], shuffle=False)

    print("\n📠 Extracting Features ...")
    embeddings = extraction_model.get_embeddings(dataloader)
    embeddings_np = embeddings.cpu().numpy()

    # --- FOR DEBUG ---

    # reducer = umap.UMAP(n_components=2, random_state=42)
    # embeddings_2d = reducer.fit_transform(embeddings_np)
    # plt.figure(figsize=(8,6))
    # plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], cmap='Spectral', s=5)
    # plt.title("UMAP projection of embeddings")
    # plt.xlabel("UMAP-1")
    # plt.ylabel("UMAP-2")
    # plt.colorbar()
    # plt.show()

    embeddings_df = pd.DataFrame(
        embeddings_np.astype('float16'),
        columns=[f'z{i}' for i in range(embeddings.shape[1])])
    
    embeddings_df.insert(0, "slide id", 0)
    embeddings_df.insert(1, "center_x", centers[:, 0])
    embeddings_df.insert(2, "center_y", centers[:, 1])

    embeddings_df.to_parquet("data/processed/embeddings.parquet.gzip", compression="gzip")

    print("Its over")

if __name__ == "__main__":
    main()