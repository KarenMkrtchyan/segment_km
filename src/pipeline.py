import yaml
from pathlib import Path
from segmentation_module.Segmenter import Segmenter
from extraction_module.Extraction_Module import Extractor
from downtream_tasks.spikein.SpikeIn import SpikeIn
from extraction_module.Data_Handler import CustomImageDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


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
                      )   
        
    print("\n📠 Segmenting frames in directory: ", config['data_dir'])
    segmentor_model.load_data(Path(config['data_dir'])) 
    print("\n📠 Combining 4 scans into 1 image ...")
    segmentor_model.preprocess()
    print("\n📠 Computing masks ...")
    segmentor_model.segment()
    print("\n📠 Cropping images ...")
    image_crops, mask_crops, centers = segmentor_model.postprocess()
    del segmentor_model

    print("\n📠 Preping segmentation output for extraction input ...")
    dataset = CustomImageDataset(image_crops, mask_crops, labels=np.zeros(image_crops.shape[0]), tran=False)
    dataloader = DataLoader(dataset, batch_size=config['inference_batch'], shuffle=False)

    extraction_model = Extractor(
        model_path=config['extraction_model'],
        device=config['device']
    )
    
    print("\n📠 Extracting Features ...")
    embeddings = extraction_model.extract(dataloader)
    embeddings_np = embeddings.cpu().numpy()

    print("\n🔍 Predicting cell types ...")
    predictions = []
    probabilities = []

    spikein_model = SpikeIn(model_path=config['spikein_model'])

    for embedding in embeddings_np:
        pred = spikein_model.prediction(embedding)
        prob = spikein_model.probability(embedding)
        predictions.append(pred)
        probabilities.append(prob)

    probabilities = np.array(probabilities)  
    predictions = np.array(predictions)


    print("\n📠 Saving Features ...")
    results = pd.DataFrame(
        embeddings_np.astype('float16'),
        columns=[f'z{i}' for i in range(embeddings.shape[1])])
    
    results.insert(0, "slide id", 0)
    results.insert(1, "center_x", centers[:, 0])
    results.insert(2, "center_y", centers[:, 1])
    results.insert(3, "pred", predictions)
    results.insert(4, "prob1", probabilities[:, 0])
    results.insert(5, "prob2", probabilities[:, 1])
    results.insert(6, "prob3", probabilities[:, 2])

    results.to_parquet("data/processed/embeddings_with_pred.parquet.gzip", compression="gzip")

    print("\n Pipeline finished \n")

if __name__ == "__main__":
    main()