# Assignment02 - DIP with PyTorch 
**冯洋SC25005002**
## 1. Data_poisson

>### [Data_possion01]<img src="pics/data_poisson_result01.png" alt="alt text" width="800">

## Training

This project does not require model training


## Evaluation

To evaluate my model on ImageNet, run:

```eval
python run_blending_gradio.py
```

## Results



### [Data_possion02 ]
<img src="pics/data_poisson_result02.png" alt="alt text" width="800">


### [Data_possion03 ]
<img src="pics/data_poisson_result03.png" alt="alt text" width="800">

## 2. Pix2pix

## Requirements

To install requirements:

```setup
pip install torch numpy opencv-python gradio
```

## Training

To train the model(s) in the paper, run this command:

```
bash download_facades_dataset.sh --python train.py
```

## Results

Our model achieves the following performance on :
### [first_loss]
<img src="pics/first_loss.png" alt="alt text" width="800">

### [end_loss]
<img src="pics/end_loss.png" alt="alt text" width="800">

We can see that the train loss decrease from 0.7523 to 0.0458 and the val loss decrease from 0.7126 to 0.3675

Here is the result of train 13 and train 279 :

### [train result_13]
<img src="pics/train_result_13.png" alt="alt text" width="800">

### [train result_279]
<img src="pics/train_result_279.png" alt="alt text" width="800">

Here is the result of val 13 and val 279 :

### [val result_13]
<img src="pics/val_result_13.png" alt="alt text" width="800">

### [val result_279]
<img src="pics/val_result_279.png" alt="alt text" width="800">




