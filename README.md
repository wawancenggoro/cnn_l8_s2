# Supervised Conversion from Landsat-8 Images to Sentinel-Images with Deep Learning

This is the official code for the paper titled "Supervised Conversion from Landsat-8 Images to Sentinel-Images with Deep Learning."

The dataset can be downloaded here. Put the dataset in the directory named "dataset" at the same level as the code directory.

To train a model, run this command:

'''
python main.py -m [MODEL NAME]
'''

The model name options are: "sub", "submax", "trans", and "transmax".

To calculate the performance of each model, put the model_path.pth file generated by the training code the directory named as the corresponding model name in the save directory. Afterward, run this command:

'''
python calculate_performance.py
'''
