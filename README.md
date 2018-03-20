# MultiChannelSleepNet

## Authors: Denisolt Shakhbulatov, Arshit Arora, Dr. Nabi Sertac Artan

The objective of the research is to improve the accuracy of the sleep stage classification by using multiple channels of different EEG, EOG, EMG, and respiratory signals and using popular Deep Learning approaches. Most of the time researchers focus on EEG data only, there are only a few papers that have combined different types of channels together.

## Working with EEG

There has been interesting work done in creating a novel approach to work with EEG data by Bashivan in 2016. His research team has worked with Classification of the Memory Task Load. They have turned the raw EEG data into an **image**, using FFT and Polar Projection. Later they fed the images into a Convolutional Neural Network with LSTM memory cells, that gave them a test error of 8.89%. Another approach to working with EEG data is transforming the raw signal into **Spectograms**. This has been done by Biswal in 2017 and many others. Their approach is to use created models of image classification algorithms (such as ImageNet, Inception) to classify the sleep stages. The best results were shown in approaches that used **raw EEG data**. The paper by Tsinalis showed 86% of the accuracy.

## Architecture of the Network

In order to perform classification, there shall be specific features that the algorithm has to look for. In most of the cases, the features are defined by the researchers or experts in the field. Research by Supratak in 2017 has tried Representation Learning. It is a deep learning technique for feature extraction. The rest of researchers vary on either using RNN or CNN or both. There is only one research that looked into Multivariate Networks with multiple inputs. This was done by Chambon in 2017. They have created a 2 Convolutional Networks, one for EEG/EOG data, and the second one for EMG. The results of the network were then combined to give a high result of 84% accuracy. The approach with the highest accuracy used a simple Convolutional Neural Network with single raw EEG data input.

## Our approach

Dr. Artan and I have worked on preprocessing multiple channels of data. We have combined 2 channels of EEG (Fpz-Cz and Pz-Oz), EOG, EMG and respiratory data. They were combined into a multidimensional array or a tensor. Where each layer represented different channels, preserving the independence of the data in each channel, although combining all of the data together for sleep stage classification. The network that was created was based on the DeepSleepNet by Supratak in 2017 to mimic their representation learning. The main difference between their work and ours is that they used only one channel of the EEG data. Our architecture is a network consisting of 2 main parts: representation learning and classification. First, we extract features using CNN and then perform classification using RNN. The Representation Learning is divided into 2 parts: the first one is set to half of the sampling rate and stride size to 1/16 to detect patterns in the signal, the second one is set to 4 times of the sampling rate and stride to half of the sampling rate to better capture the frequency components of the signal. Below is the image of the network:
![alt text](https://github.com/Denisolt/MultiChannelSleepNet/blob/master/additional%20materials/images/diagram.png?raw=true)

## Additional Materials

- [Google Cloud VM with GPU Setup](https://github.com/Denisolt/MultiChannelSleepNet/blob/master/additional%20materials/Google%20Cloud%20Setup.md)
- [papers:](https://github.com/Denisolt/MultiChannelSleepNet/tree/master/additional%20materials/papers)
   - [Paper Summary](https://github.com/Denisolt/MultiChannelSleepNet/blob/master/additional%20materials/PapersSummary.csv)
   - [Sleep](https://github.com/Denisolt/MultiChannelSleepNet/tree/master/additional%20materials/papers/sleep)
   - [Other](https://github.com/Denisolt/MultiChannelSleepNet/tree/master/additional%20materials/papers/etc)
-  [Grant Proposal](https://github.com/Denisolt/MultiChannelSleepNet/blob/master/additional%20materials/NYIT%20Undergraduate%20Research%20and%20Entrepreneurs%20Program%20Mini%20Grant%20Proposal%20(dragged).pdf)
- [Deep Learning Presentation](https://github.com/Denisolt/MultiChannelSleepNet/tree/master/additional%20materials/presentation)
- [Data](https://physionet.org/physiobank/database/capslpdb/)

## Environment ##

- Ubuntu 16.04
- MongoDB
- CUDA toolkit 8.0 and CuDNN v5
- Python 2.7
- [tensorflow-gpu (0.12.1)](https://www.tensorflow.org/versions/r0.12/get_started/os_setup)
- [tensorlayer](https://github.com/zsdonghao/tensorlayer)
- pandas


## Prepare the environment

- Setup Ubuntu 16.04
- Setup MongoDB
- Download and install Cuda 8.0 and CuDNN v5

## Prepare dataset ##

For the [Sleep-EDF](https://physionet.org/pn4/sleep-edfx/) dataset, you can run the following scripts to download SC subjects.

    cd data
    chmod +x download_physionet.sh
    ./download_physionet.sh

Then run the jupyter notebook  to extract all the channels and their corresponding sleep stages.

## Training a model ##

Start MongoDB:

```
mongod --port [port number from deesleep/trainer.py]
```

Run this script to train a AlexandersSleep model for the first fold of the 20-fold cross-validation.

    python train.py --data_dir data/output --output_dir output --n_folds 20 --fold_idx 0 --pretrain_epochs 100 --finetune_epochs 200 --resume False

You need to train a AlexandersSleep model for every fold (i.e., `fold_idx=0...19`) before you can evaluate the performance. You can use the following script to run batch training

    chmod +x batch_train.sh
    ./batch_train.sh data/output/ output 20 0 19 0

## Scoring sleep stages ##
Run this script to determine the sleep stages for the withheld subject for each cross-validation fold.

    python predict.py --data_dir data/eeg_fpz_cz --model_dir output --output_dir output

The output will be stored in numpy files.


## Get a summary ##
Run this script to show a summary of the performance of our AlexandersSleep compared with the state-of-the-art hand-engineering approaches. The performance metrics are overall accuracy, per-class F1-score, and macro F1-score.

    python summary.py --data_dir output
