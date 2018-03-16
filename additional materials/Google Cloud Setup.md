# Google Cloud Setup

## Setup Google VM instance

You need to upgrade your account, then go to **Quotas**

request a GPU **NVIDIA Tesla P100** in **us-central1-c** zone

Now create a new VM instance with:

- 4 CPUs and 16gb of RAM
- 1 NVIDIA Tesla P100
- Ubuntu 16 with 200gb of storage
- Enable HTTTP and HTTPS traffic

## Environment

Run these commands to update your environment:

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y build-essential
```

Ubuntu comes with Python pre installed, so do not need to download it. 

## Cuda

Download Cuda by running:

```bash
cd /tmp
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
```

Now installation process:

```bash
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda-8-0
```

To add some environment variables to your `.bashrc` file run:

```bash
cat <<EOF >> ~/.bashrc
export CUDA_HOME=/usr/local/cuda-8.0
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64
export PATH=\${CUDA_HOME}/bin:\${PATH}
EOF
```

Source the `.bashrc` file:

```bash
source ~/.bashrc
```

**Testing:**

Run `nvidia-smi` to check if drivers are installed properly

## cuDNN

First you need to download the **cuDNN** on your local machine and transfer it to your Google Cloud instance. 

I have mine uploaded to Google Cloud Storage, so:

```bash
gsutil cp gs://deep-research/*.gz .
```

This will download the cuDNN file from the Google Storage. 

Once it is downloaded, we need to uncompress the file and copy the cuDNN library to Cuda:

```bash
cd /tmp
tar -xvf cudnn-8.0-linux-x64-v5.1.tar
sudo cp -P cuda/include/cudnn.h $CUDA_HOME/include
sudo cp -P cuda/lib64/libcudnn* $CUDA_HOME/lib64
sudo chmod u+w $CUDA_HOME/include/cudnn.h
sudo chmod a+r $CUDA_HOME/lib64/libcudnn*
```

## The Project:

### Cloning the repo and downloading the data:

Move to home folder and clone the repo:

```bash
cd /home/dshakhbu
git clone -b Final https://github.com/Denisolt/DeepLearning_research.git
```

Now delete the data in the repo with old data, create a new folder and download the data from Google Cloud Storage:

```bash
cd DeepLearning_research
cd data
rm -r output
mkdir output
cd mkdir
gsutil cp gs://deep-research/data*.npz .
```

### TensorDB:

TensorDB is a package for storing Tensors based on **MongoDB**

To install MongoDB:

```bash
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2930ADAE8CAF5059EE73BB4B58712A2291FA4AD5
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.6 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.6.list
sudo apt-get update
sudo apt-get install -y mongodb-org
```

## Packages:

```bash
cd home/dshakhbu/DeepLearning_research
pip install -r req.txt
```

## Running:

In first SSH window:

Start MongoDB with a specific port:

```
sudo mongod —port 27018
```

Open a new SSH window.

Now a little problem, since my `batch_train.sh` is wrong:

```bash
rm /home/dshakhbu/DeepLearning_research-master/batch_train.sh
cd /home/dshakhbu
git clone https://github.com/akaraspt/deepsleepnet.git
mv deepsleepnet/batch_train.sh /home/dshakhbu/DeepLearning_research
```

Run the command:

```bash
chmod +x batch_train.sh
./batch_train.sh data/eeg_fpz_cz/ output 20 0 19 0
```



