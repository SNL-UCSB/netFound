# netFound-v2: Production-Ready Network Encoder Model
This repository contains the **source code for netFound**, a foundation model for network data developed by the **Systems & Networking Lab (SNL) at UC Santa Barbara**.
## Description
netFound is designed to learn **spatial-temporal relationships** from raw network traffic, making it a powerful tool for network analysis, anomaly detection, and traffic prediction. 
## :key: Key Features 
- **Raw Packet Processing**: Directly processes raw *PCAP* files as input, enabling full-scale network traffic analysis. 
- **Pretraining on Unlabeled Data**: Requires pretraining on large-scale, *unlabeled* network telemetry datasets, leveraging *self-supervised learning*. 
- **Hierarchical Transformer Architecture**: Captures both *packet bursts* and *flow-level behavior*, ensuring robust feature extraction. 
- **Metadata-Aware Processing**: Integrates **burst-level metadata** such as: 
  - Inter-arrival time*
  - Number of bytes per burst
  - Packet-level timing and structure 
## :pushpin: Why Use netFound? 
netFound is part of a larger effort to develop **self-driving networks** - autonomous, adaptive network systems that require minimal human intervention. By leveraging *network foundation models*, we aim to improve the efficiency and scalability of *AI-powered Network Operations (AIOps)*. 
Corresponding paper: https://arxiv.org/abs/2310.17025
## Checkpoints

We provide three different checkpoint sizes:
- https://huggingface.co/snlucsb/netFound-small
- https://huggingface.co/snlucsb/netFound-base
- https://huggingface.co/snlucsb/netFound-large

You can provide the checkpoints with the corresponding `--size` flag: `small`, `base`, or `large`.

These checkpoints are pretrained on ~10bln tokens of the real-world network traffic of the University of California, Santa Barbara.  

## :rocket: Quick Start: Running netFound with Docker & Makefile 
The *easiest way* to verify that the *preprocessing code and model work correctly* is to use the *provided Dockerfile and Makefile*. This setup ensures a *reproducible environment* with all dependencies installed and includes a *small test dataset* to validate the pipeline. 
### :hammer_and_wrench: **Step 1: Build the Docker Container** 
Run the following command to build the container: 
```sh
docker build -t netfound:test .
``` 
This will create a Docker image named `netfound:test`, including the *source code* and a *test dataset* located in `data/test`. 
### :arrow_forward: **Step 2: Run the Container** 
Start an interactive session inside the container: 
```sh
docker run -it netfound:test
``` 
This will launch a shell inside the container in the `/workspace` directory. 
### :zap: **Step 3: Run the Full Pipeline** 
Inside the container, execute: 
```sh
make all
``` 
This will sequentially run the following *four steps* on the test dataset: 
1. **Testing**: Runs unit-tests to verify the correctness of the source code.
2. **Preprocessing**: Converts raw PCAP files into a format suitable for training. 
3. **Pretraining**: Runs *self-supervised learning* on preprocessed data. 
4. **Finetuning**: Adapts the model for downstream tasks using the preprocessed test dataset. 

## :building_construction: **Understanding the Makefile & Dockerfile** 
The *Dockerfile and Makefile* automate the pipeline and provide a structured workflow: 
### :pushpin: **Dockerfile** 
- Creates a *containerized environment* with all necessary dependencies installed. 
- Ensures consistent execution across different systems. 
### :pushpin: **Test Dataset (`data/test/`)** 
- Contains *raw PCAP files* formatted for preprocessing. 
- Used to verify the pipeline’s functionality. 
### :pushpin: **Makefile Structure** 
- **`make preprocess`**: 
  - Filters, splits, and tokenizes the raw packet data. 
- **`make pretrain`**: 
  - Runs **self-supervised pretraining** on the preprocessed dataset. 
- **`make finetune`**: 
  - Trains the model on task-specific labeled data. 
# :rocket: Bring Your Own Data (BYOD) 
To train or fine-tune **netFound** on your own dataset, follow the steps below to **preprocess and tokenize your PCAP files**. 
## :pushpin: Preprocessing Your Dataset 
The easiest way to preprocess your dataset is to use the **`scripts/preprocess_data.py`** script. 
### :open_file_folder: Folder Structure for Pretraining 
Organize your dataset as follows: 
```
folder_name/
 ├── raw/
 │   ├── file1.pcap
 │   ├── file2.pcap
 │   ├── ...
```
Then, run the following command: 
```bash
python3 scripts/preprocess_data.py --input_folder folder_name --action pretrain --tokenizer_config configs/TestPretrainingConfig.json --combined
```
:small_blue_diamond: **What happens next?** 
- The script will generate **intermediate folders** (`extracted`, `split`, etc.). 
- The resulting **tokenized data** will be stored in the `"tokens"` folder. 
- The **`--combined`** flag merges all tokenized files into a single **Arrow** file (useful for training). 
- If you **remove `--combined`**, multiple **Arrow** files (one per PCAP) will be created—this is beneficial for parallel processing across multiple nodes. 
- You can **modify the tokenizer configuration** (`configs/TestPretrainingConfig.json`) to control how internal and external IPs are handled. 
### :open_file_folder: Folder Structure for Fine-Tuning 
To fine-tune netFound, structure your dataset into **class-separated folders**, where **folder names should be integers** (used as class labels). 
```
raw/
 ├── benign/
 │   ├── class1_sample1.pcap
 │   ├── class1_sample2.pcap
 │   ├── ...
 ├── malicious/
 │   ├── class2_sample1.pcap
 │   ├── class2_sample2.pcap
 │   ├── ...
```
Run the preprocessing script again, changing the `--action` to `finetune`: 
```bash
python3 scripts/preprocess_data.py --input_folder folder_name --action finetune --tokenizer_config configs/DefaultConfigNoTCPOptions.json --combined
```
:small_blue_diamond: **Fine-Tuning Notes:** 
- Class labels should be strings (e.g., `benign, malicious, 1, 42, ...`). 
- The resulting **Arrow files** will include a `"labels"` column. 
- You can **manually edit the `"labels"` column** for **custom class adjustments** (including regression tasks).
- As default validation data split does not shuffle the data file before the split, if your data is not shuffled, please use `scripts/shuffler.py` to shuffle the train file to ensure that the resulting test file contains instances of different classes.
## :wrench: Advanced Options 
### **Handling TCP Options** 
- To include **TCPOptions** in your preprocessed data, use the `--tcp_options` flag: 
```bash
python3 scripts/preprocess_data.py --input_folder folder_name --action pretrain --tokenizer_config configs/DefaultConfigWithTCPOptions.json --combined --tcp_options
```
- **Prerequisite**: Your dataset must be **preprocessed with an additional flag** when using `3_extract_fields.py`: 
```bash
python3 scripts/3_extract_fields.py input.pcap output.pcap 1
```
- Ensure you use a **config file that includes TCPOptions processing** (e.g., `configs/DefaultConfigWithTCPOptions.json`). 
## How to cite
```
@misc{guthula2024netfoundfoundationmodelnetwork,
      title={netFound: Foundation Model for Network Security},
      author={Satyandra Guthula and Roman Beltiukov and Navya Battula and Wenbo Guo and Arpit Gupta and Inder Monga},
      year={2024},
      eprint={2310.17025},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2310.17025},
}
```
## Acknowledgements
NSF Awards CNS-2323229, OAC-2126327, and OAC2126281 supported this work. This research used resources at the National Energy Research Scientific Computing Center (NERSC), a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award NERSC DDR-ERCAP0029768. Additionally, we would like to thank Cisco Research for their support.
