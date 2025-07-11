

# PAEViT
Our work has been published in the 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing, titled <b>Patch Attention Excitation Based Vision Transformer for Small-Sized Datasets</b>, which you can read at IEEE Xplore <a href=https://ieeexplore.ieee.org/document/10888545>here</a>. To cite the same, please either copy or download the included bib file here, or go to the IEEE Xplore page for more options.

This is a complete guide on how to download, install, and train PAEViT on CIFAR10, CIFAR100, TinyImageNet, or even your own variations of the same. 
---
To get started, you will first need all the files on your local machine. To do this, you must either download or clone this repository.
## Cloning the Repository
You can get the **PAEViT** repository in two ways:
1. **Download:** Download the repository as a ZIP file from GitHub, then extract it.
2. **Clone using Git:** Run the following command in your terminal:
   ```bash
   git clone https://github.com/AkashVermaIN/PAEViT
   ```

Once done, you should have a folder named `PAEViT` containing all the required files to start training. However, you still need to install the prerequisites such as PyTorch and other modules if you don't usually perform similar tasks. 

---

## Setting Up the Environment
Before you can begin training, you need to install the necessary dependencies, such as **PyTorch** and other Python libraries. To do this, you can simply go inside the PAEViT folder, then right click on any blank space and click "Open in Terminal", then run the following command (or just change directory inside existing terminal session to make sure you're in the PAEViT folder):

### Steps:
1. **Navigate to the PAEViT folder:**
   - Open the folder, right-click on any blank space, and select **"Open in Terminal"**.
   - Or use the terminal and run:
     ```bash
     cd PAEViT
     ```

2. **Install Dependencies:**
   Run the following command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```

This will install all the required Python libraries for the project. In case it fails for any reason, you can simply go through the requirements.txt file and try to install each dependency one at a time.

---

## Running the Project
Once all the dependencies are installed, you’re ready to start training. Once done, you should simply be able to double click on the run file inside the `PAEViT` folder, then execute any of the training configurations.  To get started:
1. Open the **PAEViT** folder.
2. Double-click the **run** file (recommended) to start the PAEViT Training program and select the appropriate options according to your requirements, or open terminal to run main.py along with all the appropriate arguments (for highly customized testing).

You can use any of the predefined training configurations provided in the repository or create your own.

---

## Customizing the Model
You can modify the model configurations to experiment with different setups:
- Open the `custom.py` file located in the `models` folder.
- Make your changes.
- Save the file and run the training process again while selecting the 'custom' model in the PAEViT Training program.

---

## Viewing Results
After the training is complete, all results, including model checkpoints, logs, and metrics, will be saved in the following directory in separate folders:
```
./CheckpointResults/
```

You can review these results to analyze the performance of the experiments.

## If you find any of our work helpful, please cite our paper in your publications. Thank you.
