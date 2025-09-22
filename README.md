## Table of Contents
- [Installation](#installation)
- [Usage](#usage)


## Installation
Follow these steps to install the RDM framework:

1. **Clone recursively:**
    ```bash
    git clone --recurse-submodules 
    cd RDM
    ```

2. **OpenAI key:** To run the GPT-4/LLaMA, you will need to configure an OpenAI key. This can be done by signing up for an account e.g. here, and then creating a key in account/api-keys. Create a file .env in the root of this project and store the key in it.
    ```
    echo YOUR_OPENAI_API_KEY_HERE > key.env
    ```

3. **Prepare environment:**
   ```bash
    conda create -n rdm python=3.11 -y
    conda activate rdm
    conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install -c conda-forge pybullet
    pip install -r requirements.txt
   ```

4. **Download pretrained model (tools)**
    ```bash 
    bash download.sh
    ```
## Usage

- Check `demo.ipynb` notebook for simple demo inference. This notebook includes detailed instructions and executing queries with visualization.
- All toolset are in `src/toolset.py`
- Run run simple inference using `src/pipeline.py`.

