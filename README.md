# Recurrent-Memory-Transformer_PreTrained
This repository contains a pre-trained version of the **Recurrent-Memory-Transformer (RMT)**, a memory-augmented transformer model designed to handle long sequences and improve performance on tasks requiring long-term dependencies. The RMT model is based on the standard transformer architecture but incorporates a memory mechanism that allows it to store and process both local and global information more effectively.

This model 

## What is the Recurrent-Memory-Transformer?
The **Recurrent-Memory-Transformer (RMT)** is a variant of the transformer architecture that enhances its ability to process long sequences by using special memory tokens. These memory tokens are added to the input or output sequence, allowing the model to store and pass information between segments of a long sequence through recurrence. This memory mechanism enables the model to handle tasks that require long-term dependencies, such as language modeling and algorithmic tasks, more efficiently than traditional transformers.
Key features of RMT include:
- **Memory-augmented** architecture: Special memory tokens allow the model to retain and process information across long sequences.

- **Recurrence mechanism**: Information is passed between sequence segments, enabling the model to handle sequences longer than the standard transformer context window.

- **Improved performance**: RMT performs on par with or better than models like Transformer-XL on tasks requiring long sequence processing, especially for smaller memory sizes.

For more details, refer to the original paper:
[Aydar Bulatov, Yuri Kuratov, Mikhail S. Burtsev (2022). Recurrent Memory Transformer.](https://arxiv.org/abs/2207.06881)
## Repository Information
This repository is necessary for using the models published on Hugging Face, specifically the gpt2-RMT-(2-8) and gpt2-RMT-(2-8)-mem512 models. If you want to easily try out the Recurrent-Memory-Transformer, please clone and use this repository.
This repository is based on the code from the [recurrent-memory-transformer repository](https://github.com/booydar/recurrent-memory-transformer/tree/framework_accel), which provides the original implementation of the Recurrent-Memory-Transformer.
## Installation
To use the pre-trained Recurrent-Memory-Transformer model, follow these steps to set up the environment and install the required dependencies.
### 1. Clone the Repository
```bash

git clone https://github.com/KotShinZ/Recurrent-Memory-Transformer_PreTrained.git
```

### 2. Set Up a Virtual Environment (Recommended)
It's recommended to use a virtual environment to manage dependencies:
```bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
Install the required packages:
```bash

pip install torch transformers
pip install -e Recurrent-Memory-Transformer_PreTrained
```
## Usage
### 1. use Recurrent-Memory-Transformer
You can load pretrained Recurrent-Memory-Transformer by this code.
```python
model = RecurrentMemoryTransformer.from_pretrained(model_name)
```
The example notebook is use_pretrained_model.ipynb

### 2. use Recurrent-Memory-Transformer
You can train Recurrent-Memory-Transformer by original model
The example notebook is train_original_model.ipynb

## Additional Information
This repository uses code from the recurrent-memory-transformer repository.

The Recurrent-Memory-Transformer is particularly useful for tasks that require processing long sequences or maintaining context over extended inputs, such as document-level language modeling or tasks involving long-term dependencies.

For further customization or to train the model from scratch, refer to the original implementation and the associated paper.

