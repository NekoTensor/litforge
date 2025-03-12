# Litforge Library for LLaMA Model

Litforge is a lightweight Python library that provides a custom transformer implementation inspired by LLaMA. It leverages Hugging Face’s model hub to load pre-trained weights, configurations, and tokenizers, while implementing a simplified transformer forward pass using PyTorch. Designed to be modular and extensible, Litforge is easy to integrate into your projects for inference, debugging, or internal analysis.

## Features

- **Hugging Face Integration:**  
  Automatically downloads and loads model weights, configuration, and tokenizer data directly from Hugging Face.

- **Custom Transformer Implementation:**  
  Implements key components of the transformer architecture:
  - Token embedding lookup
  - Self-attention with cached key/value support
  - Residual connections and feed-forward networks with GELU activation
  - Final layer normalization with weight tying to produce output logits

- **Visualization Ready:**  
  Exposes internal model parameters (such as weight matrices) so you can easily analyze or visualize them in your own workflow.

- **Modular and Extensible:**  
  Easily integrate the library into larger projects or extend the functionality—for example, by modifying the forward pass or implementing alternative decoding strategies.

## Installation

##  Clone from GitHub

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/NekoTensor/litforge.git
cd litforge
pip install -r requirements.txt
```
Alternatively, install it locally in editable mode:
```bash
pip install -e .
```
## Using Litforge as a Library

Once installed, you can import and integrate Litforge in your projects. Follow these steps:

### 1. Import and Initialize

Import the `LitForge` class and instantiate it with a valid Hugging Face model identifier:

```python
from litforge import LitForge
```

# Initialize the model using a Hugging Face model name (e.g., "meta-llama/Llama-3.2-1B")
model = LitForge("meta-llama/Llama-3.2-1B")

This call downloads the model weights, tokenizer, and configuration from Hugging Face and prepares the model for inference or further analysis.

### 2. Integrate into Your Workflow

**Model Loading and Inference**  
The `LitForge` object encapsulates all functionalities required to run forward passes. Integrate it into your data processing pipeline wherever model inference is needed.

**Accessing Internal Parameters**  
The library exposes the underlying weight tensors via the `weights` attribute. For example, you can access the embedding weights as follows:

```python
embedding_weights = model.weights["model.embed_tokens.weight"]
print("Embedding weight shape:", embedding_weights.shape)
```
**Customization and Extension**  
Litforge is designed with simplicity in mind, making it easy to extend the `LitForge` class. Modify the forward pass, implement alternative decoding strategies, or integrate additional functionalities as needed.

**Configuration and Customization**  
The model configuration is loaded directly from the Hugging Face model and is accessible via the `config` attribute:

```python
print(model.config)
```
Use this configuration to guide architecture adjustments or debug issues related to model parameters.

## Repository Structure

- **litforge.py** – Contains the `LitForge` class implementation.
- **notebook.ipynb** – A Google Colab notebook demonstrating how to authenticate, load a model, visualize internal parameters, and use the library.
- **README.md** – This file.

## Contributing

Contributions and suggestions are welcome! If you’d like to extend or improve the library, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Hugging Face Transformers
- PyTorch

Repository: [https://github.com/NekoTensor/litforge](https://github.com/NekoTensor/litforge)



