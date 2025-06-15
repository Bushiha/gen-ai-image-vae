# Generative AI Image Model — Variational Autoencoder (VAE)

A simple generative AI model that learns to generate handwritten digit images using a Variational Autoencoder (VAE) trained on the MNIST dataset.

## 📦 Project Structure

```
src/
├── model.py        # VAE architecture
├── train.py        # Training script
└── generate.py     # Image generation script
images/             # Output generated images
models/             # Saved trained models
requirements.txt    # Dependencies
```

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python src/train.py
```

3. Generate images:
```bash
python src/generate.py
```

## 📊 Output

Generated digit images will be saved in `images/` folder.

## 📚 Tech Used

- Python
- TensorFlow 2.x
- MNIST Dataset
- Matplotlib

## 📑 License

MIT License
