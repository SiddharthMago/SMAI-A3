# SMAI Assignment 3: Build a Real ML App

## T3.1 - Handwritten Devanagari Digit Recognition

## Assignment Variant

- **Theme:** T3 - Handwritten Indic Script Recognition
- **Variant:** T3.1 - Devanagari digits
- **Tier:** 1
- **Dataset:** UCI Devanagari Handwritten Character Dataset, id 389
- **Task:** 10-class image classification over `digit_0` to `digit_9`
- **Required app:** Streamlit app with drawable canvas and prediction
- **Target model:** Small CNN trained from scratch, around 300k parameters, 97%+ accuracy

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── model.ipynb
├── model_outputs.zip
├── streamlit_app/
│   ├── app.py
│   ├── model.py
│   ├── .streamlit/
│   │   └── config.toml
│   └── models/
│       ├── devanagari_digit_cnn.pt
│       └── class_metadata.json
└── SMAI_Assignment_3_topics.pdf
```

## Setup

Create and activate a virtual environment from the repository root:

```bash
cd path/to/SMAI-A3

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Streamlit App

From the repository root:

```bash
source .venv/bin/activate
streamlit run streamlit_app/app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

## Run or Reproduce Model Training

The training code is in `model.ipynb`.

After setting up the environment, register the venv as a Jupyter kernel:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name smai-a3 --display-name "Python (SMAI A3)"
```

Then open `model.ipynb` in Jupyter, VS Code, or Kaggle/Colab and run all cells.

To launch the notebook locally from this environment:

```bash
python -m pip install notebook
jupyter notebook model.ipynb
```

If running locally, the notebook will create:

```text
data/
models/devanagari_digit_cnn.pt
models/class_metadata.json
models/training_history.json
model_outputs.zip
```

To use a newly trained model in the Streamlit app:

```bash
mkdir -p streamlit_app/models
cp models/devanagari_digit_cnn.pt streamlit_app/models/
cp models/class_metadata.json streamlit_app/models/
```

If training on Kaggle or Colab, download the generated output zip and place/extract the model and metadata into `streamlit_app/models/`.

## Code Overview

### Training Notebook: `model.ipynb`

The notebook contains the full model-building workflow:

1. Downloads and extracts the UCI Devanagari Handwritten Character Dataset.
2. Filters the full 46-class dataset to the 10 digit classes.
3. Builds train, validation, and test loaders from the official dataset split.
4. Defines and trains a small PyTorch CNN from scratch.
5. Evaluates the model with accuracy, classification report, confusion matrix, and misclassified examples.
6. Saves the best model checkpoint, class metadata, and a zipped output bundle.

The digit subset contains:

- `17,000` training images
- `3,000` official test images
- `20,000` total digit images

### Streamlit App: `streamlit_app/app.py`

The app provides the user-facing demo required by the assignment:

- **Draw mode:** write a digit on a canvas using `streamlit-drawable-canvas`.
- **Upload mode:** upload a `png`, `jpg`, or `jpeg` digit image.
- **Prediction panel:** shows the predicted Devanagari digit, Unicode code, class index, and confidence.
- **Probability chart:** displays the class probability distribution for all 10 digits.
- **Practice mode:** lets the user choose a target digit and gives feedback after prediction.

### Model Definition: `streamlit_app/model.py`

The Streamlit app keeps a copy of the CNN architecture so that the saved PyTorch checkpoint can be loaded without depending on the notebook.

The model uses:

- 3 convolutional layers
- Batch normalization
- ReLU activations
- Max pooling
- Dropout regularization
- A compact fully connected classifier

The trained model has about **290,002 trainable parameters**, matching the small-CNN scale suggested in the assignment.

### Saved Artifacts

The app uses the following files:

- `streamlit_app/models/devanagari_digit_cnn.pt`: trained PyTorch checkpoint
- `streamlit_app/models/class_metadata.json`: class names, Devanagari digits, Unicode codes, image size, model details, and evaluation metrics

The saved metadata records a test accuracy of **99.6%** on the official digit test split.

## Design Choices

- **CNN from scratch:** The assignment recommends a lightweight CNN instead of transfer learning for this task. Devanagari digits are small grayscale symbols, so a compact CNN is sufficient.
- **32x32 grayscale inputs:** This matches the original dataset format and keeps training fast.
- **Validation split from training data:** The official test split is kept untouched until final evaluation.
- **Best-checkpoint saving:** The notebook saves the model with the best validation accuracy instead of simply using the final epoch.
- **Separate training and app code:** The notebook is for experimentation and evaluation, while `streamlit_app/` contains only the files needed for the demo.
- **Canvas preprocessing:** Drawn and uploaded images are converted to grayscale, cropped to foreground content, padded to a square, resized to 32x32, and normalized before inference.
- **Unicode-aware output:** The app reports both the visual Devanagari digit and its Unicode code point.

## Expected Results

The included trained model reports:

- **Best validation accuracy:** 99.7059%
- **Test accuracy:** 99.6000%
- **Incorrect test predictions:** 12 out of 3000

These results exceed the assignment's suggested 97%+ benchmark for T3.

## Deployment Notes


## References

- UCI Machine Learning Repository: Devanagari Handwritten Character Dataset, id 389
- SMAI Assignment 3 topic catalogue, T3 - Handwritten Indic Script Recognition
- PyTorch documentation
- Streamlit documentation
