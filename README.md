# CT-ML
As text classification becomes increasingly granular, it is necessary to identify the relationship between content and multiple labels. Multi-label text classification (MLTC) addresses the limitations of traditional text classification tasks, including coarse classification granularity, single classification angle, and low classification accuracy. In real-world scenarios, the quality of data labeling can be a major concern due to various factors that can introduce noise to the labeled data. This noise can significantly affect the performance of classifiers, making it difficult to generalize accurately. To overcome this limitation, we use loss correction methods to prevent the model from memorizing noisy labels. Meanwhile, we integrate the loss correction into the co-teaching learning model in order to avoid the accumulation of errors in the iterations of the loss correction. It trains two networks concurrently to learn clean labels by cross-referencing each other's predictions. Furthermore, we used clustering methods to classify probability distributions flexibly and improve the accuracy of model prediction labels. Our model is designed to adapt to the training set bias and label noise, which helps maintain stability in the model's accuracy even when learning with noise. Extensive results on three publicly available datasets of multi-labeled text validate the effectiveness of our model and demonstrate its superiority over the state-of-the-art methods.
## Dependencies

This project requires the following environments and libraries:

- Python 3.7
- PyTorch 1.13.1
- Transformers 4.49.2

Ensure you have the correct Python version installed on your system. You can install the required libraries using `pip`:

```bash
pip install torch==1.13.1 transformers==4.49.2
```

# Installation

```bash
git clone https://github.com/yuccccci/CT-ML.git
cd CT-ML
```
# Usage
```bash
python main.py
```
- main.py: The entry point of the code. It orchestrates the workflow of the application and is responsible for executing the program.
- loss_lm.py: The core part of the code, handling loss adjustment. It implements LC (Loss Correction) and LR (Loss Regularization), critical for optimizing the model's performance.

