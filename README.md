# Language_Translation_Transformer_in_Tensorflow
Transformer architecture made in tensorflow to be applied towards an NLP problem: language translation. Made following a guide online (https://www.tensorflow.org/text/tutorials/transformer) while adding additional documentation throughout the notebook for beginners to understand these advanced concepts, as well as adding docstrings.

Developed for personal learning as well as outlining these concepts (NLP, Self-Attention, Transformers) to other students in Milwaukee School of Engineering's "Intro to Artificial Intelligence" course.

# Get Started Guide (PT to EN)
- Download Repo
- See Transformer breakdown in Jupyter-Notebook file (`.ipynb`) of "Portuguese_to_English" file
- Feel free to adjust the notebook to match your implementation/context
- Convert the Notebook (`.ipynb`) to Python (`.py`) running the command: `bash topy.sh <ipynb_filename>.ipynb`. `topy.sh` is provided.
- Run `.sh` file provided, changing file name of `.py` file to run (matching result of `bash topy.sh <ipynb_filename>.ipynb`) and any hyperparameters processed by `argparse` within the Python file

# Get Started Guide (EN to FR)
- Uses HuggingFace transformer (pre-trained T5), requiring a HuggingFace account.
- Please follow [this online guide](https://huggingface.co/docs/transformers/tasks/translation#inference) to configure your account and get started: 
