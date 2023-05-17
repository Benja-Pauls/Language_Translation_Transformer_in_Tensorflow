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

# Key Results
Both transformer models trained were able to produce comparable results to the outputs present from "Attention Is All You Need". As outlined in the associated presentations -- found in the `Presentation_Materials` directory -- the English to French model achieved a BLEU score of 36.0; slightly less than the amount achieved with a comparable model in "Attention Is All You Need" of 38.1. If you would like more info about how a BLEU score is conducted, as well as how it compares to using a simple masked-accuracy calculation (as was used with the Portuguese->English model), you can find that info [here](https://www.youtube.com/watch?v=-UqDljMymMg).<br/>
The other transformer model, namely Portuguese to English, used a masked attention callback to outline its masked accuracy at the end of each epoch during training. These values were viewed -- even for longer jobs which required a batch script to be used -- by viewing the associated slurm-job file. From this analysis, the Portuguese to English model achieved a masked accuracy of around 71%, meaning 71% of the time the model presents the perfect output sequence. Because "Attention Is All You Need" didn't deal with Portuguese, there is no comparison that can be made between this model and the paper. Therefore, a BLEU score was not calculated as this was a novel application that was done to ensure our understanding of the underlying concepts presented in the "Attention Is All You Need" paper. A visualization of the loss during this training process can be found at `Portuguese_to_English/100Epoch_Loss_Plot.png`.
