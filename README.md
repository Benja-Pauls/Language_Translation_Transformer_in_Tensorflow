# Language_Translation_Transformer_in_Tensorflow
**Ben Paulson & John Cisler**

Transformer architecture made in tensorflow to be applied towards an NLP problem: language translation. Made following a guide online (https://www.tensorflow.org/text/tutorials/transformer) while adding additional documentation throughout the notebook for beginners to understand these advanced concepts, as well as adding docstrings.

Developed for personal learning as well as outlining these concepts (NLP, Self-Attention, Transformers) to other students in Milwaukee School of Engineering's "Intro to Artificial Intelligence" course.


# Get Started Guide (PT to EN)
- Download Git Repository for zip file
- **Compile the Model Yourself Route (Jupyter):**
    - See Transformer breakdown in Jupyter-Notebook file (`.ipynb`) of "Portuguese_to_English" file
    - Feel free to adjust the notebook to match your implementation/context
    - Follow the in-notebook instructions on how to perform the implementation
- **Train the Model on Linux System (ROSIE):**
      - **Option 1:** Convert the Notebook (`.ipynb`) to Python (`.py`) running the command: `bash topy.sh <ipynb_filename>.ipynb`. `topy.sh` is provided. Run `tf_transformer_batched.sh` after changing the srun command within the `.sh` script to match the name of your newly-generated `.py` file, as well as any hyperparameters you would like to specify and have processed by `argparse` within the Python file.
      - **Option 2:** Run the existing `tf_transformer_batched.sh`, this will run the existing `First_Type_4L_20E_8H.py` which has 4 encoder/decoder layers and 8 heads for the multi-headed attention mechanisms. The model will train for 20 epochs and will output weights and a graphic each 10 save iterations.


# Get Started Guide (EN to FR)
- Uses HuggingFace transformer (pre-trained T5), requiring a HuggingFace account.
- Please follow [this online guide](https://huggingface.co/docs/transformers/tasks/translation#inference) to configure your account and get started: 

# Key Results
Both transformer models trained were able to produce comparable results to the outputs present from "Attention Is All You Need". As outlined in the associated presentations -- found in the `Presentation_Materials` directory -- the English to French model achieved a BLEU score of 36.0; slightly less than the amount achieved with a comparable model in "Attention Is All You Need" of 38.1. If you would like more info about how a BLEU score is conducted, as well as how it compares to using a simple masked-accuracy calculation (as was used with the Portuguese->English model), you can find that info [here](https://www.youtube.com/watch?v=-UqDljMymMg).<br/>
The other transformer model, namely Portuguese to English, used a masked attention callback to outline its masked accuracy at the end of each epoch during training. These values were viewed -- even for longer jobs which required a batch script to be used -- by viewing the associated slurm-job file. From this analysis, the Portuguese to English model achieved a masked accuracy of around 71%, meaning 71% of the time the model presents the perfect output sequence. Because "Attention Is All You Need" didn't deal with Portuguese, there is no comparison that can be made between this model and the paper. Therefore, a BLEU score was not calculated as this was a novel application that was done to ensure our understanding of the underlying concepts presented in the "Attention Is All You Need" paper. A visualization of the loss during this training process can be found at `Portuguese_to_English/100Epoch_Loss_Plot.png`.
