# File Description
## Preprocessing
- `data-preprocessing.ipynb` : For loading, cleaning, and tokenising the data
- `en_hi.model` and `en_hi.vocab` : Tokenisation trained model
## Training and base code
- `main-validation.py` : Training code
-  `transformer.py` : Transformer class which merge the encoder and decoder
- `encoder/` : Folder containing encoder code
- `decoder/` : Folder containing decoder code
- `utils/` : Folder containing common functions
## Trained model checkpoints
- `translation_model_{epoch}.pth` : Trained model after each epoch. Last (20th) epoch has best output.
## Inference code
- `predict.py` : Prediction code with manual input
- `length-predict-bleu-set1&2.py` : Bleu Score prediction by length
- `by-length-results.txt` : 

# Average Bleu scores by length with **Greedy Search** for 50 records each
```
Length:  9  Bleu Score:  0.47317584185316136
Length:  10 Bleu Score:  0.36808628946933775
Length:  11 Bleu Score:  0.4377701425288816
Length:  12 Bleu Score:  0.44453161305487837
Length:  13 Bleu Score:  0.4667191845585687
Length:  14 Bleu Score:  0.4462303301552929
Length:  15 Bleu Score:  0.4986597178135541
Length:  16 Bleu Score:  0.4850131733007086
Length:  17 Bleu Score:  0.42851941211043526
Length:  18 Bleu Score:  0.43671164431501247
Length:  19 Bleu Score:  0.43259285059202357
Length:  20 Bleu Score:  0.46458283766867714
Length:  21 Bleu Score:  0.475760403107931
Length:  22 Bleu Score:  0.4612374010401127
Length:  23 Bleu Score:  0.4440704429705374
Length:  24 Bleu Score:  0.4330057996301799
Length:  25 Bleu Score:  0.44726939490965734
Length:  26 Bleu Score:  0.45596596380959376
Length:  27 Bleu Score:  0.4681787157761271
Length:  28 Bleu Score:  0.4469493630088984
Length:  29 Bleu Score:  0.46226916712979943
Length:  30 Bleu Score:  0.4387080595023949
Length:  31 Bleu Score:  0.4366396830992947
```
# Average Bleu scores by length with **Beam Search** for 50 records each
```
Length:  9  Bleu Score:  0.4858710012609847
Length:  10 Bleu Score:  0.37565620566143204
Length:  11 Bleu Score:  0.44464086782898815
Length:  12 Bleu Score:  0.4242085432810601
Length:  13 Bleu Score:  0.474365618517872
Length:  14 Bleu Score:  0.4744230123650267
Length:  15 Bleu Score:  0.507801116338046
Length:  16 Bleu Score:  0.5040076892168642
Length:  17 Bleu Score:  0.42801351385501313
Length:  18 Bleu Score:  0.4177521440300274
Length:  19 Bleu Score:  0.41508380013237955
Length:  20 Bleu Score:  0.47638535680296035
Length:  21 Bleu Score:  0.48293294866774517
Length:  22 Bleu Score:  0.48522904429479163
Length:  23 Bleu Score:  0.43440588049445444
Length:  24 Bleu Score:  0.44700444041239995
Length:  25 Bleu Score:  0.44671512618976533
Length:  26 Bleu Score:  0.44172605889189137
Length:  27 Bleu Score:  0.47877799091588913
Length:  28 Bleu Score:  0.45783219654879626
Length:  29 Bleu Score:  0.4390365261513691
Length:  30 Bleu Score:  0.44840655983181693
Length:  31 Bleu Score:  0.4300475736888136
```

# Added Beam search for prediction
- greedy approch
```
Enter a sentence in English: i am happy
[906, 5984, 19, 79, 906, 5984, 19]
Translation: मुझे खुशी है कि मुझे खुशी है
```
- beam search
```
Enter a sentence in English: i am happy
Current beams: [([1, 906, 5984, 19, 13489, 2], -4.647991448640823), ([1, 906, 6962, 961, 19, 2], -4.798599556088448), ([1, 906, 6962, 961, 19, 13489, 2], -4.810402885079384), ([1, 906, 5984, 19, 79, 906, 5984, 19, 2], -7.418285638093948), ([1, 906, 
5984, 19, 79, 906, 5984, 19, 2358, 2], -7.928842989727855)]
Best score: -4.647991448640823
Translation: मुझे खुशी है ,
```
- **Although the greedy sequence `[906, 5984, 19, 79, 906, 5984, 19]` appears among the beam candidates (with a log‑probability of –7.42), it yields an incorrect translation. Beam search instead selects the higher‑scoring hypothesis (–4.65) and produces the correct result.**







