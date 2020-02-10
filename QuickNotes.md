# Experimental Results using spacy GLOVE

## Train Result

* 549367/549367 [==============================]
* 34s 62us/step - loss: 0.4453 - acc: 0.8279 - val_loss: 0.3989 - val_acc: 0.8461
* Saving to /home/ulgen/anaconda3/envs/python36/lib/python3.6/site-packages/en_core_web_lg/en_core_web_lg-2.2.5/similarity

## Eval Result

* 8269.0 / 9824.0 
* Total Acc = 0.8417141693811075

## Real Life Demo Result

* Sentence 1: option one is much more better
* Sentence 2: option one is worst
* Entailment type: contradiction (Confidence: 0.5369793 )

Notes: Model is based on similarity and implemented to spacy as pipeline. Model trained on Stanford SNLI corpus using attention
with feedforward network. Model can only distinguish basic contradiction it lacks of syntactic information. 
Using glove as vectorizer gives good results but padding it to fixed size of vectors takes additional process power instead doc
vectorizer could be more useful. 

# Experimental Results using BERT sentence based approach 

## Train Result

* 549367/549367 [==============================] 
* 38s 70us/step - loss: 0.6105 - acc: 0.7454 - val_loss: 0.6051 - val_acc: 0.7484
* Saving to /home/ulgen/anaconda3/envs/python36/lib/python3.6/site-packages/en_core_web_lg/en_core_web_lg-2.2.5/similarity

| lr            | hidden        | batch | Acc  |
| ------------- |:-------------:| -----:| ----:|
| 0.001         | 200           | 1024  | 33   |
| 0.00001       | 200           | 1024  | 70   |
| 0.00001       | 200           | 512   | 72   |
| 0.00001       | 200           | 64    | 72   |
| 0.00001       | 400           | 128   | 74   |

# Experimental Results using BERT word based approach

## Train Results

* 549367/549367 [==============================] 
* 107s 195us/step - loss: 0.4935 - acc: 0.8064 - val_loss: 0.4377 - val_acc: 0.8316
* Saving to /home/ulgen/anaconda3/envs/python36/lib/python3.6/site-packages/en_core_web_lg/en_core_web_lg-2.2.5/similarity
