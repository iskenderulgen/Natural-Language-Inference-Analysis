# Experimental Results using Decomposable Attention Model

### Results using Spacy Glove based En_Core_Web_Lg weights

* Train Results = loss: 0.4453 - acc: 0.8279 - val_loss: 0.3989 - val_acc: 0.8461
* Evalutaion results = 8269.0 / 9824.0  -  Total Acc = 0.8417141693811075

### Real Life Demo Result Using En_Core_Web_Lg

* Sentence 1: option one is much more better
* Sentence 2: option one is worst
* Entailment type: contradiction (Confidence: 0.5369793 )

Notes: Model is based on similarity and implemented to spacy as pipeline. Model trained on Stanford SNLI corpus using attention
with feedforward network. Model can only distinguish basic contradiction it lacks of syntactic information. 
Using glove as vectorizer gives good results but padding it to fixed size of vectors takes additional process power instead doc
vectorizer could be more useful. 

### Experimental Results using BERT(BASE) sentence based approach 

* Train Results = loss: 0.6105 - acc: 0.7454 - val_loss: 0.6051 - val_acc: 0.7484

| lr            | hidden        | batch | Acc  |
| ------------- |:-------------:| -----:| ----:|
| 0.001         | 200           | 1024  | 33   |
| 0.00001       | 200           | 1024  | 70   |
| 0.00001       | 200           | 512   | 72   |
| 0.00001       | 200           | 64    | 72   |
| 0.00001       | 400           | 128   | 74   |

Notes: For this approach only last layer of contextualized sentence embeddings are used. It's expected to see
better result when using sum of last 4 layer approach is used with bert larger. Current results consist of bert
base vectors. 

### Experimental Results using BERT (BASE) word based approach with bert base
 
* Train Results = loss: 0.4935 - acc: 0.8064 - val_loss: 0.4377 - val_acc: 0.8316

Notes: For this approach, BERT's initial word embeddings are used instead of contextualised word embeddings.
Reason for this approach, bert contextualised requires 366 GB np array to store embeddings. Bert initial works
as intended and requires much less computational power. 
Bert base creates 768 dimensional word embedding.

### Experimental Results using BERT (LARGE) word based approach with bert Large

* Train Results = loss: 0.3795 - acc: 0.8564 - val_loss: 0.3802 - val_acc: 0.8596

| lr            | hidden        | batch | Acc  |
| ------------- |:-------------:| -----:| ----:|
| 0.00001       | 200           | 128   | 85   |
| 0.00001       | 512           | 128   | 85.96|

Note: Bert Large creates 1024 dimensional word embedding which takes the benefit of fine representation of large 
dimension.

# Experimental Results Using Enhanced Sequential Inference Model (ESIM)

Context:  ESIM is based on two layers of BiLSTM network which uses attention. 

#### ESIM + Spacy Web_Core embeddings
* Train Results = loss: 0.1817 - acc: 0.9338 - val_loss: 0.4930 - val_acc: 0.8490

| learning_rate| hidden_size   | batch | Train_Acc  | Val_Acc|
| -------------|:-------------:| -----:| ----------:| ------:|
| 0.0001       | 200           | 200   | 93.38      | 84.90  |

* Train time = 1 hrs 55 mins

#### ESIM + Bert Initial Embeddings Results

| learning_rate| hidden_size   | batch | Train_Acc  | Val_Acc|
| -------------|:-------------:| -----:| ----------:| ------:|
| 0.0001       | 200           | 100   | 88.81      | 86.30  |
| 0.0001       | 400           | 100   | 90.17      | 86.55  |

* Train time for 400 hidden = 4.5 hrs
* Train Time for 200 hidden = 70 mins

* TO - DO , make batch size 50 make adam reg param = 0.0005 (Estimated train time 10 hrs)

To pick ideal hidden size =  NS / (a * (Ni + No))

 - Ns = size of the train input
 - Ni = size of the input neuron size
 - No = size of the output neuron size
 - a = scaling factor

a works best with 2