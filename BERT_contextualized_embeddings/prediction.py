import argparse

import plac
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from bert import tokenization
from utilities.utils import read_nli, attention_visualization, load_configurations, \
    predictions_to_html, set_memory_growth, read_test_json

set_memory_growth()
configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="evaluate",
                    help="This argument is to select whether to carry out 'evaluate' or 'demo' operation. Evaluate"
                         "operation takes labeled test data and measures the accuracy of the model. Demo operation"
                         "is for real-life usage. Demo compares two individual sentence or list of sentences"
                         "as input data.")

parser.add_argument("--nli_type", type=str, default="snli",
                    help="This parameter defines the train data which the model trained with. By specifying this"
                         "one can see the model behaviour on prediction time based on train data. There are 3 main "
                         "nli dataset 'snli', 'mnli', 'anli'. One can combine each of these according to their needs."
                         "If you combine train sets, dont use underline to define combination. Send parameter with one"
                         "blank space. It will shorten the html cell size. For example 'snli mnli' for combination of "
                         "snli and mnli train sets. This will be used for result columns and graphs.")

parser.add_argument("--bert_tf_hub", type=str, default="bert_tf_hub_contextualized",
                    help="When used with imported config parameter from yaml file it gives the path of the "
                         "BERT tensorflow-hub model. It is also the type of the transformer,"
                         "This will also be used to label resulting graphs and other files.")

parser.add_argument("--model_type", type=str, default="esim",
                    help="Type of the model architecture that the model is trained on. Contextualized word embeddings"
                         "are only used with ESIM model. This parameter takes only ESIM model architecture. This "
                         "parameter also carries the model save path information hence this is used for both defining"
                         "architecture and carrying path information.")

parser.add_argument("--visualization", type=bool, default=True,
                    help="shows attention heatmaps between two opinion sentences, best used with single"
                         "premise- hypothesis opinion sentence.")

parser.add_argument("--max_length", type=str, default=configs["max_length"],
                    help="Max length of the sentences, longer sentences will be pruned and shorter ones will be zero"
                         "padded. longer sentences mean longer sequences to train. Pick best length based on your rig.")

parser.add_argument("--test_loc", type=str, default=configs["nli_set_test"],
                    help="Test data location which will be used to measure the accuracy of the model")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path of the folder where results and graphs will be saved.")
args = parser.parse_args()

entailment_types = ["entailment", "contradiction", "neutral"]


def bert_encode(text, max_length, tokenizer, attention_heatmap):
    """
    Bert requires special pre-processing before feeding information to BERT model. Each text must be converted in to
    token_id, pad_mask and segment_ids. Bert takes three inputs as described and converts text in to contextualized
    word embeddings.
    :param text: opinion sentence.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param tokenizer: bert tokenizer object.
    :param attention_heatmap: boolean value to show attention heatmap of premise - hypothesis comparison.
    :return: token_ids, masks_ids, segments_ids.
    """

    text = tokenizer.tokenize(text)

    text = text[:max_length - 2]
    input_sequence = ["[CLS]"] + text + ["[SEP]"]
    pad_len = max_length - len(input_sequence)

    tokens = tokenizer.convert_tokens_to_ids(input_sequence)
    tokens += [0] * pad_len
    pad_masks = [1] * len(input_sequence) + [0] * pad_len
    segment_ids = [0] * max_length

    if attention_heatmap:
        return tf.convert_to_tensor(tokens), tf.convert_to_tensor(pad_masks), \
               tf.convert_to_tensor(segment_ids), input_sequence
    else:
        return tf.convert_to_tensor(tokens), tf.convert_to_tensor(pad_masks), tf.convert_to_tensor(segment_ids)


def evaluate(test_loc, max_length, bert_tf_hub, model_type):
    """
    Evaluates the trained NLI model with labeled NLI test data and prints accuracy metric.
    :param test_loc: labeled evaluation test data location.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param bert_tf_hub: the path definition of the BERT tensorflow-hub model.
    :param model_type: trained model architecture type. carries both model path and type definition information.
    :return: None
    """

    print("Loading trained NLI model")
    model = load_model(configs[model_type] + "model", custom_objects={"tf": tf, "KerasLayer": hub.KerasLayer})
    print("Trained NLI model loaded")

    bert_encoder = hub.KerasLayer(handle=configs[bert_tf_hub], trainable=False)
    vocab_file = bert_encoder.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_encoder.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    model.summary()
    # burada model outpu degistirildi attentin degerlerine ihtiyac yok bunu test et
    model = Model(inputs=model.input, outputs=model.output)
    model.summary()

    premise, hypothesis, dev_labels = read_nli(path=test_loc)

    total = 0.0
    true_p = 0.0

    for text1, text2, label in zip(premise, hypothesis, dev_labels):
        tokens1, masks1, segments1 = bert_encode(text=text1, max_length=max_length,
                                                 tokenizer=tokenizer, attention_heatmap=False)

        tokens2, masks2, segments2 = bert_encode(text=text2, max_length=max_length,
                                                 tokenizer=tokenizer, attention_heatmap=False)

        outputs = model.predict([[tokens1], [masks1], [segments1],
                                 [tokens2], [masks2], [segments2]])
        # scores = outputs[0]
        if entailment_types[outputs[0].argmax()] == entailment_types[label.argmax()]:
            true_p += 1
        total += 1
    print("NLI Model Accuracy is:", true_p / total)


def demo(premise, hypothesis, nli_type, bert_tf_hub, model_type, max_length, attention_map, result_path):
    """
    Performs demo operation using trained NLI model. Either takes two strings or list of strings. Compares the
    premise - hypothesis pairs and returns the NLI result.
    :param premise: opinion sentence.
    :param hypothesis: opinion sentence.
    :param nli_type: type of the nli set which is the model trained with.
    :param bert_tf_hub: BERT contextualized transformer Tensorflow Hub module path and definition information.
    :param model_type: type of the model. In this case arguments takes only ESIM.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param attention_map: boolean value to show attention heatmap of premise - hypothesis  pair.
    :param result_path: path of the file where the results will be saved.
    :return: None.
    """

    print("Loading NLI model")
    model = load_model(configs[model_type] + "model", custom_objects={"tf": tf, "KerasLayer": hub.KerasLayer})
    print("NLI model loaded")

    bert_encoder = hub.KerasLayer(handle=configs[bert_tf_hub], trainable=False)
    vocab_file = bert_encoder.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_encoder.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    model.summary()
    model = Model(inputs=model.input,
                  outputs=[model.output, model.get_layer('sum_x1').output, model.get_layer('sum_x2').output])

    if type(premise) and type(hypothesis) is str:

        print("Premise:", premise)
        print("Hypothesis:", hypothesis)

        tokens1, masks1, segments1, words1 = bert_encode(text=premise, max_length=max_length,
                                                         tokenizer=tokenizer,
                                                         attention_heatmap=configs['visualization'])

        tokens2, masks2, segments2, words2 = bert_encode(text=hypothesis, max_length=max_length,
                                                         tokenizer=tokenizer,
                                                         attention_heatmap=configs['visualization'])

        outputs = model.predict([[tokens1], [masks1], [segments1],
                                 [tokens2], [masks2], [segments2]])
        scores = outputs[0]

        print("Entailment type is:", entailment_types[scores.argmax()],
              "\nEntailment confidence is: ", scores.max(),
              "\nContradiction score is", float("{:.3f}".format(float(outputs[0][0][1]))),
              "\nEntailment score is", float("{:.3f}".format(float(outputs[0][0][0]))),
              "\nNeutral score is,", float("{:.3f}".format(float(outputs[0][0][2]))))

        if attention_map:
            attention_visualization(tokens1=words1, tokens2=words2,
                                    attention1=outputs[1], attention2=outputs[2],
                                    results_path=result_path, transformer_type=bert_tf_hub)

    elif type(premise) and type(hypothesis) is list:
        # takes the shortest list length as base in case lists are not contain the same amount of examples.
        a = min(len(premise), len(hypothesis))
        premises = premise[:a]
        hypothesises = hypothesis[:a]

        prediction_type = []
        contradiction_score = []
        neutral_score = []
        entailment_score = []

        total = 0.0
        contradiction = 0.0
        entailment = 0.0
        neutral = 0.0

        for text1, text2 in zip(premises, hypothesises):
            tokens1, masks1, segments1 = bert_encode(text=text1, max_length=max_length,
                                                     tokenizer=tokenizer, attention_heatmap=False)

            tokens2, masks2, segments2 = bert_encode(text=text2, max_length=max_length,
                                                     tokenizer=tokenizer, attention_heatmap=False)

            outputs = model.predict([[tokens1], [masks1], [segments1],
                                     [tokens2], [masks2], [segments2]])
            prediction = entailment_types[outputs[0].argmax()]

            prediction_type.append(prediction)
            contradiction_score.append(float("{:.3f}".format(float(outputs[0][0][1]))) * 100)
            neutral_score.append(float("{:.3f}".format(float(outputs[0][0][2]))) * 100)
            entailment_score.append(float("{:.3f}".format(float(outputs[0][0][0]))) * 100)

            if prediction is 'contradiction':
                contradiction += 1
            elif prediction is 'entailment':
                entailment += 1
            elif prediction is 'neutral':
                neutral += 1
            total += 1

        print("Total Contradiction = ", float("{:.3f}".format(float(contradiction / total))) * 100)
        print("Total Entailment =", float("{:.3f}".format(float(entailment / total))) * 100)
        print("Total Neutral =", float("{:.3f}".format(float(neutral / total))) * 100)

        # last line of the result output html show the total amount of data and predictions.
        prediction_type.append(total)
        contradiction_score.append(contradiction)
        neutral_score.append(neutral)
        entailment_score.append(entailment)

        predictions_to_html(nli_type=nli_type,
                            premises=premises,
                            hypothesises=hypothesises,
                            prediction=prediction_type,
                            contradiction_score=contradiction_score,
                            neutral_score=neutral_score,
                            entailment_score=entailment_score,
                            result_path=result_path
                            )


def main():
    if args.mode == "evaluate":
        print("test mode is", args.mode)
        evaluate(test_loc=args.test_loc,
                 max_length=args.max_length,
                 bert_tf_hub=args.bert_tf_hub,
                 model_type=args.model_type)

    elif args.mode == "demo":
        print("test mode is", args.mode)

        premise, hypothesis = read_test_json(
            path="/media/ulgen/Samsung/contradiction_data/results/evaluation_entailment.jsonl")

        # premise = "in the park alice plays a flute solo"
        # hypothesis = "someone playing music outside"

        demo(premise=premise,
             hypothesis=hypothesis,
             nli_type=args.nli_type,
             bert_tf_hub=args.bert_tf_hub,
             model_type=args.model_type,
             max_length=args.max_length,
             attention_map=args.visualization,
             result_path=args.result_path)


if __name__ == "__main__":
    plac.call(main)
