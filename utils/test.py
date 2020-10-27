import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
df1 = pd.read_html("/media/ulgen/Samsung/contradiction_data_depo/results/results/SNLI/christ_over_atheism.html")
df2 = pd.read_html(
    "/media/ulgen/Samsung/contradiction_data_depo/results/results/Mnli_Snli/christ_over_atheism.html")
df3 = pd.read_html(
    "/media/ulgen/Samsung/contradiction_data_depo/results/results/snli_mnli_anli/christ_over_atheism.html")

premises = []
hypothesis = []

snli_multiclass_prediction = []
snli_multiclass_prediction_entailment = []
snli_multiclass_prediction_contradiction = []
snli_multiclass_prediction_neutral = []

snli_mnli_multiclass_prediction = []
snli_mnli_multiclass_prediction_entailment = []
snli_mnli_multiclass_prediction_contradiction = []
snli_mnli_multiclass_prediction_neutral = []

snli_mnli_anli_multiclass_prediction = []
snli_mnli_anli_multiclass_prediction_entailment = []
snli_mnli_anli_multiclass_prediction_contradiction = []
snli_mnli_anli_multiclass_prediction_neutral = []

for i in range(len(df1[0])):
    if (str(df1[0]['three_class_type'][i]) != str(df2[0]['three_class_type'][i]) or
            str(df1[0]['three_class_type'][i]) != str(df3[0]['three_class_type'][i]) or
            str(df2[0]['three_class_type'][i]) != str(df3[0]['three_class_type'][i])):
        premises.append(str(df1[0]['premises'][i]))
        hypothesis.append(str(df1[0]['hypothesis:'][i]))

        snli_multiclass_prediction.append(df1[0]['three_class_type'][i])
        snli_multiclass_prediction_entailment.append(df1[0]['entailment score'][i])
        snli_multiclass_prediction_contradiction.append(df1[0]['contradiction score'][i])
        snli_multiclass_prediction_neutral.append(df1[0]['neutral score'][i])

        snli_mnli_multiclass_prediction.append(df2[0]['three_class_type'][i])
        snli_mnli_multiclass_prediction_entailment.append(df2[0]['entailment score'][i])
        snli_mnli_multiclass_prediction_contradiction.append(df2[0]['contradiction score'][i])
        snli_mnli_multiclass_prediction_neutral.append(df2[0]['neutral score'][i])

        snli_mnli_anli_multiclass_prediction.append(df3[0]['three_class_type'][i])
        snli_mnli_anli_multiclass_prediction_entailment.append(df3[0]['entailment score'][i])
        snli_mnli_anli_multiclass_prediction_contradiction.append(df3[0]['contradiction score'][i])
        snli_mnli_anli_multiclass_prediction_neutral.append(df3[0]['neutral score'][i])

df1 = pd.DataFrame(
    data={'premises': premises,
          'hypothesis': hypothesis,

          'snli multiclass prediction': snli_multiclass_prediction,
          'snli multiclass prediction entailment score': snli_multiclass_prediction_entailment,
          'snli multiclass prediction contradiction score': snli_multiclass_prediction_contradiction,
          'snli multiclass prediction neutral score': snli_multiclass_prediction_neutral,

          'snli_mnli multiclass prediction': snli_mnli_multiclass_prediction,
          'snli_mnli multiclass prediction entailment score': snli_mnli_multiclass_prediction_entailment,
          'snli_mnli multiclass prediction contradiction score': snli_mnli_multiclass_prediction_contradiction,
          'snli_mnli multiclass prediction neutral score': snli_mnli_multiclass_prediction_neutral,

          'snli_mnli_anli multiclass prediction': snli_mnli_anli_multiclass_prediction,
          'snli_mnli_anli multiclass prediction entailment score': snli_mnli_anli_multiclass_prediction_entailment,
          'snli_mnli_anli multiclass prediction contradiction score': snli_mnli_anli_multiclass_prediction_contradiction,
          'snli_mnli_anli multiclass prediction neutral score': snli_mnli_anli_multiclass_prediction_neutral})

print((df1.var()))
variance_list = list(df1.var(ddof=0))


# def plotting(entailment_type, snli, s_mnli, s_m_anli):
#     fig1, ax1 = plt.subplots()
#     ax1.set_title(entailment_type + 'variance analysis')
#     plt.boxplot(x=[snli, s_mnli, s_m_anli], labels=['SNLI Model', 'SNLI MNLI Model', 'SNLI MNLI ANLI Model'],
#                 manage_ticks=True, autorange=True, meanline=True)
#     # plt.savefig(path + data_type + 'Similarity.png', bbox_inches='tight')
#     plt.show()


# plotting(entailment_type='Entailment',
#          snli=snli_multiclass_prediction_entailment,
#          s_mnli=snli_mnli_multiclass_prediction_entailment,
#          s_m_anli=snli_mnli_anli_multiclass_prediction_entailment)
#plotting(entailment_type='Contradiction', snli=variance_list[1], s_mnli=variance_list[4], s_m_anli=variance_list[7])
#plotting(entailment_type='Entailment', snli=variance_list[2], s_mnli=variance_list[5], s_m_anli=variance_list[8])

# html = df1.to_html()
#
# # write html to file
# text_file = open("/home/ulgen/Downloads/aa/index.html", "w")
# text_file.write(html)
# text_file.close()
