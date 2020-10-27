import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from bioinfokit.analys import stat
import scikit_posthocs as sp
#df1 = pd.read_html("/home/ulgen/Downloads/aa/Differences.html")

# entailment_type = []
# snli_model = []
# snli_mnli_model = []
# snli_mnli_anli_model = []

#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.width', 2000)
#
# for i in range(len(df1[0])):
#     snli_model.append(df1[0]['snli multiclass prediction neutral score'][i])
#     snli_mnli_model.append(df1[0]['snli_mnli multiclass prediction neutral score'][i])
#     snli_mnli_anli_model.append(df1[0]['snli_mnli_anli multiclass prediction neutral score'][i])
#
# df1 = pd.DataFrame(data={'snli_model': snli_model,
#                          'snli_mnli_model': snli_mnli_model,
#                          'snli_mnli_anli_model': snli_mnli_anli_model})
#
#
# def variance(data):
#     n = len(data)
#     mean = sum(data) / n
#     deviations = [(x - mean) ** 2 for x in data]
#     variance = sum(deviations) / n
#     print(variance)
#
#
# print(df1.var())
#
# fvalue, pvalue = stats.f_oneway(np.asarray(df1['snli_model']),
#                                 np.asarray(df1['snli_mnli_model']),
#                                 np.asarray(df1['snli_mnli_anli_model']))
# print("f_value = ", fvalue, "p_value =", pvalue)
#
# # reshape the d dataframe suitable for statsmodels package
# d_melt = pd.melt(df1.reset_index(), id_vars=['index'], value_vars=['snli_model',
#                                                                    'snli_mnli_model',
#                                                                    'snli_mnli_anli_model'])
# # replace column names
# d_melt.columns = ['index', 'models', 'value']
# # Ordinary Least Squares (OLS) model
# model = ols('value ~ C(models)', data=d_melt).fit()
# anova_table = sm.stats.anova_lm(model, typ=1)
# print(anova_table)
#
# res = stat()
# res.tukey_hsd(df=d_melt, res_var='value', xfac_var='models', anova_xfac_var='models')
# print(res.tukey_summary)
#
# df2 = pd.DataFrame(data={
#                          'snli_mnli_model': snli_mnli_model,
#                          'snli_mnli_anli_model': snli_mnli_anli_model})
#
# print("\n### SCHEFFE TABLE")
# df2 = df2.melt(var_name='groups', value_name='values')
# print(sp.posthoc_scheffe(a=df2, val_col='values', group_col='groups'))


"""up side is the original one"""

df1 = pd.read_html("/home/ulgen/Downloads/aa/Differences.html")

snli_contradiction_score = []
snli_entailment_score = []
snli_neutral_score = []

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)

for i in range(len(df1[0])):
    snli_contradiction_score.append(df1[0]['snli multiclass prediction contradiction score'][i])
    snli_neutral_score.append(df1[0]['snli multiclass prediction neutral score'][i])
    snli_entailment_score.append((df1[0]['snli multiclass prediction entailment score'][i]))

df1 = pd.DataFrame(data={'snli_contra': snli_contradiction_score,
                         'snli_neutral': snli_neutral_score,
                         'snli_entailment': snli_entailment_score
                         })

print("#### variances")
print(df1.var())

fvalue, pvalue = stats.f_oneway(np.asarray(df1['snli_contra']),
                                np.asarray(df1['snli_neutral']),
                                np.asarray(df1['snli_entailment']))

print("\n f_value = ", fvalue, "p_value =", pvalue)

# reshape the d dataframe suitable for statsmodels package
d_melt = pd.melt(df1.reset_index(), id_vars=['index'], value_vars=['snli_contra',
                                                                   'snli_neutral',
                                                                   'snli_entailment'])
# replace column names
d_melt.columns = ['index', 'models', 'value']
# Ordinary Least Squares (OLS) model
model = ols('value ~ C(models)', data=d_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\n#### ANOVA TABLE")
print(anova_table)

res = stat()
res.tukey_hsd(df=d_melt, res_var='value', xfac_var='models', anova_xfac_var='models')
print("\n### TUKEY SHD TABLE")
print(res.tukey_summary)

df2 = pd.DataFrame(data={
                         'snli_neutral': snli_neutral_score,
                         'snli_entailment': snli_entailment_score
                         })
print("\n### SCHEFFE TABLE")
df2 = df2.melt(var_name='groups', value_name='values')
print(sp.posthoc_scheffe(a=df2, val_col='values', group_col='groups'))

