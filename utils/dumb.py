import pandas as pd

f = open("/home/ulgen/Desktop/Rapor/differences_labeling.txt", "r")
text_labels = []
for line in f:
    text_labels.append(line.split("-")[1].strip())

df1 = pd.read_html("/home/ulgen/Desktop/Rapor/Differences.html")
html_array = []
for i in range(57):
    html_array.append(df1[0]['snli_mnli_anli multiclass prediction'][i].strip())

tp = 0
total = 0

for i in range(len(html_array)):
    print(html_array[i])
    if text_labels[i] == html_array[i]:
        tp += 1
    total += 1

print("accuracy :", tp / total)
