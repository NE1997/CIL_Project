import csv

predictions = []

with open("log/prediction_sat.txt", "r") as pre:
    pre_lines = pre.readlines()

for line in pre_lines:
    preds = line.strip("\n").split()
    if float(preds[0])>float(preds[1]):
        predictions.append("-1")
    else:
        predictions.append("1")

predictions = predictions[:10000]

header = ['Id','Prediction']

with open("log/submission_sat.csv", "w", encoding='UTF8') as of:
    writer = csv.writer(of)

    # write the header
    writer.writerow(header)

    # write the data
    for i in range(len(predictions)):
      data = [i+1, predictions[i]]
      writer.writerow(data)
