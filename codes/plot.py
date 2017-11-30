import matplotlib.pyplot as plt
import csv

with open('output.csv') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    x_data = []
    y_data = []
    for row in reader:
        if i == 0:
            x_label, y_label = row[0], row[1]
        elif len(row) == 2:
            x_data.append(float(row[0]))
            y_data.append(float(row[1]))
        i += 1
plt.figure()
plt.plot(x_data, y_data)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title('loss~iter')
plt.savefig('plot.png')
