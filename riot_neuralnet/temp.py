#502
# arousal(least, most), valence(negative, positive), power, anticipation
prelabels = {1: [10, [.6, .4, .7, .6], [.9, .1, .8, .9]],
             2: [9, [.2, .5, .6, .1], [.3, .4, .5, .2]],
             3: [10, [.8, .9, .2, .9], [.99, .99, .1, .99]],
             4: [10, [.2, .4, .4, .5], [.8, .2, .7, .6]],
             5: [8, [.2, .4, .2, .1], [.5, .5, .5, .5]],
             6: [10, [.8, .2, .2, .5], [.9, .1, .1, .5]],
             7: [10, [.7, .4, .5, .6], [.8, .2, .8, .7]],
             8: [9, [.5, .5, .4, .5], [.6, .4, .5, .3]],
             9: [10, [.6, .4, .4, .7], [.9, .1, .1, .9]],
             10: [10, [.1, .5, .2, .1], [.7, .2, .2, .5]]
             }


labels = dict()

for i, key in enumerate(prelabels):
    row = prelabels[key]
    numImages = row[0]

    increment = list()
    for j in range(4):
        increment.append((row[2][j] - row[1][j]) / numImages)

    rowLabels = list()
    for k in range(numImages):
        newLabel = list()
        for labelIdx in range(4):
            newLabel.append(increment[labelIdx]*k+(row[1][labelIdx]))
        rowLabels.append(newLabel)
    labels[i] = rowLabels

return labels