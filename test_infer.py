import scirknn

clf = scirknn.MLPClassifier("test.rknn")
pred = clf.predict([0, 0])
print(pred)
