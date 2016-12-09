#!/usr/bin/python3

import struct
import sys

if len(sys.argv) != 4:
    print("Usage: python parse.py [SET_NAME] [IMAGE_FILE] [LABEL_FILE]")
    sys.exit(1)

testsetName = sys.argv[1]
imageFileName = sys.argv[2]
labelFileName = sys.argv[3]

imageFile = open(imageFileName, "rb")
labelFile = open(labelFileName, "rb")
inputFile = open("%s-input.dsstne" % testsetName, "w")
outputFile = open("%s-output.dsstne" % testsetName, "w")


# read image file's header
imageMagicNumber = struct.unpack(">i", imageFile.read(4))[0]
if imageMagicNumber != 2051:
    print("this is not image file: %d" % imageMagicNumber)
    sys.exit(1)
numImages = struct.unpack(">i", imageFile.read(4))[0]
row = struct.unpack(">i", imageFile.read(4))[0]
col = struct.unpack(">i", imageFile.read(4))[0]
n = row * col

# read label file's header
labelMagicNumber = struct.unpack(">i", labelFile.read(4))[0]
if labelMagicNumber != 2049:
    print("this is not label file: %d" % labelMagicNumber)
    sys.exit(1)
numItems = struct.unpack(">i", labelFile.read(4))[0]
if numImages != numItems:
    print("number of Images is not same with number of Index:%d %d" % (numImages, numItems))
    sys.exit(1)

# make dsstne files
print("images: %d\nrow: %d\tcol:%d" % (numImages, row, col))

for i in range (1, numImages+1):
    labelNum = struct.unpack(">B", labelFile.read(1))[0]
    if i % 10000 == 0:
        print("processing image #%d ..." % i)
    inputFile.write("case{0}\t".format(i))
    outputFile.write("case{caseIndex}\t{label}\n".format(caseIndex = i, label = labelNum))

    cnt = 0
    for j in range(1, n + 1):
        pixelNum = struct.unpack(">B", imageFile.read(1))[0]
        if pixelNum < 128:
            continue
        if cnt > 0:
            inputFile.write(":")
        inputFile.write("{0}".format(j))
        cnt = cnt + 1
    inputFile.write("\n")

# close files
imageFile.close()
labelFile.close()
inputFile.close()
outputFile.close()

