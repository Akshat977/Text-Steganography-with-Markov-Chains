# -*- coding: utf-8 -*-


from pywebio.input import *
from pywebio.output import *
from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server




import config
import utils
import bigBitField
import fixedSizeCode
import fixedSizeDecode
import json
import time

app = Flask(__name__)

"""
encodes variable sized data to text

- data: data to be encoded (a list of integers, every one is a byte)
- bitsForSize: how many bits will be used to encode the size
- markovChain: markov chainto use

returns the encoded text
"""
def encodeDataToWordList(data, bytesForSize, markovChain, wordsPerState = 1):

	# encode the data length first
	lenData = convertNumberToByteList(len(data), bytesForSize)
	bitsField = bigBitField.BigBitField(lenData)

	if wordsPerState == 1:
		lastWord = startSymbol
	elif wordsPerState == 2:
		lastWord = (startSymbol, startSymbol)

	lenDataCode = encodeBitsToWordList(bitsField, markovChain, lastWord, wordsPerState)

	# compute last word (or bigram)
	if wordsPerState == 1:
		lastWord = lenDataCode[-1]
	elif wordsPerState == 2:
		if len(lenDataCode) <= 1:
			lastWord = (startSymbol, lenDataCode[-1])
		else:
			lastWord = (lenDataCode[-2], lenDataCode[-1])

		if lastWord[1] == config.startSymbol:
			lastWord = (startSymbol, startSymbol)

	# encode the actual message
	bitsField = bigBitField.BigBitField(data)
	mainDataCode = encodeBitsToWordList(bitsField, markovChain, lastWord, wordsPerState)

	return lenDataCode + mainDataCode

"""
decodes a text to data (variable sized data)

- wordList: encoded data in a text
- bitsForSize: how many bits will be used to encode the size
- markovChain: markov chainto use

returns the decoded data (in a bigBitField object)
"""
def decodeWordListToData(wordList, bytesForSize, markovChain, wordsPerState = 1):

	if wordsPerState == 1:
		lastWord = startSymbol
	elif wordsPerState == 2:
		lastWord = (config.startSymbol, config.startSymbol)

	# decode the message length
	(lenRawData, wordsUsed) = fixedSizeDecode.decodeWordListToBits(wordList, bytesForSize * 8, markovChain, lastWord, wordsPerState)
	lenRawData = lenRawData.getAllBytes()
	lenData = utils.convertByteListToNumber(lenRawData)

	# compute last word (or bigram)
	if wordsPerState == 1:
		lastWord = wordList[wordsUsed - 1]
	elif wordsPerState == 2:
		if wordsUsed == 1:
			lastWord = (config.startSymbol, wordList[wordsUsed - 1])
		elif wordsUsed == 0:
			raise RuntimeError("only 0 words used in decode word list to data")
		else:
			lastWord = (wordList[wordsUsed - 2], wordList[wordsUsed - 1])

		if lastWord[1] == config.startSymbol:
			lastWord = (config.startSymbol, config.startSymbol)

	# decode the actual message
	wordList = wordList[wordsUsed:]
	(decodedData, wordsUsed) = fixedSizeDecode.decodeWordListToBits(wordList, lenData * 8, markovChain, lastWord, wordsPerState)

	return decodedData


# given 2 input files, encode and save to the output file
def encodeDataFromFile(inputFile, outputFile, markovInputFile, textFileFormat, wordsPerState,text):
	f=open(inputFile,'w')
	f.write(text)
	f.close()
	initTime = time.time()
	#print(inputFile)
	f = open(markovInputFile, 'r')
	jsonData = f.read()
	f.close()
	markovData = json.JSONDecoder().decode(jsonData)

	if (wordsPerState == 1 and type(markovData[0][0]) != str and type(markovData[0][0]) != unicode) or (wordsPerState == 2 and type(markovData[0][0]) != list):
		raise RuntimeError("error; markov chain structure doesn't match wordsPerState value")

	inputData = []
	f = open(inputFile, 'rb')
	char = None
	while char != b"":
		char = f.read(1)
		##print(char)
		if char != b"": 
			#print(char)
			##print(str(char))
			##print(ord(char))
			o_char=ord(char)
			#print(o_char)
			inputData.append(ord(char))
	f.close()

	##print("This worked")

	initTimeCode = time.time()

	encodedData = encodeDataToWordList(inputData, 4, markovData, wordsPerState)

	# save
	if textFileFormat:
		outputData = wordListToText(encodedData)
	else:
		outputData = json.JSONEncoder().encode(encodedData)

	endTimeCode = time.time()

	f = open(outputFile, 'w')
	f.write(outputData)
    
	f.close()
	with open('encoded.txt', 'r') as file:
		s = file.read().replace('\n', '')
	popup('Encoded text', s)

	#print ("wrote " + repr(len(inputData) * 8) + " bits")
	#print ("elapsed time: " + repr(time.time() - initTime) + " seconds")
	#print (" - encoding time: " + repr(endTimeCode - initTimeCode) + " seconds")

# given 2 input files, decode and save to the output file
def decodeDataFromFile(inputFile, outputFile, markovInputFile, textFileFormat, wordsPerState,text):
	initTime = time.time()
	f=open(inputFile,'w')
	f.write(text)
	f.close()
	f = open(markovInputFile, 'r')
	jsonData = f.read()
	f.close()
	markovData = json.JSONDecoder().decode(jsonData)

	if (wordsPerState == 1 and type(markovData[0][0]) != str and type(markovData[0][0]) != unicode) or (wordsPerState == 2 and type(markovData[0][0]) != list):
		raise RuntimeError("error; markov chain structure doesn't match wordsPerState value")

	f = open(inputFile, 'r')
	inputData = f.read()
	f.close()

	initTimeDecode = time.time()
	if textFileFormat:
		inputData = textToWordList(inputData)
	else:
		inputData = json.JSONDecoder().decode(inputData)

	decodedData = decodeWordListToData(inputData, 4, markovData, wordsPerState)
	#print ("read " + repr(decodedData.totalFieldLen()) + " bits")
	decodedData = decodedData.getAllBytes()
	endTimeDecode = time.time()

	# save
	f = open(outputFile, 'w')
	#print(decodedData)
	#print("--------------------------------------")
	for b in decodedData:
		#print(b,type(b))
		f.write(chr(b))
	f.close()
	with open('output.txt', 'r') as file:
		s = file.read().replace('\n', '')
	popup('Decoded text', s)

	#print ("elapsed time: " + repr(time.time() - initTime) + " seconds")
	#print (" - decoding time: " + repr(endTimeDecode - initTimeDecode) + " seconds")


"functions in this file encode digits of fixed size into text; and decode them from text to digits"
import config
import utils
import bigBitField
import time

"""
encodes a single word

- bits: a bit field object with the data to encode
- bitsRange: the range to encode this from
- startWord: previous word encoded (either 1 or 2 words depending on wordsPerState)
- markovChainDict: dictionary with the chain to use to encode data

returns (word, newRange)
"""
def encodeBitsToWord(bitsField, bitsRange, startWord, markovChainDict):

	# get probabilities for the start word
	wordProbs = markovChainDict[lowerWordOrList(startWord)][1]

	# get the range covered by every word
	wordRanges = computeWordRanges(bitsRange, wordProbs, bitsField.totalFieldLen())

	# look for the right partition for the bits
	precision = len(wordRanges[0][1][0])
	bits = bitsField.getFirstNBits(precision)

	bestWord = filter(
		lambda wr:
			binaryLowerEqualThan(wr[1][0], bits) and binaryLowerEqualThan(bits, wr[1][1]),
		wordRanges)

	someword=list(bestWord)
	#print(someword[0])

	return someword[0]


"""
encodes all a bit field to a list of words, using a markov chain

- bitsField: a bit field object with the data to encode
- markovChain: chain to use to encode data
- startWord: if it is not the default, what was the previous word before this text

returns wordList
"""
def encodeBitsToWordList(bitsField, markovChain, startWord = config.startSymbol, wordsPerState = 1):

	bitsField = bitsField.copy()
	lastTime = time.time()
	secondsForStatusprint = 20

	words = []
	nextRange = ["0", "1"]
	markovChainDict = markovChainToDictionary(markovChain)

	while True:
		# encode to one word
		(word, nextRange) = encodeBitsToWord(bitsField, nextRange, startWord, markovChainDict)
		words.append(word)

		# compute previous word (or bigram) for next iteration
		if wordsPerState == 1:
			startWord = word
		elif wordsPerState == 2:
			if word == config.startSymbol:
				startWord = (config.startSymbol, config.startSymbol)
			else:
				startWord = (startWord[1], word)

		# optimization, remove start of range when it is identical in both sides
		nextRange2 = utils.removeCommonBitsInRange(nextRange)
		bitsField.popFirstNBits(len(nextRange[0])-len(nextRange2[0]))
		nextRange = nextRange2

		if time.time()-lastTime > secondsForStatusprint:
			print (" - remaining bits: " + repr(bitsField.totalFieldLen()))
			lastTime = time.time()

		# we exit when our range describes only to our number
		if bitsField.totalFieldLen() == 0 or (bitsField.totalFieldLen() == 1 and nextRange[0][0] == nextRange[1][0]):
			break

	return words


"defines a big bit field object"
import utils
import math

"""
a class for managing big amounts of bits (these bits are internally stored as bytes, except in the left and right extremes)
"""
class BigBitField:

	def __init__(self, data = None, dataIsBytes = True):
		if data is None:
			data = []

		self.lastBitsCache = ""

		if dataIsBytes:
			self.firstBitsCache = ""
			self.remainingBytes = data
		else:
			self.firstBitsCache = data
			self.remainingBytes = []


	def copy(self):
		bitField = BigBitField()
		bitField.firstBitsCache = self.firstBitsCache
		bitField.lastBitsCache = self.lastBitsCache
		bitField.remainingBytes = list(self.remainingBytes)
		return bitField

	"""
	internal; we store data as bytes and as a string with the explicit bits... this function extracts n bytes into the
	string, and converts them to ascii 1 and 0 that is easy to operate with
	"""
	def popBytesToBitsCache(self, bytesToPop):
		if len(self.remainingBytes) < bytesToPop:
			raise RuntimeError("not enough bytes for popToBits operation")

		for x in range(bytesToPop):
			byte = self.remainingBytes.pop(0)
			bits = utils.toBinary(byte, 8)
			self.firstBitsCache = self.firstBitsCache + bits

	def totalFieldLen(self):
		return len(self.firstBitsCache) + len(self.remainingBytes) * 8 + len(self.lastBitsCache)


	"internal: gets at least n bits extra ready in firstBitsCache"
	def getNBitsReady(self, bitsCount):
		if self.totalFieldLen() < bitsCount:
			raise RuntimeError("not enough bits for getNBitsReady")
		else:
			while len(self.firstBitsCache) < bitsCount:
				# push bytes to bits
				bytesToGet = int(math.ceil((bitsCount - len(self.firstBitsCache)) / 8.0))
				bytesToGet = min(len(self.remainingBytes), bytesToGet)
				self.popBytesToBitsCache(bytesToGet)

				# if no more bytes, move all bits from one extreme to the other
				# (even if this means getting more bits ready than what the user asked for)
				if self.remainingBytes == []:
					self.firstBitsCache = self.firstBitsCache + self.lastBitsCache
					self.lastBitsCache = ""

	"get n bits from the field, but don't change the field"
	def getFirstNBits(self, bitsCount):
		self.getNBitsReady(bitsCount)

		return self.firstBitsCache[0:bitsCount]

	"pop the first n bits from the field"
	def popFirstNBits(self, bitsCount):
		self.getNBitsReady(bitsCount)
		firstNBits = self.firstBitsCache[0:bitsCount]
		self.firstBitsCache = self.firstBitsCache[bitsCount:]

		return firstNBits

	"push a number of bits, as in a stack (from the top or first bits)"
	def pushNBits(self, bits):
		self.firstBitsCache = bits + self.firstBitsCache
		while len(self.firstBitsCache) >= 8:
			idx = len(self.firstBitsCache) - 8
			self.remainingBytes.insert(0, utils.fromBinary(self.firstBitsCache[idx:]))
			self.firstBitsCache = self.firstBitsCache[0:idx]

	"push a number of bits, as in a queue (from the bottom or last bits)"
	def pushQueueNBits(self, bits):
		self.lastBitsCache = self.lastBitsCache + bits
		while len(self.lastBitsCache) >= 8:
			idx = 8
			self.remainingBytes.append(utils.fromBinary(self.lastBitsCache[0:idx]))
			self.lastBitsCache = self.lastBitsCache[idx:]

	# returns all bytes if the data stored can be returned as bytes
	def getAllBytes(self):
		if self.firstBitsCache != "" or self.lastBitsCache != "":
			raise RuntimeError("can't getAllBytes from bitField; not all data stored in bytes now")
		else:
			return self.remainingBytes


#UTILS

import config
import math
import re

def wordListToText(strl):
	text = ""
	lastWord = config.startSymbol

	for word in strl:
		if lastWord == config.startSymbol and word != config.startSymbol:
			word = word[0].capitalize() + word[1:]

		if word != config.startSymbol and text != "":
			text = text + " "

		if not(text == "" and word == config.startSymbol):
			if word == config.startSymbol:
				text = text + "."
			else:
				text = text + word

		lastWord = word


	return text.rstrip("")

def textToWordList(text):
	words = re.findall(r"(?:\w[\w']*)|\.", text)

	def convert(w):
		if w == ".":
			return config.startSymbol
		else:
			return w.lower()

	words = [convert(w) for w in words]

	return words

def fromBinary(numberStr):
	return int(numberStr, 2)

def toBinary(number, minDigits):
	binary = bin(int(number))[2:]

	if len(binary) < minDigits:
		binary = "".join(["0" for x in range(minDigits - len(binary))]) + binary

	return binary

def binaryLowerThan(a, b):
	if len(a) != len(b):
		raise RuntimeError("can't compare two binary numbers of different size")
	else:
		return a < b

def binaryLowerEqualThan(a, b):
	return (a == b or binaryLowerThan(a, b))

"""
this function expands digitRanges to make them cover at least as many values as those in desiredRangeLen

- digitRanges: the actual range of digits
- rangeNums: the ranges already converted to integers
- desiredRangeLen: length goal; how many numbers must contain the range (eg. if this value is 256, the range needs 8 bits)
- maxDigits: max length allowed for the range, in bits
"""
def addDigitsToRange(digitRanges, rangeNums, desiredRangeLen, maxDigits):

	rangePossibleValues = 1 + rangeNums[1] - rangeNums[0]

	if desiredRangeLen <= rangePossibleValues:
		return digitRanges

	extraDigitsCount = int(math.ceil(math.log(1.0 * desiredRangeLen / rangePossibleValues, 2)))
	if len(digitRanges[0]) + extraDigitsCount > maxDigits:
		extraDigitsCount = maxDigits - len(digitRanges[0])

	digitRanges = (
		digitRanges[0] + "".join(["0" for x in range(extraDigitsCount)]),
		digitRanges[1] + "".join(["1" for x in range(extraDigitsCount)])
		)

	return digitRanges

"""
- digitRanges: a pair of binary numbers (strings), telling what the range to subdivide is
- wordProbabilities: a list of elements in this format: (word, (numerator, denominator))
- maxDigits: maximum digits possible in the digit ranges

returns a list of elements in this format: (word, range)
"""
def computeWordRanges(digitRanges, wordProbabilities, maxDigits):

	denominator = wordProbabilities[0][1][1]
	rangeNums = (fromBinary(digitRanges[0]), fromBinary(digitRanges[1]))

	# add more binary digits to range, if needed
	digitRanges = addDigitsToRange(digitRanges, rangeNums, denominator, maxDigits)

	rangeNums = (fromBinary(digitRanges[0]), fromBinary(digitRanges[1]))

	totalDigits = len(digitRanges[0])

	# typical double is 53 bits long; we limit range to some lower amount of bits
	if math.log(max(1, abs(rangeNums[1] - rangeNums[0])), 2) > 45:
		raise RuntimeError("error; range too long")

	# compute word ranges
	# first we compute float ranges, then we distribute the actual integer ranges as well as possible
	step = (1.0 * (rangeNums[1] - rangeNums[0])) / denominator

	base = rangeNums[0]
	start = 0

	wordRanges = []
	for wordP in wordProbabilities:
		end = start + wordP[1][0] * step

		wordRanges.append([wordP[0], [start, end]])
		start = end

	# the last element could be wrong because of float precision problems, force it
	# it is very important that we force this change in wordRanges and not in wordRanges2; otherwise the list could lose extra elements
	wordRanges[-1][1][1] = rangeNums[1] - base

	start = 0

	wordRanges2 = []
	for wordR in wordRanges:
		if wordR[1][1] >= start:
			wordR2 = [wordR[0], [start, int(math.floor(wordR[1][1]))]]
			wordR3 = [wordR2[0], [wordR2[1][0]+base, wordR2[1][1]+base]]
			wordRanges2.append(wordR3)

			start = wordR2[1][1] + 1

	# convert to binary before returning
	return [
		(wordP[0], (toBinary(wordP[1][0], totalDigits), toBinary(wordP[1][1], totalDigits)))
		for wordP in wordRanges2]

# given a range, removes its common digits (excepting the last one); doesn't modify the original range object
def removeCommonBitsInRange(rangeDigits):
	while len(rangeDigits[0]) > 1 and rangeDigits[0][0] == rangeDigits[1][0]:
		rangeDigits = (rangeDigits[0][1:], rangeDigits[1][1:])

	return rangeDigits

# converts an integer to a list of bytesCount bytes
def convertNumberToByteList(number, bytesCount):
	bytes = []

	for i in range(bytesCount):
		b = number % 256
		number = (number - b) / 256
		bytes.insert(0, b)

	bytes.reverse()
	return bytes

# converts an integer to a list of bytesCount bytes
def convertByteListToNumber(bytes):
	number = 0

	for b in reversed(bytes):
		number = number * 256 + b

	return number

# set all words to lower case, except the start symbol
def lowerWord(w):
	if w != config.startSymbol:
		return w.lower()
	else:
		return w


def lowerWordOrList(word):
	if type(word) is list:
		return [lowerWord(w) for w in word]
	elif type(word) is tuple:
		return tuple([lowerWord(w) for w in word])
	else:
		return lowerWord(word)

def listToTuple(t):
	if type(t) == list:
		return tuple(t)
	else:
		return t

def markovChainToDictionary(markovChain):
	dictionary = {}

	for wp in markovChain:
		dictionary[lowerWordOrList(listToTuple(wp[0]))] = wp

	return dictionary


#MARKOV.PY

import json
import math
import re
import random

import utils
import config

# CONFIG
minLineLen = 4


def countRepeatedWords(words):
	# given a list of words, count how many times each one is listed; case insensitive
	count = {}

	for word in words:
		w = word.lower()

		if w in count:
			count[w] = (word, count[w][1] + 1)
		else:
			count[w] = (word, 1)

	return count.values()


def computeProbabilities(words):
	# given a list of words, compute the probability (in a fraction) for each word
	count = countRepeatedWords(words)

	total = sum([c[1] for c in count])
	return [(c[0], (c[1], total)) for c in count]

# wordsPerState is either 1 (the chain keeps probabilities per bigram; 1 input word to 1 output word) or 2
# (the chain keeps probabilities for each 2 input words that go to 1 output word)
def createMarkovChain(inputData, wordsPerState):
	# split sentences, get bigrams
	lines = [re.findall(r"\w[\w']*", line) for line
		in re.split(r"\r\n\r\n|\n\n|\,|\.|\!", inputData)]
	lines = [[config.startSymbol] + line + [config.startSymbol] for line
		in lines if len(line) >= minLineLen]

	if wordsPerState == 1:
		bigrams = [[(line[word], line[word+1]) for word in range(len(line)-1)] for line in lines]
	elif wordsPerState == 2:
		bigrams1 = [[(line[word], line[word+1], line[word+2]) for word in range(len(line)-2)] for line in lines]
		# add special (start, start) -> out cases
		bigrams2 = [[(line[0], line[0], line[1])] for line in lines]
		bigrams = bigrams1 + bigrams2
	else:
		raise RuntimeError("wordsPerState should be either 1 or 2 only")

	# compute markov chain
	# in this context, we call bigrams the pairs (input state, output state); not the best name
	# when the input state has more than 1 word unfortunately
	bigramsDict = {}

	def addBigramToDict(word1, word2):
		word1b = utils.lowerWordOrList(word1)

		if word1b in bigramsDict:
			(w1, w2) = bigramsDict[word1b]
			w2.append(word2)
		else:
			bigramsDict[word1b] = (word1, [word2])

	for line in bigrams:
		for bigram in line:
			if wordsPerState == 1:
				addBigramToDict(bigram[0], bigram[1])
			elif wordsPerState == 2:
				addBigramToDict((bigram[0], bigram[1]), bigram[2])

	fullBigrams = bigramsDict.values()

	fullBigrams = [(bigram[0], computeProbabilities(bigram[1])) for bigram in fullBigrams]
	# at this point, fullBigrams contains the markovChain with probabilities in fractions

	return fullBigrams


# wordsPerState is either 1 (the chain keeps probabilities per bigram; 1 input word to 1 output word) or 2
# (the chain keeps probabilities for each 2 input words that go to 1 output word)
def createMarkovChainFromFile(inputFile, outputFile, wordsPerState):
	f = open(inputFile, 'r')
	inputData = f.read()
	f.close()

	bigrams = createMarkovChain(inputData, wordsPerState)

	# save
	jsonData = json.JSONEncoder().encode(bigrams)
	f = open(outputFile, 'w')
	f.write(jsonData)
	f.close()



# check markov file
def testMarkovChain(inputMarkov):
	f = open(inputMarkov, 'r')
	jsonData = f.read()
	f.close()

	data = json.JSONDecoder().decode(jsonData)

	errors = 0

	for bigram in data:
		(wordFrom, wordsTo) = bigram
		total = wordsTo[0][1][1]
		total2 = 0

		for word in wordsTo:
			total2 = total2 + word[1][0]

		if total != total2:
			print ("error, denominator and total numerators are different!")


	if errors == 0:
		print ("OK: no errors found in markov file"	)
	else:
		print ("ERROR: " + repr(errors) + " errors found in markov file")


# input is a markov chain
# see createMarkovChain for a description of the parameter wordsPerState
def generateTextUsingMarkovChain(inputMarkov, wordsPerState):
	f = open(inputMarkov, 'r')
	jsonData = f.read()
	f.close()

	data = json.JSONDecoder().decode(jsonData)

	words = []
	if wordsPerState == 1:
		prev = config.startSymbol
	elif wordsPerState == 2:
		prev = (config.startSymbol, config.startSymbol)

	markovDict = {}
	for bigram in data:
		markovDict[utils.lowerWordOrList(utils.listToTuple(bigram[0]))] = bigram[1]

	while True:
		m = markovDict[utils.lowerWordOrList(prev)]
		denominator = m[0][1][1]
		rnd = random.randint(1, denominator)
		total = 0
		nextWord = None

		for word in m:
			total = total + word[1][0]
			if total >= rnd:
				nextWord = word[0]
				break

		if nextWord == config.startSymbol:
			break

		words.append(nextWord)

		if wordsPerState == 1:
			prev = nextWord
		elif wordsPerState == 2:
			prev = (prev[1], nextWord)

	return words



# CONFIG
startSymbol = "<START>"


# used by many tests
testMarkov = [
	[startSymbol, [ ["A", [1, 3]], ["B", [1, 3]], ["C", [1, 3]] ]],
	["A", [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	["B", [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	["C", [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]]
	]

testMarkov2 = [
	[[startSymbol, startSymbol], [ ["A", [1, 3]], ["B", [1, 3]], ["C", [1, 3]] ]],
	[[startSymbol, "A"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	[[startSymbol, "B"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	[[startSymbol, "C"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	[["A", "A"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	[["A", "B"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	[["A", "C"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	[["B", "A"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	[["B", "B"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	[["B", "C"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	[["C", "A"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	[["C", "B"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]],
	[["C", "C"], [ ["A", [1, 4]], ["B", [1, 4]], ["C", [1, 4]], [startSymbol, [1, 4]] ]]
	]

def actual():
    while(True):
        data = input_group("Text Based Steganography",[select('What do you want to do?', ["createMarkov",  "encodeFullText", "decodeFullText",'exit'],name='mode'),
                select('Input File to Create Markov Chain',['example1A.txt','example1B.txt','dream_catcher.txt','hinglish_1.txt','theory_of_everything.txt'], name='Markov_file'),
                   textarea('Input Text', rows=10, placeholder='Some text',name='Input'),

                                    
    
                                 ])
        wordsPerState = 1
        jsonoutput='markovChain.json'
        if data['mode'] == "createMarkov":
            print ("creating markov chain")
            print ("using wordsPerState = " + repr(wordsPerState))
            createMarkovChainFromFile(data['Markov_file'], jsonoutput,wordsPerState)
            print ("done")
    
        elif data['mode'] == "encodeFullText":
            markovInputFile = jsonoutput
            print ("encoding file (number to text) using markov chain, saving as txt")
            encodeDataFromFile('input.data', 'encoded.txt', markovInputFile, True, wordsPerState,data['Input'])
            print ("done")
        elif data['mode'] == "decodeFullText":
            markovInputFile = jsonoutput
            print ("decoding file (text in .txt to number) using markov chain")
            decodeDataFromFile('encoded.txt','output.txt', markovInputFile, True, wordsPerState,data['Input'])
            print ("done")
        if data['mode']=='exit':
            break

app.add_url_rule('/tool', 'webio_view', webio_view(actual),
            methods=['GET', 'POST', 'OPTIONS'])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(actual, port=args.port)
    


