# Homework One - Exercise Two

# Libraries
import numpy as np
import math as m

# Code
a = np.array([[1, -2, 2], [-1, 1, 3], [1, -1, -4]])
letDic = {" ":0, "A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7, "H":8, "I":9, "J":10, "K":11, "L":12, "M":13,
		 "N":14, "O":15, "P":16, "Q":17, "R":18, "S":19, "T":20, "U":21, "V":22, "W":23, "X":24, "Y":25, "Z":26}

# Take in message from user and convert characters to upper case
message = input("Please enter a message to be encoded: ").upper()

# Create Uncoded Row Matricies and Coded Row Matricies
uRM = np.array(list([0, 0, 0] for _ in range(m.ceil(len(message) / 3))))
cRM = np.array(list([0, 0, 0] for _ in range(m.ceil(len(message) / 3))))
dRM = np.array(list([0, 0, 0] for _ in range(m.ceil(len(message) / 3))))

# Initialize the variable i with the value 0
i = 0

# Assign uRM indexes with numerical letter value from user message
for k in range(len(uRM)):
	for j in range(0, 3):
		if i >= len(message):
			uRM[k][j] = letDic.get(" ")
			
		else:
			uRM[k][j] = letDic.get(message[i])
			
		i = i + 1

print(f"Forming Uncoded Row Matrices of message:\n{uRM}\n")

# Initialize the variable cryptogram with an empty list to store cryptogram
cryptogram = []
# Encoding Message
for k in range(len(cRM)):
	cRM[k] = np.matmul(uRM[k], a)

for k in range(len(cRM)):
	for j in range(0, 3):
		cryptogram.append(cRM[k][j])

print(f"Cryptogram of message:")
for x in cryptogram:
	print(x, end=" ")
print("\n")

# Inverse Matrix a
a = np.linalg.inv(a)

# Decoding Message
for k in range(len(cRM)):
	dRM[k] = np.matmul(cRM[k], a)

print(f"Decoded Row Matrices of message:\n{dRM}")

# Initialize the variable message with an empty list to store the original message
message = []

# Original Message
for k in range(len(dRM)):
	for j in range(0, 3):
		message.append(list(letDic.keys())[list(letDic.values()).index(dRM[k][j])])

print(f"\nOrignal Message:\n{''.join(message)}")
