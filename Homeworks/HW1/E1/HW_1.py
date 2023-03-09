# Homework One - Exercise One

# Libraries
import numpy as np

# Code
num = int(input("Please enter the number of lockers: "))

arr = np.zeros((1, num))

for y in range(len(arr[0])):
    for x in range(len(arr[0])):
        if (x + 1) % (y + 1) == 0:
            # print(f"{x + 1} + {y + 1} = {(x + 1) % (y + 1)}")
            if arr[0][x] == 0:
                arr[0][x] = 1
            else:
                arr[0][x] = 0

print(f"These are the lockers that will be open:")

cnt = 0
for x in range(len(arr[0])):
    if arr[0][x] == 1:
        print(x + 1)
        cnt = cnt + 1

print(f"\nThe number of open lockers is: {cnt}")
print(f"To verify, the number of open locker should be the square root of the number of lockers. \nThe square root of the number of lockers is {num ** 0.5}")
