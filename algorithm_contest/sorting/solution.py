import sys 
  
numbers = []
  
for line in sys.stdin: 
    numbers.append(int(line))    

numbers.sort()

for number in numbers:
    print(number)