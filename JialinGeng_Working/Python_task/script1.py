# ---------------------------------------------------------------
# Script Name: Data Type and Control FLow Statement Introduction
# Author: Hongyi Shen
# Description: Section 1
# ----------------------------------------------------------------


# Basic concepts of programing languages
# Python and R follow the general principles of encoding characters to numeric representations and decoding them back.
# The specifics of how these encodings are implemented and managed can differ, but the fundamental goal is to represent and manipulate text data consistently.

# Setting up
# Create a new project, setting up virtual environments in PyCharm, installing necessary libraries (pandas, numpy, matplotlib) via PyCharm
# Project: <Your Project Name> > Python Interpreter
# pip install <library-name>
# pip freeze > requirements.txt (requirements.txt, is used to specify the libraries your project depends on.
# This file lists each library and its version, making it easy to replicate the environment elsewhere.)
# pip install -r requirements.txt (Use the requirements.txt to install all the necessary libraries.)'''

######## Task
# Set up a new project named 'Block_course'
# installed packages 'pandas', 'numpy', 'matplotlib', and 'seaborn'
# create a requirments.txt


####### Basic data type
### Strings are sequences of characters enclosed in single quotes ('...'), double quotes ("..."), or triple quotes ('''...''' or """...""" for multi-line strings).
s = 'credit_risk'
type(s)
help(type)
# use help() function to get documentation and information about objects, functions, modules, or classes.
len(s) # Returns the length of the string
## The . is used to access attributes or methods of an object in Python
## Empty Parentheses: When the parentheses are empty (like s.upper()), it means the method doesn’t require any arguments, and it typically operates on the object itself
s.upper() # Converts all characters to uppercase
s.lower() # Converts all characters to lowercase
s = 'credit_risk '
s.strip() # Removes leading and trailing whitespace
s_split = s.split('_') # Splits the string into a list using the delimiter
s_split_0 = s.split('_')[0]
s = 'credit risk'
s.split(' ')
s_replace = s.replace(' ', '_')
## Arguments Inside Parentheses: If the method requires arguments, you would pass them inside the parentheses
del s # deleting object


### int and float
type(4)
type(0.6)
int(0.6) # Converting a float to an integer truncates the decimal part.
round(0.6)
round(0.639, 2)
float(4)
round(4, 2)
type(round(4, 2))
type(round(float(4),2))
formatted = f"{4:.3f}"
# :.3f: The 3 specifies the number of decimal places, and f indicates that you want a floating-point number
# f"": This denotes an f-string (formatted string literal), which is a way to embed expressions inside string literals using {}.
type(formatted)
a = 7
b = 3
print(a + b)  # Output: 10
print(a - b)  # Output: 4
print(a * b)  # Output: 21
print(a / b)  # Output: 2.3333...
print(a // b)  # Output: 2 (integer division)
print(a % b)   # Output: 1 (remainder of division)
print(a ** 2) # Computes the square of a
print(a ** 0.5) # Computes the square root of a

### boolean type (or bool) is a data type that can have one of two possible values: True or False.
x = 5
y = 10
print(x == y)
print(x < y)
a = True
b = False
c = False
print(a and b) # if both a and b are true, then the whole thing is true
print(a or b) # if either a or b is true, then the whole part is true
print(not a)
a or b and c # `and` is evaluated first, so this is equivalent to: a or (b and c)
# boolean values (True and False) can be used in arithmetic operations and are often treated as integers in such contexts.
# True is equivalent to the integer 1.
# False is equivalent to the integer 0.
bool_list = [True, False, True]
sum(bool_list)
print(True == 1)
print(False == 0)
print(True == 0)
type(True)
type(1)  # boolean as a subclass of integer
print(True + 1)
print(True - 1)


### Lists are ordered collections of items that can be of different types.
l = [] # an empty list, square brackets
l = list()
l = list([1,2])
l1 = [1, 2, 3, 'credit', 'risk']
l2 = [1, 4, 7, 8]
l3 = ['credit', 'risk']
l4 = [1, 2, 3, 'credit', 'risk', 3.4, True]
l5 = [1, 2, 3, 'credit', 'risk', 3.4, True, [1, 2]]
l = l1 + l2
l1.append(l2)
len(l1)
len(l2)
l2.append(9)
l2.append([4, 5]) # Append an element
l2.extend([0, 8]) # Extend the list with another list
l.insert(5, 'measurement') # Insert an element at index 4
l2.remove([4, 5]) # list.remove() takes exactly one argument
popped = l1.pop(4) # Pop an element by index (or the last element if no index is provided)
l2.pop(5)
l2.index(8) # Find the index of the first occurrence of 8
l2.count(8) # Count the occurrences of 8
l2.sort()
l2 = sorted(l2, reverse=True)
l1.reverse()
l.reverse() # Reverse the list in place
a = l2[0] # Access the first element, using the square brackets to access elements from indexable objects
l2[-1] # Access the last element
l2[-2] # Second to the last
l2[1:4] # Get elements from index 1 to 3 (exclusive of 4)
l2[:-1]
l2[1:-1]
del l2[0]

### Tuples are ordered collections of items similar to lists, but they are immutable
# meaning their content cannot be changed after creation.
t = () # parentheses
t = tuple()
t = tuple((1,2))
# t = tuple(1,2) TypeError: tuple expected at most 1 argument, got 2
# l = list(1,2)
t1 = (1, 2, 3, 'credit', 'risk') # directing creating a tuple with parentheses
l = [1, 2, 3]
t2 = tuple(l) # convert a list into a tuple
t1.count(2) # Counts the occurrences of an item.
t1.index('credit') # Returns the index of the first occurrence of an item.
t1[0]
# t1[5] = 7 TypeError: 'tuple' object does not support item assignment
# del t1[0]

### Dictionary is a collection of key-value pairs.
d = {} # curly brackets
d = dict()
d = {'name': 'Alice', 'age': 22, 'job': 'Student'}
d = dict({'name': 'Alice', 'age': 22, 'job': 'Student'})
# d[0], KeyError: 0, a dictionary doesn't have orders, and is based on key
d['name']
d['gender'] = 'Female' # add a new key-value pair
d['age'] = 23 # updating an existing pair
del d['gender'] # remove a pair
d.pop('age') # remove a pair
d.popitem() # remove the last pair
d['job'] = 'Student'
d['age'] = 23
d.get('name') # help(dict.get)
len(d)
infos = {'A': {'name': 'Max', 'age': 30}, 'B': {'name': 'John', 'age': 27}} # nested dictionaries
infos['A']['name']

### print()
print('credit risk')
print('credit risk', '2025', 'summer semester', sep=',')
print('credit\nrisk') # \n is used to insert new lines
print("credit risk\n\t2025\n\tsummer semester") # \t adds a tab space before
# two lines
print('credit risk')
print('2025')
# one line
# end=' ' keeps the products on the same line, separated by a space.
# print() with no arguments creates a new line after each row is printed.
print('credit risk', end=' ')
print('2025')
# format string
course = 'Credit risk'
times = 'eight'
print(f'{course} has {times} classes')
a = 1.3423421
print(f'{a:.2f}') # format numbers to a specific precision printed as a string, type(f'{a:.2f}')
print(round(a, 2)) # type(round(a, 2))
a = 42
print(a)
print(f"{a:5}")  # Pads with spaces to make the total width 5

####### Control flow statements
# Conditional Statements: if, elif, else
# Looping Statements: for, while
# Control Flow Modifiers: break, continue, pass
# Function Control: return
# Exception Control: raise, try/except

### Conditional Statements
# if:  checks if a condition is true
# elif statement allows you to check multiple conditions. If the if condition is false, Python moves to the elif condition
# else statement catches all conditions that were not met by the preceding if or elif statements

grade = 30
if grade >= 90:
    print("Excellent")
elif grade >= 80:
    print("Good")
elif grade >= 70:
    print("Fair")
else:
    print("Needs Retake")

### Looping Statements
## for loops: Used to iterate over a sequence (like a list, tuple, string, or dictionary) or any iterable object.
fruits = ['apple', 'banana', 'cherry']
for f in fruits:
    print(f)
for index, fruit in enumerate(fruits): # enumerate() in for loops to get both the index and the value of the items in an iterable.
    print(index, fruit)

# loop through the keys, values, or both
for key in d:
    print(key)
for key in d.keys():
    print(key)
for value in d.values():
    print(value)
for key, value in d.items():
    print(key, value)


# Create a list of squares
squares = [x**2 for x in range(5)]
print(squares)
# A range object is an immutable sequence of numbers that is memory-efficient because it generates the numbers
# on the fly as you iterate over it, rather than storing them in memory all at once.
for x in range(5): print(x)

# Print the even number until 20
even = [x for x in range(21) if x % 2 == 0]
print(even)

# usinging loop to create a dictionaries num : num**2
squares = {x: x**2 for x in range(5)}
print(squares)


### Task A: word count
text = 'the job of a student is to study and the job of a teacher is to teach'
# how many times each word shows up

word_count = {}
for word in text.split():
    word_count[word] = word_count.get(word, 0) + 1

for word, num in word_count.items():
    print(word, num)


word_counts = {}
for word in text.split():
    print(word)
    word_counts[word] = word_counts.get(word, 0) +1
    print(word_counts[word])


### Task B: Print the multiplication table in a triangular format
for i in range(10):
    print(i, end=' ')


for i in range(1,10):
    for j in range(1, i+1):
        value = i * j
        print(f'{value:2}', end=' ')
    print() # Move to the next line after each row


### Task C: Filtering Even Numbers
numbers = [1, 2, 3, 4, 5, 6, 9, 11, 13, 14, 18, 24, 26]
even_numbers = [x for x in numbers if x % 2 == 0]
print(even_numbers)


## while loop: A while loop continues to execute a block of code as long as a given condition is True.
count = 0
while count < 5:
    print(count)
    count = count + 1 #count += 1

## break, continue, pass
# break: to exit the loop entirely when a certain condition is met
for i in range(1, 10):
    if i == 5:
        break  # exit the loop when i is 5
    print(i)

count = 0
while True:  # infinite loop
    print(count)
    count = count + 1
    if count == 3:
        break  # exit the loop when count reaches 3

# continue: to skip the current iteration of the loop and move to the next one, without breaking the loop.
for i in range(1, 10):
    if i % 2 != 0:  # Check if the number is odd
        continue  # Skip the odd numbers
    print(i)

count = 0
while count < 5:
    count = count +1
    if count == 3:
        continue # skip when count is 3
    print(count)

# pass: used as a placeholder.
# It does nothing and allows you to write code where a statement is syntactically required but you don’t want to execute anything yet.
for i in range(5):
    if i == 3:
        pass  # skipping the implementation for i == 3 for now
    else:
        print(i)


for i in range(1, 10):
    pass



### Task A: print the odd number from the range of 1 to 10, but skip 7
for i in range(1,11):
    if i % 2 == 0:
        continue
    if i == 7:
        continue
    print(i)

### Task B: guess the hidden number, the number range is from 1 to 30, the allowed attempts are 4
# pass a string argument to input() to display a prompt to the user
# when input() is called, the program pauses and waits the user type something in the terminal or console
# when the user presses enter, the program continues
name = input("Enter your name: ")
print("Hello, " + name + "!")


