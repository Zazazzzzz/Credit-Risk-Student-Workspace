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
# 激活venv： Windows：.\venv\Scripts\activate  Mac/Linux：source venv/bin/activate
# installed packages 'pandas', 'numpy', 'matplotlib', and 'seaborn'
# pip install pandas numpy matplotlib seaborn
# create a requirments.txt
# pip freeze > requirements.txt


####### Basic data type
### Strings are sequences of characters enclosed in single quotes ('...'), double quotes ("..."), or triple quotes ('''...''' or """...""" for multi-line strings).
s = 'credit_risk'
type(s)
help(type)
# use help() function to get documentation and information about objects, functions, modules, or classes.
len(s) # Returns the length of the string
## The . is used to access attributes or methods of an object in Python
## Empty Parentheses: When the parentheses are empty (like s.upper()), it means the method doesn’t require any arguments, and it typically operates on the object itself
## 空括号：当括号内为空时（例如 s.upper()），意味着该方法不需要任何外部参数，它通常直接作用于对象本
s.upper() # Converts all characters to uppercase
s.lower() # Converts all characters to lowercase
s = 'credit_risk '
s.strip() # Removes leading and trailing whitespace
s_split = s.split('_') # Splits(切开) the string into a list using the delimiter
s_split_0 = s.split('_')[0]
s = 'credit risk'
s.split(' ')
s_replace = s.replace(' ', '_')
## Arguments Inside Parentheses: If the method requires arguments, you would pass them inside the parentheses
del s # deleting object


### int and float
type(4)
type(0.6)
int(0.6) # Converting a float to an integer truncates the decimal part.  将浮点数转换为整数（会直接截断抹去小数部分，结果为 0）
round(0.6)  # 对浮点数进行四舍五入（结果为 1）
round(0.639, 2)  # 保留两位小数进行四舍五入（结果为 0.64）
float(4)
round(4, 2)
type(round(4, 2))
type(round(float(4),2))
formatted = f"{4:.3f}"    # 格式化为字符串
# :.3f: The 3 specifies the number of decimal places, and f indicates that you want a floating-point number
# :.3f：其中数字 3 指定了保留的小数位数，f 表示你想将其作为浮点数格式化
# f"": This denotes an f-string (formatted string literal), which is a way to embed expressions inside string literals using {}.
# # f""：这代表 f-string（格式化字符串字面量），是一种通过大括号 {} 在字符串中嵌入表达式的便捷方法。
type(formatted)  # 查看格式化后的类型（输出 str，即变成了字符串）
a = 7
b = 3
print(a + b)  # Output: 10
print(a - b)  # Output: 4
print(a * b)  # Output: 21
print(a / b)  # Output: 2.3333...
print(a // b)  # Output: 2 (integer division) 整除
print(a % b)   # Output: 1 (remainder of division) 取余
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
# 布尔值（True 和 False）可以用于算术运算，在这些场景下它们经常被当作整数对待。
# True is equivalent to the integer 1.
# False is equivalent to the integer 0.
bool_list = [True, False, True]
sum(bool_list)
print(True == 1)
print(False == 0)
print(True == 0)
type(True)
type(1)  # boolean as a subclass of integer  (注：在 Python 内部，bool 是 int 的子类)
print(True + 1)
print(True - 1)


### Lists are ordered collections of items that can be of different types.
l = [] # an empty list, square brackets
l = list()
l = list([1,2])  # 此时，列表中的元素依然是独立的数字（整数），而不是一个列表。因为list()这个方法
l1 = [1, 2, 3, 'credit', 'risk']
l2 = [1, 4, 7, 8]
l3 = ['credit', 'risk']
l4 = [1, 2, 3, 'credit', 'risk', 3.4, True]
l5 = [1, 2, 3, 'credit', 'risk', 3.4, True, [1, 2]]
l = l1 + l2  # 列表拼接
l1.append(l2) # 将整个 l2 作为一个元素追加到 l1 的末尾
len(l1)
len(l2)
l2.append(9)
l2.append([4, 5]) # Append an element
l2.extend([0, 8]) # Extend the list with another list   用另一个列表来扩充当前列表（把里面的元素逐个拆开加进去）
l.insert(5, 'measurement') # Insert an element at index 5
l2.remove([4, 5]) # list.remove() takes exactly one argument
popped = l1.pop(4) # Pop an element by index (or the last element if no index is provided)   弹回/删除指定索引（此处为4）的元素，若不传参则默认弹出最后一个元素
l2.pop(5)
l2.index(8) # Find the index of the first occurrence of 8
l2.count(8) # Count the occurrences of 8
l2.sort()  # 对 l2 进行原位排序（升序）
l2 = sorted(l2, reverse=True)  # 使用 sorted() 函数进行逆序排序并返回新列表
l1.reverse()  # 将 l1 列表倒序排列（原位修改）
l.reverse() # Reverse the list in place
a = l2[0] # Access the first element, using the square brackets to access elements from indexable objects   访问第一个元素，使用方括号通过索引访问支持索引的对象
l2[-1] # Access the last element
l2[-2] # Second to the last
l2[1:4] # Get elements from index 1 to 3 (exclusive of 4)
l2[:-1]  # 切片：获取从开头直到倒数第一个元素之前的所有元素
l2[1:-1]  # 切片：获取从索引 1 开始直到倒数第一个元素之前的所有元素
del l2[0]  # 删除 l2 中索引为 0 的元素

### Tuples are ordered collections of items similar to lists, but they are immutable(不可改变)
# meaning their content cannot be changed after creation.
t = () # parentheses
t = tuple()
t = tuple((1,2))
# t = tuple(1,2) TypeError: tuple expected at most 1 argument, got 2
# l = list(1,2)
# 这两个是转换函数，只能接收一个可迭代参数。
t1 = (1, 2, 3, 'credit', 'risk') # directing creating a tuple with parentheses
l = [1, 2, 3]
t2 = tuple(l) # convert a list into a tuple
t1.count(2) # Counts the occurrences of an item.
t1.index('credit') # Returns the index of the first occurrence of an item.
t1[0]
# t1[5] = 7 TypeError: 'tuple' object does not support item assignment
# del t1[0]

### Dictionary is a collection of key-value pairs.
d = {} # curly brackets  花括号
d = dict()
d = {'name': 'Alice', 'age': 22, 'job': 'Student'}
d = dict({'name': 'Alice', 'age': 22, 'job': 'Student'})
# d[0], KeyError: 0, a dictionary doesn't have orders, and is based on key
d['name']
d['gender'] = 'Female' # add a new key-value pair
d['age'] = 23 # updating an existing pair
del d['gender'] # remove a pair
d.pop('age') # remove a pair  移除一个键值对并返回其值
d.popitem() # remove the last pair  移除并返回字典中的最后一对键值
d['job'] = 'Student'
d['age'] = 23
d.get('name') # help(dict.get)  获取指定键的值（可用 help(dict.get) 查看其高级用法，如设置默认返回值）
len(d)
infos = {'A': {'name': 'Max', 'age': 30}, 'B': {'name': 'John', 'age': 27}} # nested dictionaries  嵌套字典
infos['A']['name']  # 访问嵌套字典中的数据（获取 A 的 name，结果为 'Max'）

### print()
print('credit risk')
print('credit risk', '2025', 'summer semester', sep=',')  # sep=',' 表示用逗号作为多个输出项之间的分隔符
print('credit\nrisk') # \n is used to insert new lines
print("credit risk\n\t2025\n\tsummer semester") # \t adds a tab space before  # \t 用于插入一个制表符（缩进进空）

# two lines
print('credit risk')
print('2025')

# one line
# end=' ' keeps the products on the same line, separated by a space.   end=' ' 让输出留在同一行，并以空格结尾，而不是默认的换行。
# print() with no arguments creates a new line after each row is printed.
print('credit risk', end=' ')
print('2025')
# format string
course = 'Credit risk'
times = 'eight'
print(f'{course} has {times} classes')
a = 1.3423421
print(f'{a:.2f}') # format numbers to a specific precision printed as a string, type(f'{a:.2f}')  将数字格式化为特定精度（两位小数）并以字符串形式打印，查看类型为 str
print(round(a, 2)) # type(round(a, 2))
a = 42
print(a)
print(f"{a:5}")  # Pads(填充) with spaces to make the total width 5
# 如果 a 是数字，默认会向右对齐（左边补空格）；如果 a 是字符串，默认会向左对齐（右边补空格）。

####### Control flow statements
# Conditional Statements: if, elif, else
# Looping Statements: for, while
# Control Flow Modifiers: break, continue, pass  控制流修饰符/跳转语句：break、continue、pass
# Function Control(函数控制): return
# Exception Control(异常控制): raise, try/except

### Conditional Statements
# if:  checks if a condition is true
# elif statement allows you to check multiple conditions. If the if condition is false, Python moves to the elif condition
# else statement catches all conditions that were not met by the preceding if or elif statements    else 语句用于捕捉所有未能满足前方 if 或 elif 条件的其他剩余情况

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
## for loops: Used to iterate(迭代) over a sequence (like a list, tuple, string, or dictionary) or any iterable object.
fruits = ['apple', 'banana', 'cherry']
for f in fruits:
    print(f)
for index, fruit in enumerate(fruits): # enumerate() in for loops to get both the index and the value of the items in an iterable.   在 for 循环中使用 enumerate() 可以同时获取迭代项的索引(位置)和具体数值。
    print(index, fruit)

# loop through the keys, values, or both
for key in d:
    print(key)  # 默认遍历键
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
# on the fly(动态地，实时地) as you iterate over it, rather than storing them in memory all at once.
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


### Task B: Print the multiplication table(乘法表) in a triangular format
for i in range(10):
    print(i, end=' ')


for i in range(1,10):
    for j in range(1, i+1):    # 解释：这是内层循环。range(1, i+1) 会生成一个从 1 到 i 的整数序列（注意：不包含 i+1，所以最大值恰好是 i）。
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
while True:  # infinite loop  无限循环
    print(count)
    count = count + 1
    if count == 3:
        break  # exit the loop when count reaches 3

# continue: to skip the current iteration of the loop and move to the next one, without breaking the loop.
# continue：跳过当前这一轮循环中剩余的代码，直接进入下一轮迭代，并不会终止整个循环。
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
# 它什么都不做。当语法上需要一条语句，但你目前还不想编写任何具体执行代码时，可以用它来占位防止报错。
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
# 向 input() 函数传递一个字符串参数，可以向用户显示提示信息
# when input() is called, the program pauses and waits the user type something in the terminal or console
# 当 input() 被调用时，程序会暂停，并等待用户在终端或控制台中输入内容
# when the user presses enter, the program continues
# 当用户按下回车键（Enter）时，程序才会获取输入并继续向下执行
name = input("Enter your name: ")
print("Hello, " + name + "!")


