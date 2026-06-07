from asyncio import print_call_graph

s = 'credit_risk '
s.strip()
s_split = s.split('_')
print(type(s_split))
print(s_split)

s = 'credit risk'
s.split(' ') # 以空格 ' ' 作为分隔符切分字符串
s_replace = s.replace(' ', '_')
print(type(s_split))
print(s_split)

bool_list = [True, False, True]
print(type(bool_list))

type(True) # 输出 bool
type(1)
print(type(True))
print(type(1))

l1 = [1, 2, 3, 'credit', 'risk']
l2 = [1, 4, 7, 8]
l = l1 + l2 # 列表拼接
l1.append(l2)
print(l)
print(l1)

print("credit risk\n\t2025\n\tsummer semester")

print('credit risk')
print('2025')
print()
print()
print()
print('credit risk', end=' ')
print('2025')

a = 42
print(a)
print(f"{a:5}")

squares = [x**2 for x in range(5)]
print(squares)


text = 'the job of a student is to study and the job of a teacher is to teach'
word_counts = {}
for word in text.split():
    print(word)
    word_counts[word] = word_counts.get(word, 0) +1
    print(word_counts[word])

word_count = {}
for word in text.split():
    word_count[word] = word_count.get(word, 0) + 1

for word, num in word_count.items():
    print(word, num)

"""
for i in range(10):
   print(i, end=' ')

print()
"""

for i in range(1,10):
    for j in range(1, i+1):
        value = i * j
        print(f'{value:2}', end=' ')
    print()

'''
for i in range(1, 10):
    for j in range(1, i+1):
        value = i * j

count = 0
while count < 5:
    count = count +1
    if count == 3:
        continue # skip when count is 3
    print(count)


for i in range(5):
    if i == 3:
        pass  # skipping the implementation for i == 3 for now
    else:
        print(i)
'''

for i in range(10):
    if i == 7:
        pass
    elif i%2 != 0:
        print(i)
    else:
        pass

for i in range(1,11):
    if i % 2 == 0:
        continue # 如果是偶数，跳过
    if i == 7:
        continue # 如果是7，跳过
    print(i)