# ---------------------------------------------------------------
# Script Name: Game---Guess the Number
# Author: Hongyi Shen
# Description: Section 1_Assignment
# ----------------------------------------------------------------

import random
hidden_num = random.randint(1,40)     # 随机生成一个包含a，b的整数 （random int）
attempts = 4
print('Welcome to the Game: GUESS THE NUMBER')
print('The number range is from 1 to 40. The allowed attemps are 4')

for attempt in range(1, attempts+1):
    guess = int(input(f'Attempt {attempt}: Enter your guess: '))
    if guess < 1 or guess > 40:
        print('Invalid guess! Please guess a number between 1 and 40')
        continue # break out of this time of loop   # 跳过本次循环剩下所有代码，直接进入下一次循环。这意味着，如果玩家输错了范围，虽然浪费了一次机会，但不会触发后面的“太高”或“太低”提示。
# if 必须有且只能有一个（作为整个判断的开头）。
# elif（else if 的缩写）可以有任意多个（可以是 0 个、1 个、10 个甚至上百个），用来检查更多不同的其他条件。
# else 只能有一个，且必须放在最后（作为所有条件都不满足时的“兜底”选项）。
# 在第 14 行检查完范围后，第 17 行用了 if 而不是 elif。这是因为第 16 行有一个 continue。如果前面范围错误，continue 会让程序直接跳到下一次循环，根本不会执行到第 17 行。所以这里用 if 或 elif 效果是一样的。
    if guess == hidden_num:
        print(f"Congratulations! You guessed the number {hidden_num} in {attempt} attempts.")
        break # when it is the right guess, break out of the loop entirely   # 一旦猜对，break 会立刻彻底终止整个 for 循环，游戏提早结束。
    elif guess < hidden_num:
        print('Too low!')
    else:
        print('Too high!')

    if attempt == attempts:
        print(f"Sorry, you've run out of attempts! The number was {hidden_num}.")