# ---------------------------------------------------------------
# Script Name: Game---Guess the Number
# Author: Hongyi Shen
# Description: Section 1_Assignment
# ----------------------------------------------------------------

import random
hidden_num = random.randint(1,40)
attempts = 4
print('Welcome to the Game: GUESS THE NUMBER')
print('The number range is from 1 to 40. The allowed attemps are 4')

for attempt in range(1, attempts+1):
    guess = int(input(f'Attempt {attempt}: Enter your guess: '))
    if guess < 1 or guess > 40:
        print('Invalid guess! Please guess a number between 1 and 40')
        continue # break out of this time of loop
    if guess == hidden_num:
        print(f"Congratulations! You guessed the number {hidden_num} in {attempt} attempts.")
        break # when it is the right guess, break out of the loop entirely
    elif guess < hidden_num:
        print('Too low!')
    else:
        print('Too high!')

    if attempt == attempts:
        print(f"Sorry, you've run out of attempts! The number was {hidden_num}.")