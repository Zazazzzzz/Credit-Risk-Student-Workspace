# ---------------------------------------------------------------
# Script Name: Game---Guess the Number
# Author: Shazia
# Description: Assignment_1
# ----------------------------------------------------------------
import random
import random

hidden_num = random.randint(1, 30)
# number of tries
attempts = 4
## Displaying Welcome to players
print('Welcome to the Shazzi Game: GUESS THE NUMBER')
print('The number range is from 1 to 30. The allowed attempts are 4')
# setting attemps
for attempt in range(1, attempts+1):
    guess = int(input(f'Attempt {attempt}: Enter your guess: '))

    if guess < 1 or guess > 30:
        print('Invalid guess! Please guess a number between 1 and 30')
        continue

    if guess == hidden_num:
        print(f"Congratulations! You guessed the number {hidden_num} in {attempt} attempts.")
        break

    elif guess < hidden_num:
        print('Too low!')
    else:
        print('Too high!')

    if attempt == attempts:
        print(f"Sorry, you've run out of attempts! The number was {hidden_num}.")