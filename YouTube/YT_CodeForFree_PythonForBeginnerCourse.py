#### Lists

lucky_numbers = [4, 8, 15, 16, 23, 42]
friends = ["Kevin", "Karen", "Jim", "Oscar", "Tom"]

friends.extend(lucky_numbers) # extends an list with another list

friends.append("Ana") # automatically adds an element at the end

friends.insert(2, "Barbara") # adds an element at a given position

friends.remove("Jim") # removes a given item

friends.clear() # Clear the entire list

friends.pop() # remove last element

print(friends.index("Kevin"))

print(friends.count("Jim"))

friends.sort()

friends.reverse()

friends2 = friends.copy()

print(friends)

### Tuples

coordinates = (4,5)

### Functions

def say_hi(): # defining a function
    print("Hello User")

say_hi() # calling a function

def sayhi(name, age):
    print("Hello " + name + " , you are " + str(age))

sayhi("Mike", 30)

### Return statement

def cube(num):
    return num*num*num
    #print() won't be executed, only return function will be executed
print(cube(3))

result = cube(4)
print(result)

### If statement

is_male = True
is_tall = True

if is_male or is_tall:
    print("You are a male or tall")
else:
    print("You are not a male")

####

is_male = True
is_tall = False

if is_male and is_tall:
    print("You are a tall male")
elif is_male and not(is_tall):
    print("You are a short male")
elif not(is_male) and is_tall:
    print("You are not a male but you are tall")
else:
    print("You are not a male and not tall")
    
####

def max_num(num1, num2, num3):
    if  num1 >= num2 and num1 >= num3:
        return num1
    elif num2 >= num1 and num2 >= num3:
        return num2
    else:
        return num3
print(max_num(3,4,5))


### Calculator

num1 = float(input("Enter first number: ")) # Converting into a float value
op = input("Enter operator: ")
num2 = float(input("Enter second number: "))

if op == "+":
    print(num1 + num2)
elif op == "-":
    print(num1 - num2)
elif op == "/":
    print(num1 / num2)
elif op == "*":
    print(num1 * num2)
else:
    print("Invalid operator")
 
### Dictionaries

monthConversions = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "May": "May",
    "Jun": "June",
}

print(monthConversions.get("Mar"))

### While loop

i = 1
while i <= 10:
    prints(i)
    i = i + 1 # can be also written as i += 1

print("Done with loop")

### Guessing game

secret_word = "giraffe"
guess = ""  # variable to store the inputs
guess_count = 0
guess_limit = 3
out_of_guesses = False

while guess != secret_word and not(out_of_guesses):
    if guess_count < guess_limit:
        guess = input("Enter guess: ")
        guess_count += 1
    else:
        out_of_guesses = True

if out_of_guesses:
    print("Out of guesses, you lose!")
else:
    print("You win!") # printed out if the secret word is right

### For loop

for letter in "Giraffe Academy": # for every letter in "Giraffe Academy"
    print(letter)
    
####

friends = ["Jim", "Tom", "Kevin"]

for name in friends:
    print(name)

####

for index in range(10):
    print(index)
    
# output: values from 0 to 9 (excluding 10!)

####

for index in range(3, 10):
    print(index)

# output: values from 3 to 9 (excluding 10!)

####

friends = ["Jim", "Tom", "Kevin"]

for index in range(len(friends)):
   print(friends[index])

#output: Jim, Tom, Kevin

### Exponent function

def raise_to_power(base_num, pow_num):
    result = 1                          # Storing an actual result from for loop
    for index in range(pow_num):
        result = result * base_num
    return result

print(raise_to_power(2,3))

##### Modified :)

base_num = int(input("Enter the base: "))
pow_num = int(input("Enter the power: "))

def raise_to_power(base_num, pow_num):
    result = 1                          # Storing an actual result from for loop
    for index in range(pow_num):
        result = result * base_num
    return result

print(raise_to_power(base_num,pow_num))

### 2D lists and nested loops

number_grid = [         # 4 rows and 3 columns (lists within a list)
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [0]
]

print(number_grid[2][1])

####

number_grid = [         # 4 rows and 3 columns (lists within a list)
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [0]
]

for row in number_grid:
    print(row)

# output: number grid

####

number_grid = [         # 4 rows and 3 columns (lists within a list)
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [0]
]

for row in number_grid:
    for col in row:
        print(col)
        
# output: 1, 2, 3, ..., 0

### Translator

# Giraffe language ;) vowel -> g

def translate(phrase):
    translation = ""
    for letter in phrase:
        if letter in "AEIOUaeiou": # letter.lower() in "aeiou"
            translation = translation + "g"
        else:
            translation = translation + letter
    return translation

print(translate(input("Enter a phrase: ")))

####

def translate(phrase):
    translation = ""
    for letter in phrase:
        if letter in "AEIOUaeiou": # letter.lower() in "aeiou"
            if letter.isupper():
                translation = translation + "G"
            else:
                translation = translation + "g"
        else:
            translation = translation + letter
    return translation

print(translate(input("Enter a phrase: ")))

### Try & Except 

try:
    value = 10 / 0
    number = int(input("Enter a number: "))
    print(number)
except ZeroDivisionError:
    print("Divided by zero")
except ValueError:
    print("Invalid input")
    
####

try:
    value = 10 / 0
    number = int(input("Enter a number: "))
    print(number)
except ZeroDivisionError as err:
    print(err)
except ValueError:
    print("Invalid input")
    
# Output: division by zero

### Classes and objects

# Creating a class / object in a separate file (Student.py)

class student:

    def __init__(self, name, major, gpa, is_on_probation): # parameters that will describe a student
        self.name = name                                    # defining the parameters / assigning the information to a parameter
        self.major = major
        self.gpa = gpa
        self.is_on_probabtion = is_on_probation

# Calling the student class in another file (Test.py)

from Student import student

student1 = student("Jim", "Business", 3.1, False)

print(student1.name)


### Multiple Choice Quiz

from Question import question  # Question = Question.py (see below)

question_prompts = [
    "What color are apples?\n(a) Red/Green\n(b) Purple\n(c) Orange\n\n",
    "What color are bananas?\n(a) Teal\n(b) Magenta\n(c) Yellow\n\n",
    "What color are strawberries?\n(a) Yellow\n(b) Red\n(c) Blue\n\n"
]

questions = [
    question(question_prompts[0], "a"),
    question(question_prompts[1], "c"),
    question(question_prompts[2], "b"),
]

def run_test(questions):
    score = 0
    for question in questions:
        answer = input(question.prompt)
        if answer == question.answer:
            score+= 1
    print("You got " + str(score) + "/" + str(len(questions)) + " correct.")

run_test(questions)


# Question.py

class question:
    def __init__(self, prompt, answer):
        self.prompt = prompt
        self.answer = answer

### Object functions

# Students.py

class student:

    def __init__(self, name, major, gpa): # parameters that will describe a student
        self.name = name                                    # defining the parameters / assigning the information to a parameter
        self.major = major
        self.gpa = gpa
    def on_honor_roll(self):
        if self.gpa >= 3.5:
            return True
        else:
            return False

# Test.py

from Student import student

student1 = student("Jim", "Business", 3.1)
student2 = student("Ana", "Arts", 3.7)

print(student2.on_honor_roll()) # Output: True

### Inheritance

# Chef.py

class chef:

    def make_chicken(self):
        print("The chef makes a chicken")

    def make_salad(self):
        print("The chef makes a salad")

    def make_special_dish(self):
        print("The chef makes bbq ribs")

# ChineseChef.py

from Chef import chef

class chineseChef(chef): # Inheriting an existance class (chef class)

    def make_special_dish(self):    # Overwriting the function from chef class
        print("The chef makes an orange chicken")

    def make_fried_rice(self):
        print("The chef makes fried rice")

# Test.py where we use the classes

from Chef import chef
from ChineseChef import chineseChef

myChef = chef()
myChef.make_special_dish() # output: The chef makes bbq ribs

myChineseChef = chineseChef()
myChineseChef.make_special_dish() # output: The chef makes fried rice

