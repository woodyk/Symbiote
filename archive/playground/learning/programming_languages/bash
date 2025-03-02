#!/bin/bash  # Shebang to specify the interpreter for script execution

# 'echo' command outputs a string
echo "Hello, world!"  # Output: Hello, world!

# Variables in bash
name="John Doe"
echo $name  # Output: John Doe

# 'read' command reads input from user
# echo "Enter your name:"
# read name
# echo "Hello, $name!"

# 'if' keyword starts a conditional statement
age=30
if [ $age -gt 18 ]  # '-gt' is a comparison operator for 'greater than'
then
  echo "Adult"  # Output: Adult
fi

# 'else' keyword starts a block of code to be executed if the same condition is false
age=15
if [ $age -gt 18 ]
then
  echo "Adult"
else
  echo "Not an adult"  # Output: Not an adult
fi

# 'for' keyword creates a for loop
for i in {1..5}
do
  echo $i  # Output: 1 2 3 4 5
done

# 'while' keyword creates a while loop
i=1
while [ $i -le 5 ]  # '-le' is a comparison operator for 'less than or equal to'
do
  echo $i  # Output: 1 2 3 4 5
  ((i++))  # increment operator
done

# Functions in bash
function greet {
  echo "Hello, world!"
}
greet  # Output: Hello, world!

# 'case' keyword creates a case statement (similar to switch in other languages)
color="blue"
case $color in
  "red") echo "Color is red" ;;
  "blue") echo "Color is blue" ;;  # Output: Color is blue
  *) echo "Color is neither red nor blue" ;;
esac

# 'exit' command exits the shell or your script
# exit 0
