'''ruby
# Variables
name = "John Doe"
age = 30

# Data types
is_student = false # Boolean
height = nil # Nil

# Array
hobbies = ["Reading", "Coding", "Music"]

# Adding to an array
hobbies.push("Running")

# Removing from an array
last_hobby = hobbies.pop

# Hash (similar to dictionary in Python)
person = {
  "firstName" => "John",
  "lastName" => "Doe",
  "age" => 30,
  "hobbies" => hobbies
}

# Accessing hash values
puts person["firstName"] # Output: John
puts person["lastName"] # Output: Doe

# Function
def greet(person)
  return "Hello, " + person + "!"
end

puts greet(name) # Output: Hello, John Doe!

# Loop (for)
for hobby in hobbies
  puts hobby
end

# Loop (while)
i = 0
while i < hobbies.length
  puts hobbies[i]
  i += 1
end

# Conditional (if, elsif, else)
if age < 13
  puts name + " is a child."
elsif age < 20
  puts name + " is a teenager."
else
  puts name + " is an adult."
end

# Case statement
case age
when 10
  puts name + " is 10 years old."
when 20
  puts name + " is 20 years old."
else
  puts "Age of " + name + " is " + age.to_s
end

# 'BEGIN' keyword runs before any other code in the current file
BEGIN {
  puts "This is the beginning."
}

# 'END' keyword runs after any other code in the current file
END {
  puts "This is the end."
}

# 'alias' keyword creates an alias between two methods (or two global variables)
def hello
  puts "Hello, world!"
end
alias greeting hello
greeting()

# 'and' keyword is a logical operator
puts true and false

# 'begin' keyword starts an exception handling block
begin
  puts 10 / 0
rescue
  puts "An error occurred"
end

# 'break' keyword breaks out of a loop
i = 0
while i < 10
  if i == 5
    break
  end
  puts i
  i += 1
end

# 'case' keyword starts a case expression
case 5
when 1
  puts "It's 1"
when 2
  puts "It's 2"
else
  puts "It's something else"
end

# 'class' keyword defines a class
class MyClass
  def my_method
    puts "Hello, world!"
  end
end
MyClass.new.my_method

# 'def' keyword defines a method
def my_method
  puts "Hello, world!"
end
my_method

# 'defined?' keyword checks if a variable, method, super method, or block exists
puts defined? my_method

# 'do' keyword starts a block
3.times do |i|
  puts i
end

# 'else' keyword specifies a block of code to be executed if a condition is false
if false
  puts "This won't be printed"
else
  puts "This will be printed"
end

# 'elsif' keyword specifies a new condition if the previous condition was false
if false
  puts "This won't be printed"
elsif true
  puts "This will be printed"
end

# 'end' keyword ends a syntax block
if true
  puts "Hello, world!"
end

# 'ensure' keyword starts a block of code that will always run in a begin/end block
begin
  puts 10 / 0
rescue
  puts "An error occurred"
ensure
  puts "This will always run"
end

# 'false' keyword represents a false value
puts false

# 'for' keyword starts a for loop
for i in 0..5
  puts i
end

# 'if' keyword starts an if expression
if true
  puts "Hello, world!"
end

# 'in' keyword is used in for loops
for i in 0..5
  puts i
end

# 'module' keyword defines a module
module MyModule
  def my_method
    puts "Hello, world!"
  end
end
include MyModule
my_method

# 'next' keyword skips to the next iteration of a loop
for i in 0..5
  next if i == 2
  puts i
end

# 'nil' keyword represents a nil value
puts nil

# 'not' keyword is a logical operator
puts not true

# 'or' keyword is a logical operator
puts true or false

# 'redo' keyword restarts the current iteration of a loop
for i in 0..5
  puts i
  i = 2 if i == 2
  redo if i == 2
end

# 'rescue' keyword starts a block of code that will run if an exception is raised
begin
  puts 10 / 0
rescue
  puts "An error occurred"
end

# 'retry' keyword retries a block of code if an exception is raised
begin
  puts "Trying to divide by zero..."
  puts 10 / 0
rescue
  puts "An error occurred. Let's try that again..."
  retry
end

# 'return' keyword exits a method
def my_method
  return
  puts "This won't be printed"
end
my_method

# 'self' keyword refers to the current object
puts self

# 'super' keyword calls a method of the same name in the superclass
class Parent
  def my_method
    puts "Hello, world!"
  end
end
class Child < Parent
  def my_method
    super
  end
end
Child.new.my_method

# 'then' keyword is used with if unless and when
if true then puts "Hello, world!" end

# 'true' keyword represents a true value
puts true

# 'undef' keyword undefines a method
undef my_method

# 'unless' keyword starts an unless expression
unless false
  puts "Hello, world!"
end

# 'until' keyword starts an until loop
i = 0
until i > 5
  puts i
  i += 1
end

# 'when' keyword specifies a condition in a case expression
case 5
when 1
  puts "It's 1"
when 2
  puts "It's 2"
else
  puts "It's something else"
end

# 'while' keyword starts a while loop
i = 0
while i < 5
  puts i
  i += 1
end

# 'yield' keyword runs a block passed to a method
def my_method
  yield
end
my_method { puts "Hello, world!" }
