'''php
<?php
// 'echo' keyword outputs one or more strings
echo "Hello, world!";

// 'print' keyword outputs a string
print "Hello, world!";

// 'if' keyword starts a conditional statement
$age = 30;
if ($age > 18) {
  echo "Adult";
}

// 'else' keyword starts a block of code to be executed if the same condition is false
else {
  echo "Not an adult";
}

// 'elseif' keyword specifies a new condition to check if the first condition is false
elseif ($age < 18) {
  echo "Child";
}

// 'for' keyword creates a for loop
for ($i = 0; $i < 5; $i++) {
  echo $i;
}

// 'foreach' keyword used to loop through arrays
$arr = array(1, 2, 3, 4, 5);
foreach ($arr as $value) {
  echo $value;
}

// 'while' keyword creates a while loop
$i = 0;
while ($i < 5) {
  echo $i;
  $i++;
}

// 'do...while' keyword creates a do/while loop
$i = 0;
do {
  echo $i;
  $i++;
} while ($i < 5);

// 'switch' keyword selects one of many blocks of code to be executed
$color = "red";
switch ($color) {
  case "red":
    echo "Color is red";
    break;
  case "blue":
    echo "Color is blue";
    break;
  default:
    echo "Color is neither red nor blue";
}

// 'break' keyword breaks out of a loop, or a switch
for ($i = 0; $i < 5; $i++) {
  if ($i == 3) {
    break;
  }
  echo $i;
}

// 'continue' keyword breaks one iteration in the loop, if a specified condition occurs, and continues with the next iteration
for ($i = 0; $i < 5; $i++) {
  if ($i == 3) {
    continue;
  }
  echo $i;
}

// 'function' keyword defines a function
function greet() {
  echo "Hello, world!";
}
greet();

// 'return' keyword returns a value from a function
function square($num) {
  return $num * $num;
}
echo square(4);

// 'class' keyword defines a class
class MyClass {
  function myMethod() {
    echo "Hello, world!";
  }
}
$obj = new MyClass;
$obj->myMethod();

// 'new' keyword creates a new object
$obj = new MyClass;

// 'try...catch' keyword handles exceptions (errors)
try {
  echo 10 / 0;
} catch (Exception $e) {
  echo 'Caught exception: ',  $e->getMessage(), "\n";
}

// 'throw' keyword throws an exception
throw new Exception('This is an error!');

// 'use' keyword imports classes, and you can use it to create an alias
use MyClass as Alias;
$obj = new Alias;

// 'namespace' keyword defines a namespace
namespace MyNamespace;
class MyClass {
  static function sayHello() {
    echo "Hello, world!";
  }
}
MyNamespace\MyClass::sayHello();

// 'die' keyword outputs a message and terminates the current script
die("The script ended.");

// 'exit' keyword terminates the current script
exit("The script ended.");
?>
