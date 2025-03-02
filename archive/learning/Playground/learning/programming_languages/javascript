'''javascript
// Variables
var name = "John Doe";
var age = 30;

// Data types
var isStudent = false; // Boolean
var height = null; // Null
var weight; // Undefined

// Array
var hobbies = ["Reading", "Coding", "Music"];

// Adding to an array
hobbies.push("Running");

// Removing from an array
var lastHobby = hobbies.pop();

// Object (similar to dictionary in Python)
var person = {
    firstName: "John",
    lastName: "Doe",
    age: 30,
    hobbies: hobbies,
    greet: function() {
        return "Hello, " + this.firstName + " " + this.lastName + "!";
    }
};

// Accessing object properties
console.log(person.firstName); // Output: John
console.log(person['lastName']); // Output: Doe

// Function
function greet(person) {
    return "Hello, " + person + "!";
}

console.log(greet(name)); // Output: Hello, John Doe!

// Loop (for)
for (var i = 0; i < hobbies.length; i++) {
    console.log(hobbies[i]);
}

// Loop (while)
var i = 0;
while (i < hobbies.length) {
    console.log(hobbies[i]);
    i++;
}

// Conditional (if, else if, else)
if (age < 13) {
    console.log(name + " is a child.");
} else if (age < 20) {
    console.log(name + " is a teenager.");
} else {
    console.log(name + " is an adult.");
}

// Switch statement
switch (age) {
    case 10:
        console.log(name + " is 10 years old.");
        break;
    case 20:
        console.log(name + " is 20 years old.");
        break;
    default:
        console.log("Age of " + name + " is " + age);
}

// 'var' keyword declares a variable
var name = "John Doe";

// 'let' keyword declares a block scope local variable
let age = 30;

// 'const' keyword declares a block scope constant variable
const PI = 3.14159;

// 'if' keyword starts a conditional statement
if (age > 18) {
  console.log(name + " is an adult.");
}

// 'else' keyword starts a block of code to be executed if the same condition is false
else {
  console.log(name + " is not an adult.");
}

// 'for' keyword creates a for loop
for (let i = 0; i < 5; i++) {
  console.log(i);
}

// 'while' keyword creates a while loop
let i = 0;
while (i < 5) {
  console.log(i);
  i++;
}

// 'do' keyword is used together with while to create a do/while loop
i = 0;
do {
  console.log(i);
  i++;
} while (i < 5);

// 'switch' keyword selects one of many code blocks to be executed
switch (age) {
  case 18:
    console.log(name + " is 18 years old.");
    break;
  default:
    console.log(name + " is " + age + " years old.");
}

// 'try' keyword tests a block of code for errors
try {
  nonExistentFunction();
}

// 'catch' keyword handles the error
catch(error) {
  console.log("An error occurred: " + error.message);
}

// 'finally' keyword executes code after try and catch, regardless of the result
finally {
  console.log("Try-catch block finished.");
}

// 'throw' keyword creates custom errors
throw "This is an error!";

// 'break' keyword breaks out of a loop, or a switch
for (let i = 0; i < 5; i++) {
  if (i === 3) {
    break;
  }
  console.log(i);
}

// 'continue' keyword breaks one iteration in the loop, if a specified condition occurs, and continues with the next iteration
for (let i = 0; i < 5; i++) {
  if (i === 3) {
    continue;
  }
  console.log(i);
}

// 'function' keyword declares a function
function greet() {
  console.log("Hello, world!");
}
greet();

// 'return' keyword stops the execution of a function and returns a value from that function
function square(number) {
  return number * number;
}
console.log(square(5));

// 'class' keyword creates a class
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }
  greet() {
    console.log("Hello, my name is " + this.name + " and I'm " + this.age + " years old.");
  }
}
const person = new Person("John Doe", 30);
person.greet();

// 'extends' keyword creates a class as a child of another class
class Employee extends Person {
  constructor(name, age, jobTitle) {
    super(name, age);
    this.jobTitle = jobTitle;
  }
  greet() {
    console.log("Hello, my name is " + this.name + ", I'm " + this.age + " years old, and I work as a " + this.jobTitle + ".");
  }
}
const employee = new Employee("John Doe", 30, "web developer");
employee.greet();

// 'super' keyword calls the constructor of the parent class
// (see 'extends' keyword example above)

// 'this' keyword refers to the object that the function is a method of
// (see 'class' keyword example above)

// 'null' keyword represents a null value
let emptyVariable = null;
console.log(emptyVariable);

// 'undefined' keyword represents an undefined value
let undefinedVariable;
console.log(undefinedVariable);

// 'true' keyword represents a true value
let isTrue = true;
console.log(isTrue);

// 'false' keyword represents a false value
let isFalse = false;
console.log(isFalse);

// 'new' keyword creates a new object instance
let date = new Date();
console.log(date);

// 'delete' keyword deletes a property from an object
let myObject = {name: "John Doe", age: 30};
delete myObject.age;
console.log(myObject);

// 'typeof' keyword returns the type of a variable
console.log(typeof name);

// 'instanceof' keyword returns true if an object is an instance of an object type
console.log(person instanceof Person);

// 'in' keyword returns true if the specified property is in the specified object
console.log("name" in myObject);

// 'void' keyword is an operator that discards an expression's return value
void console.log("Hello, world!");
