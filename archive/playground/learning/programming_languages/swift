'''swift
import Foundation  // 'import' keyword is used to import a module

// 'var' keyword declares a mutable variable
var str = "Hello, playground"
print(str)  // Output: Hello, playground

// 'let' keyword declares an immutable variable
let constantStr = "Cannot change me"
print(constantStr)  // Output: Cannot change me

// 'if-else' keywords for conditional statements
var age = 20
if age > 18 {
    print("Adult")  // Output: Adult
} else {
    print("Not an adult")
}

// 'for-in' keywords for looping over a range or collection
for i in 1...5 {
    print(i)  // Output: 1 2 3 4 5
}

// Arrays
var array = ["Apple", "Banana", "Mango"]
print(array[1])  // Output: Banana

// Dictionaries
var dict = ["name": "John", "age": "30"]
print(dict["name"]!)  // Output: John

// Functions
func greet(name: String) {
    print("Hello, \(name)")  // String interpolation
}
greet(name: "John")  // Output: Hello, John

// Classes
class Person {
    var name: String
    init(name: String) {  // Initializer
        self.name = name
    }
    func greet() {
        print("Hello, \(self.name)")
    }
}

var person = Person(name: "John")
person.greet()  // Output: Hello, John

// Enums
enum Weekday {
    case monday, tuesday, wednesday, thursday, friday, saturday, sunday
}

var day = Weekday.monday
switch day {
case .monday:
    print("It's Monday")  // Output: It's Monday
default:
    print("It's another day")
}

// Optionals and optional binding
var optionalVar: String?
optionalVar = "Hello"
if let unwrapped = optionalVar {
    print(unwrapped)  // Output: Hello
}

// Error handling with do-try-catch
enum SimpleError: Error {
    case someError
}

func mightThrowError() throws {
    throw SimpleError.someError
}

do {
    try mightThrowError()
} catch {
    print("Caught an error")  // Output: Caught an error
}
