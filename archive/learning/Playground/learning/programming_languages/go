'''go
package main  // 'package' keyword defines a package name

import "fmt"  // 'import' keyword is used to import a package

func main() {  // 'func' keyword is used to declare a function

  // 'fmt.Println' function outputs a string
  fmt.Println("Hello, world!")  // Output: Hello, world!

  // 'if' keyword starts a conditional statement
  age := 30  // ':=' keyword is a shorthand for declaring and initializing a variable
  if age > 18 {
    fmt.Println("Adult")  // Output: Adult
  }

  // 'else' keyword starts a block of code to be executed if the same condition is false
  else {
    fmt.Println("Not an adult")
  }

  // 'for' keyword creates a for loop
  for i := 0; i < 5; i++ {
    fmt.Println(i)  // Output: 0 1 2 3 4
  }

  // 'switch' keyword selects one of many blocks of code to be executed
  color := "blue"
  switch color {
    case "red":
      fmt.Println("Color is red")
    case "blue":
      fmt.Println("Color is blue")  // Output: Color is blue
    default:
      fmt.Println("Color is neither red nor blue")
  }

  // 'defer' keyword schedules a function call to be run after the function completes
  defer fmt.Println("World")  // Output: World
  fmt.Println("Hello")  // Output: Hello

  // 'go' keyword starts a goroutine
  go fmt.Println("Hello, world!")  // Output: Hello, world!

  // 'return' keyword stops the execution of a function and returns a value
  // This is not used in the 'main' function because it's void
}

