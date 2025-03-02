'''perl
#!/usr/bin/env perl
#
# perl.pl

# 'print' keyword outputs one or more strings
print "Hello, world!\n";  # Output: Hello, world!

# 'if' keyword starts a conditional statement
my $age = 30;
if ($age > 18) {
  print "Adult\n";  # Output: Adult
}

# 'else' keyword starts a block of code to be executed if the same condition is false
else {
  print "Not an adult\n";
}

# 'elsif' keyword specifies a new condition to check if the first condition is false
elsif ($age < 18) {
  print "Child\n";
}

# 'for' keyword creates a for loop
for (my $i = 0; $i < 5; $i++) {
  print $i . "\n";  # Output: 0 1 2 3 4
}

# 'foreach' keyword used to loop through arrays
my @arr = (1, 2, 3, 4, 5);
foreach my $value (@arr) {
  print $value . "\n";  # Output: 1 2 3 4 5
}

# 'while' keyword creates a while loop
my $i = 0;
while ($i < 5) {
  print $i . "\n";  # Output: 0 1 2 3 4
  $i++;
}

# 'do...while' keyword creates a do/while loop
$i = 0;
do {
  print $i . "\n";  # Output: 0 1 2 3 4 5
  $i++;
} while ($i < 5);

# 'until' keyword creates an until loop
$i = 0;
until ($i > 5) {
  print $i . "\n";  # Output: 0 1 2 3 4 5
  $i++;
}

# 'sub' keyword defines a subroutine (function)
sub greet {
  print "Hello, world!\n";  # Output: Hello, world!
}
greet();

# 'return' keyword returns a value from a subroutine
sub square {
  my ($num) = @_;
  return $num * $num;
}
print square(4) . "\n";  # Output: 16

# 'package' keyword defines a package (module)
package MyPackage;
sub sayHello {
  print "Hello, world!\n";  # Output: Hello, world!
}
MyPackage::sayHello();

# 'use' keyword imports modules
use strict;
use warnings;

# 'die' keyword outputs a message and terminates the current script
# die("The script ended.");

# 'exit' keyword terminates the current script
# exit("The script ended.");

