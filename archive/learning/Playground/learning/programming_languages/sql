'''sql
-- 'CREATE' keyword creates a new table
CREATE TABLE Employees (
  ID INT PRIMARY KEY,  -- 'INT' keyword specifies an integer data type, 'PRIMARY KEY' keyword sets the ID column as the primary key
  Name VARCHAR(255),  -- 'VARCHAR' keyword specifies a variable length string
  Age INT,
  Salary DECIMAL(10, 2)  -- 'DECIMAL' keyword specifies a decimal number
);

-- 'INSERT INTO' keyword inserts new data into a table
INSERT INTO Employees (ID, Name, Age, Salary)
VALUES (1, 'John Doe', 30, 50000.00);

-- 'SELECT' keyword selects data from a database
SELECT * FROM Employees;  -- '*' keyword selects all columns

-- 'WHERE' keyword filters the results
SELECT * FROM Employees WHERE Age > 25;

-- 'UPDATE' keyword updates data in a database
UPDATE Employees SET Salary = 60000.00 WHERE ID = 1;

-- 'DELETE' keyword deletes data from a database
DELETE FROM Employees WHERE ID = 1;

-- 'ALTER TABLE' keyword is used to add, delete/drop or modify columns in an existing table
ALTER TABLE Employees ADD Email VARCHAR(255);

-- 'DROP' keyword is used to delete a table or a database
-- DROP TABLE Employees;

-- 'JOIN' keyword is used to combine rows from two or more tables, based on a related column between them
-- SELECT Orders.OrderID, Customers.CustomerName
-- FROM Orders
-- JOIN Customers
-- ON Orders.CustomerID = Customers.CustomerID;

-- 'UNION' keyword is used to combine the result-set of two or more SELECT statements
-- SELECT column_name(s) FROM table1
-- UNION
-- SELECT column_name(s) FROM table2;

-- 'GROUP BY' keyword groups the result-set by one or more columns
-- SELECT COUNT(CustomerID), Country
-- FROM Customers
-- GROUP BY Country;

-- 'ORDER BY' keyword is used to sort the result-set in ascending or descending order
-- SELECT * FROM Customers
-- ORDER BY Country;

-- 'LIMIT' keyword is used to specify the number of records to return
-- SELECT * FROM Customers
-- LIMIT 3;
