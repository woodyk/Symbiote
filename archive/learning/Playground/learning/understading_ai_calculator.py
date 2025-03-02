#!/usr/bin/env python3
#
# understading_ai_calculator.py

import numpy as np
import random
import math

def print_matrix(matrix):
    for row in matrix:
        print(row)

def fine_tune_model(model, dataset, targets, learning_rate=0.01, epochs=100):
    for epoch in range(epochs):
        for inputs, target in zip(dataset, targets):
            # Forward pass
            output = model.forward(inputs)

            # Compute the error
            error = target - output

            # Backward pass
            model.backward(error)

            # Update weights and biases
            model.update_weights(learning_rate)

        # Print the error every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Error: {error}")

def add_vectors(random_choice):
    if random_choice:
        vector1 = [random.randint(1, 10) for _ in range(3)]
        vector2 = [random.randint(1, 10) for _ in range(3)]
    else:
        vector1 = input("Enter the first vector, comma-separated: ")
        vector2 = input("Enter the second vector, comma-separated: ")

        vector1 = [float(i) for i in vector1.split(',')]
        vector2 = [float(i) for i in vector2.split(',')]

    result = [v1 + v2 for v1, v2 in zip(vector1, vector2)]

    print(f"The result of the vector addition is {result}")
    print("The equation is:")
    print(vector1, "+", vector2)
    print("=", result)

def scalar_vector_multiplication(random_choice):
    if random_choice:
        scalar = random.randint(1, 10)
        vector = [random.randint(1, 10) for _ in range(3)]
    else:
        scalar = float(input("Enter the scalar: "))
        vector = input("Enter the vector, comma-separated: ")

        vector = [float(i) for i in vector.split(',')]

    result = [scalar * v for v in vector]

    print(f"The result of the scalar-vector multiplication is {result}")
    print("The equation is:")
    print(scalar, "*", vector)
    print("=", result)

def dot_product(random_choice):
    if random_choice:
        vector1 = [random.randint(1, 10) for _ in range(3)]
        vector2 = [random.randint(1, 10) for _ in range(3)]
    else:
        vector1 = input("Enter the first vector, comma-separated: ")
        vector2 = input("Enter the second vector, comma-separated: ")

        vector1 = [float(i) for i in vector1.split(',')]
        vector2 = [float(i) for i in vector2.split(',')]

    result = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))

    print(f"The result of the dot product is {result}")
    print("The equation is:")
    print(vector1, ".", vector2)
    print("=", result)

def multiply_matrices(random_choice):
    if random_choice:
        matrix1 = [[random.randint(1, 10) for _ in range(3)] for _ in range(3)]
        matrix2 = [[random.randint(1, 10) for _ in range(3)] for _ in range(3)]
    else:
        matrix1 = input("Enter the first matrix, rows separated by semicolons (;), elements within rows separated by commas (,): ")
        matrix2 = input("Enter the second matrix, rows separated by semicolons (;), elements within rows separated by commas (,): ")

        matrix1 = [[float(j) for j in i.split(',')] for i in matrix1.split(';')]
        matrix2 = [[float(j) for j in i.split(',')] for i in matrix2.split(';')]

    result = [[sum(a*b for a, b in zip(row, col)) for col in zip(*matrix2)] for row in matrix1]

    print(f"The result of the matrix multiplication is:\n{result}")
    print("The equation is:")
    print_matrix(matrix1)
    print("*")
    print_matrix(matrix2)
    print("=")
    print_matrix(result)

def subtract_matrices(random_choice):
    if random_choice:
        matrix1 = [[random.randint(1, 10) for _ in range(3)] for _ in range(3)]
        matrix2 = [[random.randint(1, 10) for _ in range(3)] for _ in range(3)]
    else:
        matrix1 = input("Enter the first matrix, rows separated by semicolons (;), elements within rows separated by commas (,): ")
        matrix2 = input("Enter the second matrix, rows separated by semicolons (;), elements within rows separated by commas (,): ")

        matrix1 = [[float(j) for j in i.split(',')] for i in matrix1.split(';')]
        matrix2 = [[float(j) for j in i.split(',')] for i in matrix2.split(';')]

    result = [[m1 - m2 for m1, m2 in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]

    print(f"The result of the matrix subtraction is:\n{result}")
    print("The equation is:")
    print_matrix(matrix1)
    print("-")
    print_matrix(matrix2)
    print("=")
    print_matrix(result)

def sigmoid(random_choice):
    if random_choice:
        x = random.uniform(-10, 10)
    else:
        x = float(input("Enter a number: "))

    sigmoid = 1 / (1 + math.exp(-x))

    print(f"The sigmoid of {x} is {sigmoid}")

    return sigmoid

def solve_linear(random_choice):
    if random_choice:
        A = [[random.randint(1, 10) for _ in range(3)] for _ in range(3)]
        B = [random.randint(1, 10) for _ in range(3)]
    else:
        A = input("Enter the matrix A, rows separated by semicolons (;), elements within rows separated by commas (,): ")
        B = input("Enter the matrix B, comma-separated: ")

        A = [[float(j) for j in i.split(',')] for i in A.split(';')]
        B = [float(i) for i in B.split(',')]

    X = np.linalg.solve(A, B)

    print(f"The solution of the linear equations is {X}")
    print("The equation is:")
    print_matrix(A)
    print("* X =")
    print(B)

def tanh(random_choice):
    if random_choice:
        x = random.uniform(-10, 10)
    else:
        x = float(input("Enter a number: "))

    tanh = 2 / (1 + math.exp(-2*x)) - 1

    print(f"The tanh of {x} is {tanh}")

    return tanh

def relu(random_choice):
    if random_choice:
        x = random.uniform(-10, 10)
    else:
        x = float(input("Enter a number: "))

    relu = max(0, x)

    print(f"The ReLU of {x} is {relu}")

    return relu

def relu6(random_choice):
    if random_choice:
        x = random.uniform(-10, 10)
    else:
        x = float(input("Enter a number: "))

    relu6 = min(max(0, x), 6)

    print(f"The ReLU6 of {x} is {relu6}")

    return relu6

def leaky_relu(random_choice):
    if random_choice:
        x = random.uniform(-10, 10)
    else:
        x = float(input("Enter a number: "))

    leaky_relu = max(0.01*x, x)

    print(f"The Leaky ReLU of {x} is {leaky_relu}")

    return leaky_relu

def elu(random_choice):
    if random_choice:
        x = random.uniform(-10, 10)
    else:
        x = float(input("Enter a number: "))

    elu = x if x > 0 else 0.01*(math.exp(x) - 1)

    print(f"The ELU of {x} is {elu}")

    return elu

def softplus(random_choice):
    if random_choice:
        x = random.uniform(-10, 10)
    else:
        x = float(input("Enter a number: "))

    softplus = math.log(1 + math.exp(x))

    print(f"The Softplus of {x} is {softplus}")

    return softplus

def softmax(random_choice):
    if random_choice:
        x = [random.uniform(-10, 10) for _ in range(3)]
    else:
        x = input("Enter a list of numbers, comma-separated: ")
        x = [float(i) for i in x.split(',')]

    softmax = np.exp(x) / np.sum(np.exp(x))

    print(f"The Softmax of {x} is {softmax}")

    return softmax

def create_model():
    num_layers = int(input("Enter the number of layers: "))
    layers = []
    for i in range(num_layers):
        neurons = int(input(f"Enter the number of neurons in layer {i+1}: "))
        activation = input(f"Enter the activation function for layer {i+1} (sigmoid, relu, tanh, etc.): ")
        layers.append((neurons, activation))

    print("\nYour model:")
    for i, (neurons, activation) in enumerate(layers):
        print("\nLayer", i+1, "(Activation function:", activation + ")")
        print("|" + "---" * neurons + "|")
        print("|" + " o " * neurons + "|")
        print("|" + "---" * neurons + "|")

def create_dataset():
    num_samples = int(input("Enter the number of samples in the dataset: "))
    dataset = []
    targets = []

    for _ in range(num_samples):
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)

        dataset.append((a, b))
        targets.append(a + b)

    print("\nYour dataset:")
    for (a, b), target in zip(dataset, targets):
        print(f"Inputs: ({a}, {b}), Target: {target}")

    return dataset, targets

def main():
    random_choice = True
    while True:
        print("\n1. Add two vectors")
        print("2. Scalar-vector multiplication")
        print("3. Calculate dot product of two vectors")
        print("4. Multiply two matrices")
        print("5. Subtract two matrices")
        print("6. Calculate sigmoid")
        print("7. Solve linear equations")
        print("8. Calculate tanh")
        print("9. Calculate ReLU")
        print("10. Calculate ReLU6")
        print("11. Calculate Leaky ReLU")
        print("12. Calculate ELU")
        print("13. Calculate Softplus")
        print("14. Calculate Softmax")
        print("15. Create AI model")
        print("16. Create dataset")
        print("17. Exit")
        print("18. Enable random entries")

        choice = int(input("\nChoose an operation: "))

        if choice == 1:
            add_vectors(random_choice)
        elif choice == 2:
            scalar_vector_multiplication(random_choice)
        elif choice == 3:
            dot_product(random_choice)
        elif choice == 4:
            multiply_matrices(random_choice)
        elif choice == 5:
            subtract_matrices(random_choice)
        elif choice == 6:
            sigmoid(random_choice)
        elif choice == 7:
            solve_linear(random_choice)
        elif choice == 8:
            tanh(random_choice)
        elif choice == 9:
            relu(random_choice)
        elif choice == 10:
            relu6(random_choice)
        elif choice == 11:
            leaky_relu(random_choice)
        elif choice == 12:
            elu(random_choice)
        elif choice == 13:
            softplus(random_choice)
        elif choice == 14:
            softmax(random_choice)
        elif choice == 15:
            create_model()
        elif choice == 16:
            create_dataset()
        elif choice == 17:
            break
        elif choice == 18:
            random_choice = not random_choice
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()

