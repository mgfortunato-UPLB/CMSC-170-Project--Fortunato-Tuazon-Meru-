import csv
import os
import matplotlib.pyplot as plt

class Binary_Tree_Node:
    def __init__(self, array_of_data):
        self.left_child = None
        self.right_child = None
        self.expected_val = None
        self.bound = None

        if len(array_of_data) > 12:
            for i in range(len(array_of_data)):
                if i == 0:
                    continue

                left_child = array_of_data[:i]
                right_child = array_of_data[i:]

                average_left = sum(left_child) / len(left_child)
                average_right = sum(right_child) / len(right_child)

                left_squares = []
                right_squares = []

                for j in left_child:
                    left_squares.append((j-average_left)**2)
                for j in right_child:
                    right_squares.append((j-average_right)**2)

                ssr = sum(left_squares) + sum(right_squares)

                if (self.bound == None) or self.bound[1] >= ssr:
                    self.bound = (i,ssr)

            self.left_child = Binary_Tree_Node(array_of_data[:self.bound[0]])
            self.right_child = Binary_Tree_Node(array_of_data[self.bound[0]:])

        else:
            self.expected_val = sum(array_of_data) / len(array_of_data)
        


def reg_tree(path, time):
    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader) # skip header

        y_values = []

        for row in reader:
            if row[1].startswith("Quarter"):
                y_values.append(float(row[2]))
                plt.plot(float(row[0]),float(row[2]))

        regression_tree = Binary_Tree_Node(y_values)

        while regression_tree.expected_val == None:
            if time < y_values[regression_tree.bound[0]]:
                regression_tree = regression_tree.left_child
            else:
                regression_tree = regression_tree.right_child

    return regression_tree.expected_val

        # for i in range(len(y_values) - 1):
        #     threshold = i+1
        #     average_left = 0
        #     average_right = 0
        #     for i in range(len(y_values) - 1):

time = 1989.3475394579

print(reg_tree("processed/Philippines/Corn.csv", (1989.3475394579 - 1987) * 4))
