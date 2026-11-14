import csv
import os
import matplotlib.pyplot as plt
from eval_metrics import mean_absolute_error

class Binary_Tree_Node:
    def __init__(self, array_of_data, left_bound, right_bound):
        self.left_bound = left_bound
        # self.right_bound = right_bound
        self.left_child = None
        self.right_child = None
        self.expected_val = None
        self.bound = None

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
                if len(array_of_data[:i]) >= 6 and len(array_of_data[i:]) >= 6:
                    self.bound = (i,ssr)

        if self.bound == None:
            # print(str(((1987*4) + right_bound)/4 - ((1987*4) + left_bound)/4))
            self.expected_val = sum(array_of_data) / len(array_of_data)
            plt.plot([((1987*4) + left_bound)/4, ((1987*4) + right_bound)/4], [self.expected_val, self.expected_val], "b-")
        else:
            self.left_child = Binary_Tree_Node(array_of_data[:self.bound[0]], left_bound, left_bound + self.bound[0])
            self.right_child = Binary_Tree_Node(array_of_data[self.bound[0]:], left_bound + self.bound[0], right_bound)

def reg_tree(path, time):
    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader) # skip header

        x_values = []
        y_values = []

        i = 1
        for row in reader:
            if row[1].startswith("Quarter"):
                x_values.append(1987 + ((i/4)))
                y_values.append(float(row[2]))
                i+=1
        plt.plot(x_values, y_values, "r.")

        regression_tree = Binary_Tree_Node(y_values, 0, len(x_values) - 1)

        pred_values = []
        for y in y_values:
            temp_regression_tree = regression_tree
            while temp_regression_tree.expected_val == None:
                if y < y_values[temp_regression_tree.bound[0]]:
                    temp_regression_tree = temp_regression_tree.left_child
                else:
                    temp_regression_tree = temp_regression_tree.right_child
            pred_values.append(temp_regression_tree.expected_val)

        while regression_tree.expected_val == None:
            # print(x_values[regression_tree.left_bound + regression_tree.bound[0]])
            if time < x_values[regression_tree.left_bound + regression_tree.bound[0]]:
                regression_tree = regression_tree.left_child
            else:
                regression_tree = regression_tree.right_child

    plt.show()
    return (regression_tree.expected_val, mean_absolute_error(y_values, pred_values))

time = 2010.3475394579
answer = reg_tree("processed/Philippines/Corn.csv", time)

print("The expected value of the crop's yield at year " + str(time) + " is " + str(answer[0]) + " with mean error " + str(answer[1]))
