import csv
import os
import matplotlib.pyplot as plt
from eval_metrics import mean_absolute_error

class Binary_Tree_Node:
    def __init__(self, array_of_data, left_bound, right_bound, x_values, plotter):
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
            # plt.plot([((1987*4) + left_bound)/4, ((1987*4) + right_bound)/4], [self.expected_val, self.expected_val], plotter)
            plt.plot([x_values[left_bound], x_values[right_bound]], [self.expected_val, self.expected_val], plotter)
        else:
            self.left_child = Binary_Tree_Node(array_of_data[:self.bound[0]], left_bound, left_bound + self.bound[0], x_values, plotter)
            self.right_child = Binary_Tree_Node(array_of_data[self.bound[0]:], left_bound + self.bound[0], right_bound, x_values, plotter)

def reg_tree(path, time):
    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader) # skip header

        x_values = []
        y_values = []

        i = 0
        for row in reader:
            if row[1].startswith("Annual"):
                # x_values.append(1987 + ((i/4)))
                x_values.append(1987 + i)
                y_values.append(float(row[2]))
                i+=1
        plt.plot(x_values, y_values, "r.")

        regression_tree = Binary_Tree_Node(y_values, 0, len(x_values) - 1, x_values, "b-")

        x_train = x_values[:(int(len(x_values) * 7/10))]
        x_test = x_values[(int(len(x_values) * 7/10)):]

        y_train = y_values[:(int(len(y_values) * 7/10))]
        y_test = y_values[(int(len(y_values) * 7/10)):]

        regression_train = Binary_Tree_Node(y_train, 0, len(x_train) - 1, x_train, "o--")
        regression_test = Binary_Tree_Node(y_test, 0, len(x_test) - 1, x_test, "g--")

        train_values = []
        for x in x_train:
            temp_regression_train = regression_train
            while temp_regression_train.expected_val == None:
                if x < x_train[temp_regression_train.left_bound + temp_regression_train.bound[0]]:
                    temp_regression_train = temp_regression_train.left_child
                else:
                    temp_regression_train = temp_regression_train.right_child
            train_values.append(temp_regression_train.expected_val)

        test_values = []
        for x in x_test:
            temp_regression_test = regression_test
            while temp_regression_test.expected_val == None:
                if x < x_test[temp_regression_test.left_bound + temp_regression_test.bound[0]]:
                    temp_regression_test = temp_regression_test.left_child
                else:
                    temp_regression_test = temp_regression_test.right_child
            test_values.append(temp_regression_test.expected_val)

        # pred_values = []
        # for x in x_values:
        #     temp_regression_tree = regression_tree
        #     while temp_regression_tree.expected_val == None:
        #         if x < x_values[temp_regression_tree.left_bound + temp_regression_tree.bound[0]]:
        #             temp_regression_tree = temp_regression_tree.left_child
        #         else:
        #             temp_regression_tree = temp_regression_tree.right_child
        #     pred_values.append(temp_regression_tree.expected_val)

        while regression_tree.expected_val == None:
            # print(x_values[regression_tree.left_bound + regression_tree.bound[0]])
            if time < x_values[regression_tree.left_bound + regression_tree.bound[0]]:
                regression_tree = regression_tree.left_child
            else:
                regression_tree = regression_tree.right_child

    plt.show()
    return (regression_tree.expected_val, mean_absolute_error(y_train, train_values), mean_absolute_error(y_test, test_values))
    # return (regression_tree.expected_val, mean_absolute_error(y_values, pred_values))

time = 2010.3475394579
answer = reg_tree("processed/regions/REGION IV-A (CALABARZON)/Laguna/Palay.csv", time)

print("The expected value of the crop's yield at year " + str(time) + " is " + str(answer[0]) + ".\nTrain Error: " + str(answer[1]) + "\nTest Error: " + str(answer[2]))
# print("The expected value of the crop's yield at year " + str(time) + " is " + str(answer[0]) + " with MAE " + str(answer[1]))
