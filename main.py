from __future__ import print_function
import numpy as np
import numpy.random as ran


def unique_vals(rows, col):
    return set([row[col] for row in rows])


def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
        self.prob = 0

    def set_prob(self, prob):
        self.prob = prob

    def get_prob(self):
        return self.prob

    def match(self, example):
        val = example[self.column]
        return val == self.value

    def match_classifier(self, example, prob):
        val = example[self.column]
        if val == "NaN":
            p = ran.random()
            if p < prob:
                return True
            else:
                return False
        return val == self.value


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    question.set_prob(len(true_rows) / (len(true_rows) + len(false_rows)))
    return true_rows, false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain < 0.05:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match_classifier(row, node.question.get_prob()):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def delete_data(rows, prob):
    file_deleted = rows
    n_features = len(rows[0]) - 1
    for row in file_deleted:
        for col in range(n_features):
            p = 0.01*prob
            if ran.random() < p:
                row[col] = "NaN"
    return file_deleted


def main():
    train = np.loadtxt(fname="chess_train.txt", dtype="str", delimiter=",")
    test = np.loadtxt(fname="chess_test.txt", dtype="str", delimiter=",")

    dt = build_tree(train)
#    for row in test:
#        print("Actual: %s. Predicted: %s" %
#              (row[-1], print_leaf(classify(row, dt))))

    test = delete_data(test, 50)
    print(test)
    for row in test:
        print("Actual: %s. Predicted: %s" %
              (row[-1], print_leaf(classify(row, dt))))


if __name__ == '__main__':
    main()


