import sys 
import csv
import math

class PerceptronClassification:
    def __init__(self):
        #weight vector for each class. each vector will be 28x28 and initialized to 0
        self.weight_vectors = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        self.epochs = 9
        self.alpha = 0.3
        self.init_weight_vectors()
        self.train()
        self.solutions = []
        self.test()
        print self.solutions
        self.confusion_matrix = []
    
    def init_weight_vectors(self):
        """
        Initializes each of the weight vectors for the classes to 0. Each vector is 28x28
        """
        for i in range(0,10):
            weight_vector = []
            for a in range(0,28):
                weight_vector.append([])
                for b in range(0,28):
                    #every pixel in the weight vector is initialized to 0
                    weight_vector[a].append(0)
            self.weight_vectors[i] = weight_vector
        #print self.weight_vectors


    def init_confusion_matrix(self):
        for l in range(0,10):
            self.confusion_matrix.append([0,0,0,0,0,0,0,0,0,0])

    def max_class(self, feature_vector):
        #iterate through each class
        max_sum = float("-inf")
        max_class = -1
        for i in range(0,10):
            dot_product_sum = 0
            for l in range(0,28):
                for w in range(0,28):
                    dot_product_sum += feature_vector[l][w]*self.weight_vectors[i][l][w]
            if dot_product_sum > max_sum:
                max_sum = dot_product_sum
                max_class = i
        return max_class

    def update_weight(self, incorrect_label, correct_label, feature_vector):
        for l in range(0,28):
            for w in range(0,28):
                self.weight_vectors[int(incorrect_label)][l][w] -= self.alpha*feature_vector[l][w] 
                self.weight_vectors[int(correct_label)][l][w] += self.alpha*feature_vector[l][w]

    def train(self):
        tmp = 1
        while tmp < self.epochs:
            self.alpha /= tmp
            training_labels = open('./digitdata/traininglabels', 'r')
            training_images = open('./digitdata/trainingimages', 'r')
            with training_images as ti:
                data = list(csv.reader(ti))
                data = [i for i in data if i]
            count = 0
            coord = 0
            for label in training_labels:
                #print tmp
                l = 0
                feature_vector = []
                while l < 28:
                    feature_vector.append([])
                    coord = count + l
                    w = 0
                    while w < 28:
                        int_label = int(label)
                        if data[coord][0][w] == "+" or data[coord][0][w] == "#":
                            feature_vector[l].append(1)
                        else:
                            feature_vector[l].append(-1)
                        w += 1
                    l += 1
                decision = self.max_class(feature_vector)
                if decision != label:
                    self.update_weight(decision, label, feature_vector)
                count += 28      
            print self.weight_vectors
            tmp += 1

    def test(self):
        testing_images = open('./digitdata/testimages', 'r')
        with testing_images as ti:
            data = list(csv.reader(ti))
            data = [i for i in data if i]
        count = 0
        for j in range(0,1000):
            feature_vector = []
            for l in range(0,28):
                feature_vector.append([])
                coord = count + l
                for w in range(0,28):
                    if data[coord][0][w] == "+" or data[coord][0][w] == "#":
                        feature_vector[l].append(1)                
                    else:
                        feature_vector[l].append(-1)
            decision = self.max_class(feature_vector) 
            count += 28
            self.solutions.append(decision)
    
    def evaluate(self):
        test_labels = open('./digitdata/testlabels', 'r')
        self.init_confusion_matrix()
        i = 0
        class_stats = {0:[0,0], 1:[0,0], 2:[0,0], 3:[0,0], 4:[0,0], 5:[0,0], 6:[0,0], 7:[0,0], 8:[0,0], 9:[0,0]}
        total_correct = 0
        num_labels = 1000
        for label in test_labels:
            int_label = int(label)
            if int_label == self.solutions[i]:
                class_stats[int_label][0] += 1
                self.confusion_matrix[int_label][self.solutions[i]] += 1
            else:
                self.confusion_matrix[int_label][self.solutions[i]] += 1
            class_stats[int_label][1] += 1
            i += 1
        for k in class_stats:
            print "Class " + str(k) + ": " + str(float(class_stats[k][0])/class_stats[k][1])
            total_correct += float(class_stats[k][0])
        print "Overall Accuracy: " + str(total_correct/num_labels) 
        for l in range(0,10):
            for w in range(0,10):
                self.confusion_matrix[l][w] = float(self.confusion_matrix[l][w]) / class_stats[l][1]
        
        s = [[str(e) for e in row] for row in self.confusion_matrix]
        lens = [len(max(col, key=len)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print '\n'.join(table)
        #self.print_confusion_matrix() 




def main():
    pc = PerceptronClassification()
    pc.evaluate()

if __name__ == "__main__":
    main()
