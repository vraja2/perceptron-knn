import sys 
import csv
import math

class PerceptronClassification:
    def __init__(self):
        #weight vector for each class. each vector will be 28x28 and initialized to 0
        self.weight_vectors = {0:[],1:[]}
        self.epochs = 6
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
        for i in range(0,2):
            weight_vector = []
            for a in range(0,70):
                weight_vector.append([])
                for b in range(0,60):
                    #every pixel in the weight vector is initialized to 0
                    weight_vector[a].append(0)
            self.weight_vectors[i] = weight_vector
        #print self.weight_vectors


    def init_confusion_matrix(self):
        for l in range(0,2):
            self.confusion_matrix.append([0,0])

    def max_class(self, feature_vector):
        #iterate through each class
        max_sum = float("-inf")
        max_class = -1
        for i in range(0,2):
            dot_product_sum = 0
            for l in range(0,70):
                for w in range(0,60):
                    dot_product_sum += feature_vector[l][w]*self.weight_vectors[i][l][w]
            if dot_product_sum > max_sum:
                max_sum = dot_product_sum
                max_class = i
        return max_class

    def update_weight(self, incorrect_label, correct_label, feature_vector):
        for l in range(0,70):
            for w in range(0,60):
                self.weight_vectors[int(incorrect_label)][l][w] -= self.alpha*feature_vector[l][w] 
                self.weight_vectors[int(correct_label)][l][w] += self.alpha*feature_vector[l][w]

    def train(self):
        tmp = 1
        while tmp < self.epochs:
            self.alpha /= tmp
            training_labels = open('./facedata/facedatatrainlabels', 'r')
            training_images = open('./facedata/facedatatrain', 'r')
            with training_images as ti:
                data = list(csv.reader(ti))
                data = [i for i in data if i]
            count = 0
            coord = 0
            for label in training_labels:
                #print tmp
                l = 0
                feature_vector = []
                while l < 70:
                    feature_vector.append([])
                    coord = count + l
                    w = 0
                    while w < 60:
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
                count += 70      
            print self.weight_vectors
            tmp += 1

    def test(self):
        testing_images = open('./facedata/facedatatest', 'r')
        with testing_images as ti:
            data = list(csv.reader(ti))
            data = [i for i in data if i]
        count = 0
        for j in range(0,152):
            if count >= 10500:
                break
            feature_vector = []
            for l in range(0,70):
                feature_vector.append([])
                coord = count + l
                for w in range(0,60):
                    if data[coord][0][w] == "+" or data[coord][0][w] == "#":
                        feature_vector[l].append(1)                
                    else:
                        feature_vector[l].append(-1)
            decision = self.max_class(feature_vector) 
            count += 70
            self.solutions.append(decision)
    
    def evaluate(self):
        test_labels = open('./facedata/facedatatestlabels', 'r')
        self.init_confusion_matrix()
        i = 0
        class_stats = {0:[0,0], 1:[0,0]}
        total_correct = 0
        num_labels = 150
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
        for l in range(0,2):
            for w in range(0,2):
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
