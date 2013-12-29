import sys 
import csv
import math
import operator
import time 

class Profile:
    def __init__(self, label):
        self.foreground_counts = []
        self.label = label

class KNN:
    def __init__(self, k):
        self.k = k
        self.profiles = [] 
        self.build_profiles()
        self.solutions = []
        self.confusion_matrix = []
        self.classify()

    
    def init_confusion_matrix(self):
        for l in range(0,10):
            self.confusion_matrix.append([0,0,0,0,0,0,0,0,0,0])

    def build_profiles(self):
        training_labels = open('./digitdata/traininglabels', 'r')
        training_images = open('./digitdata/trainingimages', 'r')
        count = 0   
        with training_images as ti:
            data = list(csv.reader(ti))
            data = [i for i in data if i]
        for label in training_labels:
            int_label = int(label)
            new_profile = Profile(int_label)
            l = 0
            while l < 28:
                new_profile.foreground_counts.append([])
                coord = count + l
                w = 0
                while w < 28:
                    if data[coord][0][w] == "+" or data[coord][0][w] == "#":
                       new_profile.foreground_counts[l].append(1)
                    else:
                        new_profile.foreground_counts[l].append(0)
                    w += 1
                l += 1
            self.profiles.append(new_profile)
            count += 28 

    def compute_distance(self, foreground_counts_1, foreground_counts_2):
        distance = 0
        for l in range(0,28):
            for w in range(0,28):
                distance += abs(foreground_counts_1[l][w] - foreground_counts_2[l][w])
        return distance

    def classify(self):
        testing_images = open('./digitdata/testimages', 'r')
        with testing_images as ti:
            data = list(csv.reader(ti))
            data = [i for i in data if i]
        count = 0
        #loop through all the test images
        for j in range(0,1000):
            print "Iteration: " + str(j)
            local_foreground_counts = []
            classification_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}  
            for l in range(0,28):
                local_foreground_counts.append([])
                coord = count + l
                for w in range(0,28):
                    if data[coord][0][w] == "+" or data[coord][0][w] == "#":
                        local_foreground_counts[l].append(1)
                    else:
                        local_foreground_counts[l].append(0)
            count += 28
            label_distances = []
            for profile in self.profiles:
                label_distances.append( (profile.label, self.compute_distance(profile.foreground_counts, local_foreground_counts)) )
            label_distances = sorted(label_distances, key=lambda x: x[1]) 
            for i in range(0,self.k):
                classification_dict[label_distances[i][0]] += 1
            sorted_dict = sorted(classification_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
            print sorted_dict[0][0]
            self.solutions.append(sorted_dict[0][0])
    
    def evaluate_classifications(self):
        """
        Evaluates the accuracy of the digit classification. Builds a confusion matrix
        """
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
    knn = KNN(4)
    print knn.solutions
    knn.evaluate_classifications()
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print time.time() - start_time, "seconds"
