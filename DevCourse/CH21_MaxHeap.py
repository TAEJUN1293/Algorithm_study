# MaxHeap
class MaxHeap:
    def __init__(self):
        self.data = [None]

    def insert(self, item):
        self.data.append(item)
        idx = len(self.data) - 1
        while idx > 1 and self.data[idx] > self.data[idx//2]:
            self.data[idx], self.data[idx//2] = self.data[idx//2], self.data[idx]
            idx //= 2

def solution(x):
    return 0