# 노드 생성
class Node():
    def __init__(self, item):
        self.data = item
        self.prev = None
        self.next = None
# 양방향 연결리스트
class DoublyLinkedList:
    def __init__(self):
        self.nodeCount = 0
        self.head = Node(None)
        self.tail = Node(None)
        self.head.prev = None
        self.head.next = self.tail
        self.tail.prev = self.head
        self.tail.next = None

    def __repr__(self):
        if self.nodeCount == 0:
            return 'LinkedList: empty'
        s = ''
        curr = self.head
        while curr.next.next:
            curr = curr.next
            s += repr(curr.data)
            if curr.next.next is not None:
                s += ' -> '
        return s

    def getLength(self):
        return self.nodeCount
    # 리스트 순회 (빈 리스트는 진행 X)
    def traverse(self):
        result = []
        curr = self.head
        while curr.next.next:
            curr = curr.next
            result.append(curr.data)
        return result
    # 리스트 역순회 (빈 리스트는 진행 X)
    def reverse(self):
        result = []
        curr = self.tail
        while curr.prev.prev:
            curr = curr.prev
            result.append(curr.data)
        return result
    # pos번째 노드 (특정 원소 얻기)
    def getAt(self, pos):
        if pos < 0 or pos > self.nodeCount:
            return None
        # 효율성 극대화 위해 절반으로 나누어 이분탐색 진행
        # 노드 수 절반 기준 tail에 가까우므로 curr을 tail에 놓고 탐색
        if pos > self.nodeCount // 2:
            i = 0
            curr = self.tail
            while i < self.nodeCount - (pos - 1):
                curr = curr.prev
                i += 1
        # 노드 수 기준 head에 가까우므로 head에 놓고 탐색 진행
        else:
            i = 0
            curr = self.head
            while i < pos:
                curr = curr.next
                i += 1
        return curr
    # 특정 노드 이후에 원소 삽입
    def insertAfter(self, prev, newNode):
        next = prev.next
        newNode.prev = prev
        newNode.next = next
        prev.next = newNode
        next.prev = newNode
        self.nodeCount += 1
        return True
    # 특정 위치에 노드 삽입
    def insertAt(self, pos, newNode):
        if pos < 1 or pos > self.nodeCount + 1:
            return False
        prev = self.getAt(pos - 1)
        return self.insertAfter(prev, newNode)
    # 역방향 순회하며 노드 삽입
    def insertBefore(self, next, newNode):
        prev = next.prev
        newNode.prev = prev
        newNode.next = next
        prev.next = newNode
        next.prev = newNode
        self.nodeCount += 1
        return True
    # 이전 노드 다음 노드 삭제
    def popAfter(self, prev):
        if prev.next == self.tail:
            return None
        remove_node = prev.next
        next = remove_node.next
        prev.next = next
        next.prev = prev
        self.nodeCount -= 1
        return remove_node.data
    # next 이전 노드 삭제
    def popBefore(self, next):
        if next.prev == self.head:
            return None
        remove_node = next.prev
        prev = remove_node.prev
        prev.next = next
        next.prev = prev
        self.nodeCount -= 1
        return remove_node.data
    # 특정(pos번째) 위치 노드 삭제
    def popAt(self, pos):
        if pos < 1 or pos > self.nodeCount:
            return False
        prev = self.getAt(pos - 1)
        return self.popAfter(prev)
    # 양방향 리스트 병합
    def concat(self, L):
        self.tail.prev.next = L.head.next
        L.head.next.prev = self.tail.prev
        if L.tail:
            self.tail = L.tail
        self.nodeCount += L.nodeCount
# 링크드리스트 큐 생성
class LinkedListQueue:
    def __init__(self):
        self.data = DoublyLinkedList()

    def size(self):
        return self.data.getLength()
    def isEmpty(self):
        return self.size() == 0
    def enqueue(self, item):
        node = Node(item)
        self.data.insertAt(self.size()+1, node)
    def dequeue(self):
        self.data.popAt(1)
    def peek(self):
        self.data.getAt(1).data

def solution(x):
    return 0


li = [1, 2, 3]
li[3] = 4
print(li)