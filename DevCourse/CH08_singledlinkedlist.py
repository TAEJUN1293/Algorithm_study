# 노드 생성
class Node:
    def __init__(self, item):
        self.data = item
        self.next = None
# single LinkedList 생성
class LinkedList:
    def __init__(self):
        self.nodeCount = 0
        self.head = None
        self.tail = None
    # 리스트를 print로 출력할 때
    def __repr__(self):
        if self.nodeCount == 0:
            return 'LinkedList: empty'
        s = ''
        curr = self.head
        while curr is not None:
            s += repr(curr.data)
            if curr.next is not None:
                s += ' -> '
            curr = curr.next
        return s

    def getAt(self, pos):
        if pos < 1 or pos > self.nodeCount:
            return None
        i = 1
        curr = self.head
        while i < pos:
            curr = curr.next
            i += 1
        return curr

    def insertAt(self, pos, newNode):
        if pos < 1 or pos > self.nodeCount + 1:
            return False
        # newNode 위치가 Head인 경우
        if pos == 1:
            newNode.next = self.head
            self.head = newNode
        else:
            # newNode 위치가 tail인 경우
            if pos == self.nodeCount + 1:
                prev = self.tail
            else:
                prev = self.getAt(pos - 1)
            newNode.next = prev.next
            prev.next = newNode
        if pos == self.nodeCount + 1:
            self.tail = newNode
        self.nodeCount += 1
        return True
    # 연결리스트 길이 얻기
    def getLength(self):
        return self.nodeCount
    # 연결리스트를 리스트로 변환 후 원소 추가
    def traverse(self):
        result = []
        curr = self.head
        while curr is not None:
            result.append(curr.data)
            curr = curr.next
        return result
    # 두 리스트 연결
    def concat(self, L):
        self.tail.next = L.head
        if L.tail:
            self.tail = L.tail
        self.nodeCount += L.nodeCount

    def popAt(self, pos):
        # 주어진 범위 아니면 False
        if pos < 1 or pos > self.nodeCount:
            return False
        # 삭제할 노드가 Head면
        if pos == 1:
            remove_node = self.head
            self.head = self.head.next
            self.nodeCount -= 1
            if not self.nodeCount:
                self.tail = None
            return remove_node.data
        # Head가 아닌 경우
        else:
            prev = self.getAt(pos - 1)
            remove_node = prev.next
            prev.next = remove_node.next
            if pos == self.nodeCount:
                self.tail = prev
            self.nodeCount -= 1
            return remove_node.data
