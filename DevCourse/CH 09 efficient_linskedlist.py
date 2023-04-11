# 노드 생성
class Node:
	def __init__(self, item):
		self.data = item
		self.next = None

# dummy node 헤드에 포함한 연결리스트
class LinkedList:
	def __init__(self):
		self.nodeCount = 0
		self.head = Node(None)
		self.tail = None
		self.head.next = self.tail

	def __repr__(self):
		if self.nodeCount == 0:
			return 'LinkedList: empty'
		s = ''
		curr = self.head
		while curr.next:
			curr = curr.next
			s += repr(curr.data)
			if curr.next is not None:
				s += ' -> '
		return s

	def getLength(self):
		return self.nodeCount

	def traverse(self):
		result = []
		curr = self.head
		while curr.next:
			curr = curr.next
			result.append(curr.data)
		return result

	def getAt(self, pos):
		if pos < 0 or pos > self.nodeCount:
			return None
		i = 0
		curr = self.head
		while i < pos:
			curr = curr.next
			i += 1
		return curr

	def insertAfter(self, prev, newNode):
		newNode.next = prev.next
		# tail인 경우 newNode가
		if prev.next is None:
			self.tail = newNode
		prev.next = newNode
		self.nodeCount += 1
		return True

	def insertAt(self, pos, newNode):
		if pos < 1 or pos > self.nodeCount + 1:
			return False
		if pos != 1 and pos == self.nodeCount + 1:
			prev = self.tail
		else:
			prev = self.getAt(pos - 1)
		return self.insertAfter(prev, newNode)

	def concat(self, L):
		self.tail.next = L.head.next
		if L.tail:
			self.tail = L.tail
		self.nodeCount += L.nodeCount

	def popAfter(self, prev):
		if prev.next is None:
			return None
		curr = prev.next
		prev.next = curr.next
		if self.tail == curr:
			self.tail = prev
		self.nodeCount -= 1
		return curr.data

	def popAt(self, pos):
		if pos < 1 or pos > self.nodeCount:
			return False
		prev = self.getAt(pos - 1)
		return self.popAfter(prev)