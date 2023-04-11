# 스택 생성
class ArrayStack:
	# 빈 리스트 생성
	def __init__(self):
		self.data = []
	# 길이 구하기
	def size(self):
		return len(self.data)
	# 비어있나 여부 확인
	def isEmpty(self):
		return self.size() == 0
	# 스택에 값 추가
	def push(self, item):
		self.data.append(item)
	# 스택에서 값 제거 (제일 뒤부터)
	def pop(self):
		return self.data.pop()
	# 제일 뒤 원소 찾기
	def peek(self):
		return self.data[-1]

class LinkedListStack:
	# 양방향 연결리스트로 스택 생성
	def __init__(self):
		self.data = DoublyLinkedList()
	# 스택 길이 구하기
	def size(self):
		return self.data.getLength()
	# 비어있나 여부 확인
	def isEmpty(self):
		return self.size() == 0
	# 제일 뒤에 원소 추가 (append 대신 함수 적용)
	def push(self, item):
		node = Node(item)
		self.data.insertAt(self.size() + 1, node)
	# 제일 뒤 원소 뽑기 (pop대신 함수 구현)
	def pop(self):
		return self.data.popAt(self.size())
	# 제일 뒤에 위치한 원소 찾기
	def peek(self):
		return self.data.getAt(self.size()).data
class ArrayStack:
    def __init__(self):
        self.data = []
    def size(self):
        return len(self.data)
    def isEmpty(self):
        return self.size() == 0
    def push(self, item):
        self.data.append(item)
    def pop(self):
        return self.data.pop()
    def peek(self):
        return self.data[-1]


def splitTokens(exprStr):
    tokens = []
    val = 0
    valProcessing = False
    for c in exprStr:
        if c == ' ':
            continue
        if c in '0123456789':
            val = val * 10 + int(c)
            valProcessing = True
        else:
            if valProcessing:
                tokens.append(val)
                val = 0
            valProcessing = False
            tokens.append(c)
    if valProcessing:
        tokens.append(val)
    return tokens

def infixToPostfix(tokenList):
    prec = {
        '*': 3,
        '/': 3,
        '+': 2,
        '-': 2,
        '(': 1,
    }

    opStack = ArrayStack()
    postfixList = []
    for i in tokenList:
        if type(i) is int:
            postfixList.append(i)
        elif i == '(':
            opStack.push(i)
        elif i == ')':
            while opStack.peek() != '(':
                postfixList.append(opStack.pop())
            opStack.pop()
        else:
            while not opStack.isEmpty() and prec[opStack.peek()] >= prec[i]:
                postfixList.append(opStack.pop())
            opStack.push(i)
    while not opStack.isEmpty():
        postfixList.append(opStack.pop())
    return postfixList


def postfixEval(tokenList):
    valStack = ArrayStack()
    for i in tokenList:
        if type(i) is int:
            valStack.push(i)
        else:
            b = valStack.pop()
            a = valStack.pop()
            if i == '*':
                valStack.push(a*b)
            elif i == '/':
                valStack.push(a/b)
            elif i == '+':
                valStack.push(a+b)
            elif i == '-':
                valStack.push(a-b)
    if valStack.size() >= 1:
        return valStack.pop()
    else:
        return None

def solution(expr):
    tokens = splitTokens(expr)
    postfix = infixToPostfix(tokens)
    print(postfix)
    val = postfixEval(postfix)
    return val