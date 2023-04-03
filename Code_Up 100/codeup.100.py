### 6001
# print('Hello')

### 6002
# print('Hello World')

### 6003
# print('Hello\nWorld')

### 6004
# print("'Hello'")

### 6005
# print('"Hello World"')

### 6006
# print("\"!@#$%^&*()'")

### 6007
# print('"C:\\Download\\\'hello\'.py"')

### 6008
# print("print(\"Hello\\nWorld\")")

### 6009
'''
n = input()
print(n)
'''

### 6010
'''
n = int(input())
print(n)
'''

### 6011
'''
f = float(input())
print(f)
'''

### 6012
'''
a = int(input())
b = int(input())
print(a + "\n" + b)
'''

### 6013
'''a = input()
b = input()
print(b + "\n" + a)
'''
### 6014
'''
f = input()
print(3 * (f + "\n"))
'''

### 6015
'''
a, b = input().split()
print(a, "\n"+ b)
'''

### 6016
'''
a, b = input().split()
print(b, a)
'''

### 6017
'''
sen = input()
print(3 * (sen + " "))
'''

### 6018
'''
a, b = input().split(':')
print(a, b, sep=':')
'''

### 6019
'''
y,m,d = input().split('.')
print(d,m,y, sep='-')
'''

### 6021
'''
front, back = input().split('-')
print(front+back)
'''

### 6022
'''
word = input()
for i in word:
    print(i)
'''

### 6023
'''
t,m,h = input().split(':')
print(m)
'''

### 6024
'''
w1, w2 = input().split()
print(w1+w2)
'''

### 6025
'''n1, n2 = map(int,input().split())
print(n1+n2)
'''

### 6026
'''
f1 = float(input())
f2 = float(input())
print(f1+f2)


### 6027 10진수 >> 16진수
n = int(input())
print("%x" %n)

### 6028 10진수 >> 16진수 대문자
n = int(input())
print("%X" %n)

### 6029 16진수 >> 8진수
n = input()
h = int(n,16)
print("o" %h)

### 6030 영문자 1개 >> 10진수
c = ord(input()) # ord : 문자 >> 아스키코드
print(c)

### 6031 정수 >> 유니코드 문자변환
n = int(input())
print(chr(n))   # chr : 아스키코드 >> 문자 

### 6032 정수 1개 > 부호변환
n = int(input())
print(-n)

### 6033 문자 1개입력 >>> 다음문자출력
s = ord(input())
print(chr(s+1))

### 6034
a,b = map(int,input().split())
print(a-b)

### 6035
a,b = map(float,input().split())
print(a*b)

### 6036
word, n = input().split()
print(word * int(n))

### 6037
n = int(input())
sen = input()
print(n*sen)

### 6038
a, b = map(int,input().split())
print(a**b)

### 6039
a,b = map(float,input().split())
print(a**b)

### 6040
a, b = map(int,input().split())
print(a//b)

### 6041
a,b = map(int,input().split())
print(a%b)

### 6042
n = float(input())
print(format(n, ".2f"))

### 6043
a,b = map(float,input().split())
print(format(a/b, ".3f"))

### 6044
a, b = map(int, input().split())
print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)
print(format(a/b, ".2f"))

### 6045
a,b,c = map(int,input().split())
print(a+b+c, format((a+b+c)/3, ".2f"))

### 6046 비트 시프트연산 2배로 출력
a = int(input())
print(a<<1)

### 6047
a,b = map(int,input().split())
print(a<<b)

### 6048
a,b = map(int,input().split())
if a < b :
    print(True)
else:
    print(False)

### 6049
a, b = input().split()
if int(a) == int(b):
    print(True)
else:
    print(False)

### 6050
a, b = input().split()
if int(a) <= int(b):
    print(True)
else:
    print(False)

### 6051
a, b = input().split()
print(int(a) != int(b))

### 6052
n = int(input())
print(bool(n))

### 6053
n = int(input())
print(not(bool(n)))

### 6054
a, b= input().split()
a = bool(int(a))
b = bool(int(b))
print(a and b)

### 6055
a, b= input().split()
a = bool(int(a))
b = bool(int(b))
print(a or b)

### 6056 두 정수의 참 / 거짓이 다를떄만 True 출력
a, b = input().split()
a = bool(int(a))
b = bool(int(b))
print((a and (not b)) or ((not a) and b))

### 6057 두 정수의 참 거짓이 같은 경우 True
a, b = input().split()
a = bool(int(a))
b = bool(int(b))
print((a and b) or ((not a) and (not b)))

### 6058
a, b = input().split()
a = bool(int(a))
b = bool(int(b))
print((not a) and (not b))

### 6059 비트 단위 NOT
n = int(input())
print(~n)

### 6060 비트단위 AND
a, b = input().split()
result = int(a) & int(b)
print(result)

### 6061
a, b = input().split()
result = int(a) | int(b)
print(result)

### 6062
a, b = input().split()
result = int(a) ^ int(b)
print(result)

### 6063
a, b = input().split()
result = int(a) if int(a) > int(b) else int(b)
print(result)

### 6064
a, b, c = input().split()
a = int(a)
b = int(b)
c = int(c)
result = min(a, b, c)
print(result)

### 6065
data = input().split()
for i in data:
    if(int(i) % 2 == 0):
        print(int(i))

### 6066
data = input().split()
for i in data:
    if(int(i) % 2 == 0):
        print('even')
    else:
        print('odd')

### 6067
data = int(input())
if(data % 2 == 0 and data < 0):
    print('A')
elif(data % 2 != 0 and data < 0):
    print('B')
elif(data % 2 == 0 and data > 0):
    print('C')
elif(data % 2 != 0 and data > 0):
    print('D')

### 6068
score = int(input())
if(score >= 90):
    print('A')
elif(score >= 70):
    print('B')
elif(score >= 40):
    print('C')
elif(score >= 0):
    print('D')

### 6069
grade = input()
if(grade == 'A'):
    print('best!!!')
elif(grade == 'B'):
    print('good!!')
elif(grade == 'C'):
    print('run!')
elif(grade == 'D'):
    print('slowly~')
else:
    print('what?')

### 6070
month = int(input())
season = ''
if(month in [12, 1, 2]):
    season = 'winter'
elif(month in [3, 4, 5]):
    season = 'spring'
elif(month in [6, 7, 8]):
    season = 'summer'
else:
    season = 'fall'
print(season)

### 6071
while True:
    num = int(input())
    if(num == 0):
        False
    else:
        print(num)

### 6072
num = int(input())
while(num > 0):
    print(num)
    num -= 1

### 6073
num = int(input())
while(num > 0):
    num -= 1
    print(num)

### 6074
a = input()
count = 0
while(count <= ord(a)-97):
    print(chr(97+count), end=' ')
    count += 1

### 6075
n = int(input())
for i in range(n + 1):
    print(i)

### 6076
n = int(input())
for i in range(n + 1):
    print(i)

### 6077
n = int(input())
s = 0
for i in range(n + 1):
    if(i % 2 == 0):
        s += i
print(s)

### 6078
c = ''
while(c != 'q'):
    c = input()
    print(c)

### 6079
n = int(input())
i = 0
s = 0
while(s < n):
	i += 1
	s += i
print(i)

### 6080
n, m = map(int, input().split())
for i in range(1, n + 1):
    for j in range(1, m + 1):
        print('{} {}'.format(i, j))

### 6081
n = input()
for i in range(1, 16):
	print('%x*%x=%x'.upper() %(int(n, 16), int(hex(i), 16), (int(n, 16) * int(hex(i), 16))))

### 6082
n = int(input())
for i in range(1, n + 1):
    if(i % 10 == 3 or i % 10 == 6 or i % 10 == 9):
        print('X', end = ' ')
    else:
        print(i, end = ' ')

### 6083
a, b, c = map(int, input().split())
count = 0
for i in range(a):
	for j in range(b):
		for k in range(c):
			print('{} {} {}'.format(i, j, k))
			count += 1
print(count)

### 6084
h, b, c, s = map(int, input().split())
mb = round((h * b * c * s / 8) / 1024 / 1024, 1)
print('{} MB'.format(mb))

#6085
w, h, b = map(int, input().split())
mb = round(((w*h*b) / 8 / 1024 / 1024), 2)
print('{:.2f} MB'.format(mb))

### 6086
n = int(input())
total = 0
for i in range(1, n + 1):
    total += i
    if(total >= n):
        break
print(total)

### 6087
n = int(input())
for i in range(1, n + 1):
    if(i % 3 == 0):
        pass
    else:
        print(i)

### 6088
a, b, n = map(int, input().split())
total = a
for i in range(a, a + n - 1):
	total += b
print(total)

### 6089
a, b, n = map(int, input().split())
total = a
for i in range(a, a + n - 1):
	total *= b
print(total)

### 6090
a, m, d, n = map(int, input().split())
total = a
for i in range(a, a + n - 1):
	total = total * m + d
print(total)

### 6091
a, b, c = map(int, input().split())
d = 1
while d % a != 0 or d % b != 0 or d % c != 0:
    d += 1
print(d)
'''
'''
### 6092
import sys
input = sys.stdin.readline

n = int(input())
num_list = list(map(int,input().split()))

cnt = []
for i in range(24):
    cnt.append(0)
for i in range(n):
    cnt[num_list[i]] += 1
for i in range(1,24):
    print(cnt[i], end=' ')
'''
### 6093
'''
n = int(input())
num_li = list(map(int, input().split()))
for i in range(n-1,-1,-1):
    print(num_li[i], end = ' ')
'''

### 6094
'''
n = int(input())
num_li = list(map(int, input().split()))

cnt = num_li[0]
for i in range(n):
    if cnt > num_li[i] :
        cnt = num_li[i]
print(cnt)
'''

# 코드업 100제 6095
'''
import sys
input = sys.stdin.readline

n = int(input())
matrix = [[0 for i in range(20)]for j in range(20)]

for _ in range(n):
    x, y = map(int, input().split())
    matrix[x][y] = 1
for i in range(1,20):
    for j in range(1,20):
        print(matrix[i][j], end = ' ')
    print()

# 6096 코드업
import sys
input = sys.stdin.readline

matrix = [[]*19 for _ in range(19)]
for i in range(19):
    matrix[i] = list(map(int, input().split()))

n = int(input())
for i in range(n):
    x,y = map(int,input().split())
    for j in range(19):
        if matrix[x-1][j] == 1:
            matrix[x-1][j] = 0
        else :
            matrix[x-1][j] = 1
    for j in range(19):
        if matrix[j][y-1] == 1:
            matrix[j][y-1] = 0
        else:
            matrix[j][y-1] = 1
for i in range(19):
    for j in range(19):
        print(matrix[i][j], end = ' ')
    print()
'''

### 6097 코드업
'''
h, w = map(int, input().split())
n = int(input())
matrix = [[0]*w for _ in range(h)]

for i in range(n):
    l,d,x,y = map(int, input().split())
    for j in range(l):
        if d == 0:
            matrix[x-1][y-1+j] = 1
        else :
            matrix[x-1+j][y-1] = 1
for i in range(h):
    for j in range(w):
        print(matrix[i][j], end = ' ')
    print()
'''
### 6098 코드업

import sys
input = sys.stdin.readline

matrix = [[]*10 for i in range(10)]
for i in range(10):
    matrix[i] = list(map(int,input().split()))

x = 1
y = 1

matrix[x][y] = 0
while True:
    if matrix[x][y] == 2:
        matrix[x][y] = 9
        break
    if matrix[x][y+1] != 1:
        matrix[x][y] = 9
        y += 1
    else :
        if matrix[x+1][y] != 1:
            matrix[x][y] = 9
            x += 1
        else :
            matrix[x][y] = 9
            break
for i in range(10):
    for j in range(10):
        print(matrix[i][j], end = ' ')
    print()
