while True:
    number = input().rstrip()
    if number == 'X' or number == 'x':
        break
    ab = int(number[:2])
    cd = int(number[2:])
    if ab == cd:
        print('=')
    elif ab > cd:
        print('> ', '두 변수 값의 차이 : ', ab-cd)
    elif ab < cd:
        print('< ', '두 변수 값의 차이 : ', cd-ab)