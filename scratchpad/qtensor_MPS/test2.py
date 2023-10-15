def helper():
    pass
         
# Driver Code
if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        n, k = [int(i) for i in input().split()]
        a = [int(i) for i in input().split()]
        b = [int(i) for i in input().split()]

        for elem in a:
            if elem not in map:
                map[elem] = 1
            map[elem] += 1
        
        n = map.keys() #distinct elem in a

        cnt = 0 # elements in b not in b
        temp = []
        
        for elem in b:
            if elem not in a:
                cnt += 1
        
        if cnt > k:
            print()

    print(a)
