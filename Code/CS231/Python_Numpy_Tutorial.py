import numpy as np




# variable type
x = 3
print(type(x))
print(x)
print(x + 2, x - 2, x * 2, x ** 2)
print(-x)
print(complex(x))
print('-'*20)


# boolean type
t = True
f = False
print(type(t))
print(t and f)
print(t and not(f))
print(t or f)
print(t != f) # logical xor
print('-'*20)

# strings
hello = "hello"
world = "world!"
print(hello, world)
print(len(hello))
print(hello[-1])
hw = "%s %s %d" %(hello, world, 12)
print(hw)
print('-'*20)



# string functions
s = "hello"
print(s.capitalize())
print(s.upper())
print(s.rjust(5))
print(s.center(7))
print(s.replace('e', '(ee)'))
print('  world  '.strip())
print('-'*20)


# lists
xs = [1, 2, 3]
print(xs, xs[2])
print(xs[-1])
xs[0] = "fp"
print(xs)
xs.append("hello")
print(xs)
xs.pop()
print(xs)
print('-'*20)


nums = list(range(5))
print(nums)
print(nums[2:4])
print(nums[2:])
print(nums[:2])
print(nums[:])
print(nums[:-1])
print(nums[-1:])
nums[2:4] = [8, 5]
print(nums)
print('-'*20)


# for loop
animals = ["cat", "dog", "monkey"]
for animal in animals:
    print(animal)
print('-'*20)



for idx, animal in enumerate(animals):
    print("#%d: %s" %(idx + 1, animal)) 
print('-'*20)


nums = [1, 2, 3]
square = []
for i in nums:
    square.append(i ** 2)
print(square)
print('-'*20)



nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)
print('-'*20)



dic = {"cat": "cute", "dog": "furry"}
print(dic)
print(dic["cat"])
print("cat" in dic)

dic["cat"] = "shit"
print(dic["cat"])

dic["horse"] = "biautiful"
print(dic["horse"])
print(dic.get("horse", "N/A"))

del dic["horse"]
print(dic.get("horse", "N/A"))


d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print("A %s has %d legs" %(animal, legs))

print('-'*20)

for animal, legs in d.items():
    print("A %s has %d legs" %(animal, legs))
print('-'*20)



nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)
print('-'*20)
