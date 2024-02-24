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

# for loop