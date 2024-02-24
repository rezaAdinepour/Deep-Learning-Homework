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