# 8.26上课内容
"""
print("Hello World!")
print("你好！"*10)
"""
'''
qq_number="123456789"
qq_password="123"

print(type(qq_password))  #type()用来测试变量类型
print(qq_number)
print(qq_password)

price=8.5
weight=7.5
money=price*weight
print(money)
money=money-5
print(money)
print(type(money))
'''

'''
gender=False
print(gender)
print(type(gender))
'''


# 8.29上课内容
'''
price=eval(input("请输入价格"))      # eval可执行算术式计算结果
weight=float(input("请输入重量"))    # 利用强制字符类型转化
print(type(price))
money=price*weight
print(money)
'''

'''
age = float(input("请输入年龄"))
if age >= 18:
    print("你已经成年，欢迎进网吧 Happy")
    print("欢迎欢迎，热烈欢迎！")
else:
    print("丨，你还没有成年，请回家写作业吧")
'''

'''
age = float(input("请输入年龄"))
if 18<= age <=40:
    print("你已经成年，欢迎进网吧 Happy")
    print("欢迎欢迎，热烈欢迎！")
else:
    print("丨，你还没有成年或者太老了，请回家写作业吧")
'''


# 9.2上课内容

'''
def say_hello():
    print("hello 1")
    print("hello 2")
    print("hello 3")
say_hello()
'''

# 9.5上课
'''name_list = ["wangs", "zhangsan", "wlisi"]
temp_list = ["黑马楼", "黑悟空", "弼马温"]
name_list.extend(temp_list)
print(name_list)
print(temp_list)
t = len(name_list)
print(t)
z = temp_list.count("黑悟空")
print(z)
name_list.append("沙和尚")
print(name_list)
# name_list.clear()
# print(name_list)
name_list.sort(reverse=True)
print(name_list)
name_list.reverse()
print(name_list)
for lang in name_list:
    print(lang)'''

Chinese_A = {"刘德华", "张学友", "张曼玉", "钟楚红", "古天乐", "林青霞"}
Math_A = {"林青霞", "郭富城", "王祖贤", "刘德华", "张曼玉", "黎明"}
print(Chinese_A & Math_A)
print(Chinese_A | Math_A)
print(Chinese_A ^ Math_A)
print(Chinese_A - Math_A)

Color_A={' 白', '红', '黑', '蓝', '绿', '黄'}















