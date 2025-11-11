'''
play = int(input("请选择出石头（1）/剪刀（2）/布（3）"))
computer = int(1)
if play == computer:
    print("平局")
elif (play == 1 and computer == 2) or (play == 2 and computer == 3) or (play == 3 and computer == 1):
    print("玩家胜利")
elif (play == 2 and computer == 1) or (play == 3 and computer == 2) or (play == 1 and computer == 3):
    print("电脑胜利")
'''  # 最死板且单机的猜拳程序一

'''

play = int(input("请选择出石头（1）/剪刀（2）/布（3）"))
computer = int(1)
if play == computer:
    print("平局")
elif (play == 1 and computer == 2) or (play == 2 and computer == 3) or (play == 3 and computer == 1):
    print("玩家胜利")
else:
    print("电脑胜利")
'''  # 改进版单机猜拳程序二


'''import random
play = int(input("请选择出石头（1）/剪刀（2）/布（3）"))
computer = random.randint(1,3)
print (computer)
if play == computer:
    print("平局")
elif (play == 1 and computer == 2) or (play == 2 and computer == 3) or (play == 3 and computer == 1):
    print("玩家胜利")
else:
    print("电脑胜利")
  '''# 改进版非单机猜拳程序三
import sys
print(sys.executable)