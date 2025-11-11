class Vehicle:
    def __init__(self, plate_number, owner_name, owner_phone):
        self.plate_number = plate_number
        self.owner_name = owner_name
        self.owner_phone = owner_phone


class ParkingSpot:
    def __init__(self, spot_number):
        self.spot_number = spot_number
        self.vehicles = []
        self.occupied = False  # 新增状态属性

    def add_vehicle(self, vehicle):
        if len(self.vehicles) < 2:
            self.vehicles.append(vehicle)
            if len(self.vehicles) == 2:
                self.occupied = True  # 两辆车时标记为占用
            return True
        return False

    def remove_vehicle(self, plate_number):
        for vehicle in self.vehicles:
            if vehicle.plate_number == plate_number:
                self.vehicles.remove(vehicle)
                if len(self.vehicles) < 2:
                    self.occupied = False  # 少于两辆车时标记为不占用
                return True
        return False

    def is_occupied(self):
        return self.occupied

    def get_vehicle_info(self):
        if not self.vehicles:
            return f"{self.spot_number}号车位没有绑定任何车"
        info = []
        for vehicle in self.vehicles:
            info.append(f"{self.spot_number}  {vehicle.plate_number}  {vehicle.owner_name}  {vehicle.owner_phone}")
        return "\n".join(info)


class Garage:
    def __init__(self):
        self.parking_spots = [ParkingSpot(i) for i in range(1, 4)]

    def add_vehicle(self, plate_number, owner_name, owner_phone, spot_number):
        if spot_number < 1 or spot_number > len(self.parking_spots):
            return "车位号无效"
        vehicle = Vehicle(plate_number, owner_name, owner_phone)
        if self.parking_spots[spot_number - 1].add_vehicle(vehicle):
            return f"成功向{spot_number}号车库添加车辆: {plate_number}"
        return f"{spot_number}号车位已绑定2辆车，达到上限"

    def display_all_spots(self):
        result = "车位号  车牌号  车主名  车主电话\n"
        result += "==================================================\n"
        for spot in self.parking_spots:
            result += spot.get_vehicle_info() + "\n"
        return result

    def search_vehicle(self, plate_number):
        for spot in self.parking_spots:
            for vehicle in spot.vehicles:
                if vehicle.plate_number == plate_number:
                    return (spot.spot_number, vehicle.owner_name, vehicle.owner_phone)
        return None

    def can_enter(self, spot_number):
        if self.parking_spots[spot_number - 1].is_occupied():
            return f"{spot_number}号车位已占用，不能进入车库"
        return f"{spot_number}号车位可以进入"


def main():
    garage = Garage()
    while True:
        print("**************************************************")
        print("欢迎使用【车库管理系统】V1.0")
        print("1. 添加车辆")
        print("2. 显示全部车位信息")
        print("3. 查询车辆信息")
        print("4. 检查车位是否可以进入")
        print("0. 退出系统")
        print("**************************************************")
        choice = input("请选择操作功能：")

        if choice == "1":
            plate_number = input("请输入车牌号：")
            owner_name = input("请输入姓名：")
            owner_phone = input("请输入电话：")
            spot_number = int(input("请输入车位号："))
            print(garage.add_vehicle(plate_number, owner_name, owner_phone, spot_number))

        elif choice == "2":
            print(garage.display_all_spots())

        elif choice == "3":
            plate_number = input("请输入要搜索的车牌号：")
            result = garage.search_vehicle(plate_number)
            if result:
                spot_number, owner_name, owner_phone = result
                print(f"车牌号  车位号  车主名  车主电话\n----------------------------------------")
                print(f"{plate_number}  {spot_number}  {owner_name}  {owner_phone}")
                action = input("请选择要执行的操作：1.修改 2.删除 0.返回上级菜单")
                if action == "2":
                    if garage.parking_spots[spot_number - 1].remove_vehicle(plate_number):
                        print(f"已删除{spot_number}号车位中车牌号为{plate_number}的车辆")
                    else:
                        print("未找到该车辆")
            else:
                print("未找到该车辆")

        elif choice == "4":
            spot_number = int(input("请输入要检查的车位号："))
            print(garage.can_enter(spot_number))

        elif choice == "0":
            print("感谢使用车库管理系统，再见！")
            break

        else:
            print("无效选择，请重试。")


if __name__ == "__main__":
    main()
