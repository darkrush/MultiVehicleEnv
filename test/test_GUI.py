from MultiVehicleEnv.GUI import GUI
import argparse


parser = argparse.ArgumentParser(description="GUI for Multi-VehicleEnv")
parser.add_argument('--dir',type=str,default='/dev/shm/gui.data')
parser.add_argument('--fps',type=int,default=24)
args = parser.parse_args()

GUI_instance = GUI(dir = args.dir, fps = args.fps)
GUI_instance.connect()
GUI_instance.spin()