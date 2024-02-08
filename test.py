import random
import glob

anomaly_source_path = 'utils'
a = glob.glob(anomaly_source_path + f"/*.sh")
print(a)
a = glob.glob(anomaly_source_path + f"/*.py")
print(a)