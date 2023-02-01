import numpy as np  
import json
  
# Opening JSON file
f = open('C:/Users/pwawszczyk/Desktop/fl-bias-mitigation/FCFL_impl/FUEL/data/adult/test/mytest.json')
  
# returns JSON object as a dictionary
data = json.load(f)
client = 'phd'

print(np.average(np.array(data['user_data'][client]['x'])[:,57]*np.array(data['user_data'][client]['y'])))
print(np.average(np.array(data['user_data'][client]['x'])[:,57]))
print(np.array(data['user_data'][client]['x'])[:,57].shape)
