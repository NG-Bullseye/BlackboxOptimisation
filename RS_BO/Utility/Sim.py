import re
from RS_BO.Utility import load_real_data
from collections import namedtuple


class Sim:
    # Define the namedtuple
    def __init__(self):
        self.yaw_acc_mapping ,self.yaw_vec_mapping,self.yaw_list= load_real_data.get_real_data_from_db()
        self.YawData = namedtuple("YawData", ["acc", "vec"])
        # Initialize the combined dictionary
        yaw_data = {}

        # Iterate over the keys in the yaw_acc_mapping dictionary
        for yaw in self.yaw_acc_mapping:
            # Check if the key also exists in the yaw_vec_mapping dictionary
            if yaw in self.yaw_vec_mapping:
                # If it does, create a new YawData namedtuple with the acc and vec values
                yaw_data[yaw] = self.YawData(self.yaw_acc_mapping[yaw], self.yaw_vec_mapping[yaw])
    def sample(self,yaw_string):
        self.yaw_acc_mapping.get(self.extract_float_number(yaw_string))
    def extract_float_number(self,yaw_string):
        # Define a regular expression pattern to match the float number
        pattern = r"[-+]?\d+(\.\d+)?"
        # Use re.search to find the first match of the pattern in the input_string
        match = re.search(pattern, yaw_string)
        if match:
            # If a match is found, extract the matched float number and convert it to a float
            float_number = float(match.group())
            return float_number
        else:
            # If no match is found, return None or raise an exception based on your requirement.
            return None
