import numpy as np


class circBuffer:
    """
        ---
        Circular buffer implementation with Numpy.
        ---
        `len_max`:   maximum element of the buffer (buffer size). Only integer is accepted.

        `data_type`: data type of the buffer. (np.float32, np.complex64, etc.)
        
    """
    def __init__(self,len_max = int(1e6),data_type = np.float32):
        self.len_max   = len_max                                   
        self.data_type = data_type
        self.buffer    = np.zeros(self.len_max,dtype=self.data_type)
        self.RD_ptr   = 0                         
        self.WRT_ptr  = 0
        self.elements_in_buffer = 0


    def WRT(self,x):
        """
            ---
            write method
            ---
            `x`: Numpy array, input data

            Please check if the length of `x` does not exceed the size of buffer. Checking this is not included in the method for performance concerns.
        """
        len_x = len(x)
        self.elements_in_buffer += len_x
        if (len_x + self.WRT_ptr) < self.len_max:
            self.buffer[self.WRT_ptr:(self.WRT_ptr+len_x)] = x
            self.WRT_ptr += len_x

        else:
            len_x1 = self.len_max - self.WRT_ptr
            len_x2 = len_x - len_x1
            self.buffer[-len_x1:] = x[:len_x1]
            self.buffer[:len_x2]  = x[len_x1:]
            self.WRT_ptr = len_x2

    def RD(self,len_x):
        """
            ---
            read method
            ---
            `len_x`: length of the data to be read. Only integer vallues are valid.

            Please check if the `len_x` does not exceed the size of buffer. Checking this is not included in the method for performance concerns.
        """
        self.elements_in_buffer -= len_x
        if (len_x + self.RD_ptr) < self.len_max:
            x = self.buffer[self.RD_ptr:(self.RD_ptr+len_x)]
            self.RD_ptr += len_x
        else:
            x = np.zeros(len_x,dtype=self.data_type)
            len_x1 = self.len_max - self.RD_ptr
            len_x2 = len_x - len_x1

            self.buffer[-len_x1:] = x[:len_x1]
            self.buffer[:len_x2]  = x[len_x1:]
            self.RD_ptr = len_x2
        return x

    def is_RD_available(self,len_x):
        """
            ---
            Checks if `RD()` is valid.
            ---
            `len_x`: length of the data to be read. Only integer vallues are valid.

            It checks if the buffer contains adequate number of elements stored.
        """
        if self.elements_in_buffer >= len_x:
            return True
        else:
            return False
    
    def is_WRT_available(self,x):
        """
            ---
            Checks if `WRT()` is valid.
            ---
            `x`: Numpy array, input data

            It prevents the `WRT()` method from overwriting on the existing data in the buffer.
        """
        if (self.len_max - self.elements_in_buffer) >= len(x):
            return True
        else:
            return False

    def reset_buffer(self,mode = "WRT"):
        """
            ---
            Method for reseting the buffer.
            ---
            This method does not erase the buffer elements. Instead, it reinitializes the reading and writing pointers.
        """
        if mode == "WRT":
            self.WRT_ptr = self.RD_ptr
        if mode == "RD":
            self.RD_ptr = self.WRT_ptr
        self.elements_in_buffer = 0

    def extend_buffer(self,len_extension):
        """
            ---
            Method for increasing the size of the buffer.
            ---
            It inserts the new part just after the writer point.
        """
        self.buffer = np.concatenate((self.buffer[:(self.WRT_ptr+1)], np.zeros(len_extension, dtype = self.data_type), self.buffer[(self.WRT_ptr+1):]))
        if self.WRT_ptr<self.RD_ptr:
            self.RD_ptr += len_extension
