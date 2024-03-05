import numpy as np

def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    # Calculate the scale and zero point for quantization.
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0

    # Ensure that the zero point is within the valid range.
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point

def quantize_tensor(x, num_bits=3, min_val=None, max_val=None):
    # Quantize a tensor (x) to a fixed number of bits.

    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x = np.round(np.clip(q_x, qmin, qmax))

    return q_x, min_val, max_val

class quantize:
    def __init__(self, query_config, min_val=None, max_val=None):
        # Initialize the quantization class with configuration and optional min/max values.

        self.num_bits = query_config['bit']
        self.min_val = min_val
        self.max_val = max_val

    def write(self, data):
        # Quantize and return the quantized data for a write operation.

        # Calculate min_val and max_val if not provided explicitly.
        if not self.min_val and not self.max_val:

            # Check if there are any NaN values
            has_nan = np.isnan(data).any()

            if has_nan:
                self.min_val = np.nanmin(data)
                self.max_val = np.nanmax(data)               
            else:
                # Find the minimum and maximum values
                self.min_val = np.min(data)
                self.max_val = np.max(data)

        q_data, _, _ = quantize_tensor(data, num_bits=self.num_bits, min_val=self.min_val, max_val=self.max_val)

        return q_data
    
    def query(self, data):
        # Quantize and return the quantized data for a query operation.

        # Ensure that query uses the same min_val and max_val as write.
        q_data, _, _ = quantize_tensor(data, num_bits=self.num_bits, min_val=self.min_val, max_val=self.max_val)
        return q_data
