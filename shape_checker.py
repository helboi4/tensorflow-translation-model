import tensorflow as tf
import einops


class ShapeChecker():
    #Initialise storage of unique shapes for future reference
    def __init__(self):
        self.shapes = {}

    #Allows us to call object of the class directly like a function
    #tensor == the data being checked
    #names == a string that descrives the expected shape
    def __call__(self, tensor, names, broadcast=False):
        #This checks if the tensor already has concrete properties. We abort if not bc we ned those
        if not tf.executing_eagerly():
            return
    
        #Parse the shape according to the expected pattern.
        #Extracts dimensions of the tensor and returns a dict with names as keys, dimensions as values
        parsed = einops.parse_shape(tensor, names)
        
        #Loop through each name and dimension key-value pair in the above dict
        for name, new_dim in parsed.items():
            #Check for existence of the key in stored shapes dict
            old_dim = self.shapes.get(name, None)

            #Broadcasting allows dimensions of size 1 to match any size, so no further checks are needed
            if(broadcast and new_dim == 1):
                continue

            #If a dimension hasn't been assigned to this name yet, it should be stored
            if old_dim is None:
                self.shapes[name] = new_dim
                continue

            #If there is a dimension stored for the name but it does not match the new dimension,
            #that means there is a shape mismatch and we raise an error
            if new_dim != old_dim:
                raise ValueError(f"""
                    Shape mismatch for dimension: '{name}'
                        found: {new_dim}
                        expected: {old_dim}
                """)

