import os

def output(set_img: int):

    """
    Configures the input and output folder paths based on the selected image set and ensures the output folder exists.

    This function sets the paths for two data folders (representing input views) and an output folder based on the 
    `set_img` value. It creates the specified output folder if it does not already exist.

    Args:
        set_img (int): An integer specifying which image set to use. 
                       - 1: First sequence
                       - 2: Second sequence
                       - Any other value: Third sequence
    """

    # Define input and obj folders
    set_img = 1

    if set_img == 1: # third sequence
        output_folder = "outputs1"
        data_folder_1 = "data/view1"  # Folder containing input frames
        data_folder_2 = "data/view2"  # Folder containing input frames
    if set_img == 2: # second sequence
        output_folder = "outputs2"
        data_folder_1 = "data/view3"  # Folder containing input frames
        data_folder_2 = "data/view4"  # Folder containing input frames
    else: # third sequence
        output_folder = "outputs3"
        data_folder_1 = "data/view5"  # Folder containing input frames
        data_folder_2 = "data/view6"  # Folder containing input frames

    os.makedirs(output_folder, exist_ok=True)  # Create the obj folder if it doesn't exist

    return data_folder_1, data_folder_2