def roll_lon(data):
    if data.lon.min() < 0:
        # roll from -180, 180 to 0, 360
        rolled = data.assign_coords(lon=(data.lon % 360)).sortby("lon")
    else:
        # roll from 0, 360 to -180, 180
        rolled = data.assign_coords(lon=((data.lon + 180) % 360) - 180).sortby("lon")    
    return rolled