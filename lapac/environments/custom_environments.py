from lapac.environments.car_racing import ModifiedCarRacing

def load(environment_name):
    if environment_name == "CarRacing-v0":
        return ModifiedCarRacing()
    raise NotImplementedError
