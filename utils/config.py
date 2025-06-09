class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def __getitem__(self, item):
        return self.__dict__[item]
    
    def get_dict(self):
        return_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value = value.get_dict()
            return_dict[key] = value
        return return_dict